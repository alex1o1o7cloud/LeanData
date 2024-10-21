import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_holes_remain_l1078_107895

/-- Given a total number of holes and a percentage of filled holes, 
    calculate the number of unfilled holes. -/
def unfilled_holes (total : ℕ) (percent_filled : ℚ) : ℕ :=
  total - (total : ℚ) * percent_filled |>.floor.toNat

/-- Theorem stating that with 8 total holes and 75% filled, 2 holes remain unfilled -/
theorem two_holes_remain : unfilled_holes 8 (75 / 100) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_holes_remain_l1078_107895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_in_interval_l1078_107875

open Set Real

def β : Set ℝ := {α | ∃ k : ℤ, α = (2 * k * π / 3) - (π / 6)}

theorem angles_in_interval : 
  Finset.card (Finset.filter (λ α => 0 ≤ α ∧ α < 2 * π) (Finset.image (λ k => (2 * ↑k * π / 3) - (π / 6)) (Finset.range 4))) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_in_interval_l1078_107875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_iteration_theorem_f₁_similar_to_square_f₂_similar_to_rational_f₃_similar_to_double_f₄_similar_to_double_l1078_107816

noncomputable section

/-- A function is similar to another function through a bridge function -/
def IsSimilar (f g φ : ℝ → ℝ) : Prop :=
  ∀ x, f x = (φ⁻¹ ∘ g ∘ φ) x

/-- The nth iteration of a function -/
def Iterate (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (Iterate f n)

/-- Function 1: f(x) = x² / (2x - 1) -/
noncomputable def f₁ (x : ℝ) : ℝ := x^2 / (2*x - 1)

/-- Function 2: f(x) = x / ∛(1 + ax^k), where k is a positive integer -/
noncomputable def f₂ (a : ℝ) (k : ℕ+) (x : ℝ) : ℝ := x / (1 + a*x^(k:ℕ))^(1/(k:ℝ))

/-- Function 3: f(x) = 2x² - 1, for -1 ≤ x ≤ 1 -/
noncomputable def f₃ (x : ℝ) : ℝ := 2*x^2 - 1

/-- Function 4: f(x) = 4x(1-x), for 0 ≤ x ≤ 1 -/
noncomputable def f₄ (x : ℝ) : ℝ := 4*x*(1-x)

/-- The main theorem to prove -/
theorem nth_iteration_theorem :
  ∀ (n : ℕ) (x : ℝ),
    (Iterate f₁ n) x = x^(2^n) / (x^(2^n) - (x-1)^(2^n)) ∧
    (∀ (a : ℝ) (k : ℕ+), (Iterate (f₂ a k) n) x = x / (1 + a*n*x^(k:ℕ))^(1/(k:ℝ))) ∧
    (Iterate f₃ n) x = Real.cos (2^n * Real.arccos x) ∧
    (Iterate f₄ n) x = (Real.sin (2^n * Real.arcsin (Real.sqrt x)))^2 := by
  sorry

/-- Proof that f₁ is similar to g₁(x) = x² through φ₁(x) = 1 - 1/x -/
theorem f₁_similar_to_square :
  IsSimilar f₁ (λ x ↦ x^2) (λ x ↦ 1 - 1/x) := by
  sorry

/-- Proof that f₂ is similar to g₂(x) = x / (1 + ax) through φ₂(x) = x^k -/
theorem f₂_similar_to_rational (a : ℝ) (k : ℕ+) :
  IsSimilar (f₂ a k) (λ x ↦ x / (1 + a*x)) (λ x ↦ x^(k:ℕ)) := by
  sorry

/-- Proof that f₃ is similar to g₃(x) = 2x through φ₃(x) = arccos(x) -/
theorem f₃_similar_to_double :
  IsSimilar f₃ (λ x ↦ 2*x) Real.arccos := by
  sorry

/-- Proof that f₄ is similar to g₄(x) = 2x through φ₄(x) = arcsin(√x) -/
theorem f₄_similar_to_double :
  IsSimilar f₄ (λ x ↦ 2*x) (λ x ↦ Real.arcsin (Real.sqrt x)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_iteration_theorem_f₁_similar_to_square_f₂_similar_to_rational_f₃_similar_to_double_f₄_similar_to_double_l1078_107816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carries_actual_speed_l1078_107864

-- Define the given conditions
noncomputable def jerry_speed : ℝ := 40
noncomputable def jerry_time : ℝ := 0.5  -- 30 minutes in hours
noncomputable def beth_extra_distance : ℝ := 5
noncomputable def beth_extra_time : ℝ := 1/3  -- 20 minutes in hours
noncomputable def carrie_distance_multiplier : ℝ := 2
noncomputable def carrie_stated_speed : ℝ := 60
noncomputable def carrie_time : ℝ := 7/6  -- 1 hour and 10 minutes in hours

-- Define the theorem
theorem carries_actual_speed (ε : ℝ) (h : ε > 0) :
  ∃ (actual_speed : ℝ), 
    abs (actual_speed - (carrie_distance_multiplier * jerry_speed * jerry_time) / carrie_time - 34.29) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carries_actual_speed_l1078_107864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_bus_capacity_l1078_107869

/-- The capacity of the train in number of people -/
def train_capacity : ℕ := 120

/-- The number of buses -/
def num_buses : ℕ := 2

/-- The capacity of one bus as a fraction of the train's capacity -/
def bus_capacity_fraction : ℚ := 1 / 6

/-- Theorem: The combined capacity of the two buses is 40 people -/
theorem combined_bus_capacity :
  (↑num_buses * (↑train_capacity * bus_capacity_fraction)).floor = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_bus_capacity_l1078_107869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1078_107879

-- Define the line l
def line_l : ℝ → ℝ → Prop := λ x y ↦ x - 2*y + 3 = 0

-- Define the given line
def given_line : ℝ → ℝ → Prop := λ x y ↦ 2*x + y - 1 = 0

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∃ m1 m2 : ℝ, (∀ x y, l1 x y ↔ y = m1*x + (0:ℝ)) ∧ 
              (∀ x y, l2 x y ↔ y = m2*x + (0:ℝ)) ∧
              m1 * m2 = -1

theorem line_equation : 
  (line_l 1 2) ∧                        -- Line l passes through point P(1,2)
  (perpendicular line_l given_line) →   -- Line l is perpendicular to 2x+y-1=0
  ∀ x y, line_l x y ↔ x - 2*y + 3 = 0   -- The equation of line l is x-2y+3=0
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1078_107879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calligraphy_supplies_pricing_and_discounts_l1078_107893

-- Define the variables
variable (x y : ℝ)  -- x: unit price of brush, y: unit price of rice paper

-- Define the conditions
def condition1 (x y : ℝ) : Prop := 40 * x + 100 * y = 280
def condition2 (x y : ℝ) : Prop := 30 * x + 200 * y = 260

-- Define the discount schemes
def schemeA (m : ℝ) : ℝ := 0.4 * m + 280
def schemeB (m : ℝ) : ℝ := 0.32 * m + 316

-- Theorem statement
theorem calligraphy_supplies_pricing_and_discounts :
  ∀ x y : ℝ, condition1 x y ∧ condition2 x y →
  x = 6 ∧ y = 0.4 ∧
  (∀ m : ℝ, m > 200 →
    (m < 450 → schemeA m < schemeB m) ∧
    (m = 450 → schemeA m = schemeB m) ∧
    (m > 450 → schemeA m > schemeB m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calligraphy_supplies_pricing_and_discounts_l1078_107893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logo_scaling_theorem_l1078_107806

/-- Represents the dimensions of a logo -/
structure LogoDimensions where
  width : ℝ
  height : ℝ

/-- Scales a logo proportionally to a new width -/
noncomputable def scaleLogo (original : LogoDimensions) (newWidth : ℝ) : LogoDimensions :=
  let scale := newWidth / original.width
  { width := newWidth, height := original.height * scale }

theorem logo_scaling_theorem (original : LogoDimensions) 
    (h1 : original.width = 3) 
    (h2 : original.height = 2) :
  let posterLogo := scaleLogo original 12
  let badgeLogo := scaleLogo original 1.5
  posterLogo.height = 8 ∧ badgeLogo.height = 1 := by
  sorry

#check logo_scaling_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logo_scaling_theorem_l1078_107806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1078_107828

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := Real.sqrt x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 1 / (2 * Real.sqrt x)

-- Define the tangent line
def tangent_line (a : ℝ) (x : ℝ) : ℝ := f' a * (x - a) + f a

-- Define the x-intercept of the tangent line
def x_intercept (a : ℝ) : ℝ := a

-- Define the y-intercept of the tangent line
def y_intercept (a : ℝ) : ℝ := Real.sqrt a / 2

-- Define the area of the triangle
def triangle_area (a : ℝ) : ℝ := (1 / 2) * x_intercept a * y_intercept a

end noncomputable section

-- The main theorem
theorem tangent_triangle_area (a : ℝ) (h1 : a > 0) (h2 : triangle_area a = 2) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1078_107828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_theorem_l1078_107894

/-- Calculates the average speed of a car given its uphill and downhill speeds and distances. -/
noncomputable def average_speed (uphill_speed uphill_distance downhill_speed downhill_distance : ℝ) : ℝ :=
  let total_distance := uphill_distance + downhill_distance
  let total_time := uphill_distance / uphill_speed + downhill_distance / downhill_speed
  total_distance / total_time

/-- Theorem stating that the average speed of a car traveling 100 km uphill at 30 km/hr
    and 50 km downhill at 60 km/hr is approximately 36.06 km/hr. -/
theorem car_average_speed_theorem :
  let uphill_speed : ℝ := 30
  let uphill_distance : ℝ := 100
  let downhill_speed : ℝ := 60
  let downhill_distance : ℝ := 50
  abs (average_speed uphill_speed uphill_distance downhill_speed downhill_distance - 36.06) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_theorem_l1078_107894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_theorem_l1078_107802

/-- Represents the chemical reaction between HCl and AgNO3 -/
structure Reaction where
  hcl_moles : ℚ
  agno3_moles : ℚ
  volume : ℚ
  molar_mass_agcl : ℚ

/-- Calculates the mass of AgCl formed and the concentration of HNO3 produced -/
noncomputable def reaction_results (r : Reaction) : ℚ × ℚ :=
  let agcl_mass := r.hcl_moles * r.molar_mass_agcl
  let hno3_concentration := r.hcl_moles / r.volume
  (agcl_mass, hno3_concentration)

theorem reaction_theorem (r : Reaction) 
  (h1 : r.hcl_moles = 3)
  (h2 : r.agno3_moles = 3)
  (h3 : r.volume = 1)
  (h4 : r.molar_mass_agcl = 143.32) :
  reaction_results r = (429.96, 3) := by
  sorry

#check reaction_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_theorem_l1078_107802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_integers_l1078_107822

theorem existence_of_integers : ∃ (x y z : ℕ+), 
  (4 * Real.sqrt (Real.rpow 7 (1/3) - Real.rpow 2 (1/3)) = Real.rpow x (1/3) + Real.rpow y (1/3) - Real.rpow z (1/3)) ∧
  (x + y + z : ℝ) = 254 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_integers_l1078_107822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_multiplication_error_l1078_107819

theorem student_multiplication_error (x y : ℕ) : 
  (10 * (x / 10) + 5) * y = 4500 ∧ 
  (10 * (x / 10) + 3) * y = 4380 → 
  x = 75 ∧ y = 60 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_multiplication_error_l1078_107819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_for_beta_f_value_for_alpha_l1078_107882

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.cos x) ^ 2

-- Theorem for the range of f(β)
theorem f_range_for_beta : 
  ∀ β : ℝ, β ∈ Set.Icc 0 (Real.pi / 2) → 
  f β ∈ Set.Icc (-2) 1 := by
  sorry

-- Theorem for the value of f(α)
theorem f_value_for_alpha (α : ℝ) (h : Real.tan α = 2 * Real.sqrt 3) : 
  f α = 10 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_for_beta_f_value_for_alpha_l1078_107882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l1078_107803

def b : ℕ → ℕ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | n + 1 => b n + 3 * n + 1

theorem b_50_value : b 50 = 3727 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_value_l1078_107803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_one_sufficient_not_necessary_l1078_107883

def z₁ (m : ℝ) : ℂ := (m^2 + m + 1) + (m^2 + m - 4) * Complex.I
def z₂ : ℂ := 3 - 2 * Complex.I

theorem m_one_sufficient_not_necessary :
  (∃ m : ℝ, z₁ m = z₂ ∧ m ≠ 1) ∧ (z₁ 1 = z₂) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_one_sufficient_not_necessary_l1078_107883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_location_l1078_107836

theorem angle_location (x : ℝ) : 
  (Real.sin x * Real.tan x < 0) → 
  (x % (2 * Real.pi) ∈ Set.Icc (Real.pi / 2) Real.pi ∪ Set.Ioc Real.pi (3 * Real.pi / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_location_l1078_107836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_work_days_l1078_107865

/-- The number of days it takes for A to complete the work alone -/
noncomputable def x : ℝ := 30

/-- The number of days it takes for B to complete the work alone -/
def b_days : ℝ := 15

/-- The number of days A and B work together -/
def work_together : ℝ := 5

/-- The total number of days to complete the work -/
def total_days : ℝ := 20

/-- The portion of work completed by A and B together in the first 5 days -/
noncomputable def work_done_together : ℝ := work_together * (1/x + 1/b_days)

/-- The portion of work completed by A alone in the remaining days -/
noncomputable def work_done_alone : ℝ := (total_days - work_together) * (1/x)

/-- Theorem stating that A can complete the work alone in 30 days -/
theorem a_work_days : x = 30 := by
  have h1 : work_done_together + work_done_alone = 1 := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_work_days_l1078_107865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_cuts_surface_area_l1078_107897

theorem cube_cuts_surface_area :
  ∀ (cube_volume : ℝ) (first_cut_height : ℝ) (second_cut_height : ℝ),
    cube_volume = 1 →
    first_cut_height = 1/4 →
    second_cut_height = 1/6 →
    let total_height : ℝ := 1
    let bottom_slab_height : ℝ := total_height - (first_cut_height + second_cut_height)
    let reassembled_surface_area : ℝ :=
      2 * (1 * 1) +  -- front and back
      2 * (1 * 1) +  -- sides
      2 * (3 * 1)    -- top and bottom of three slabs
    reassembled_surface_area = 10 := by
  intros cube_volume first_cut_height second_cut_height h1 h2 h3
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_cuts_surface_area_l1078_107897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_differential_z_l1078_107823

-- Define the function z implicitly
noncomputable def z (x y : ℝ) : ℝ :=
  Real.sqrt (2*x*y)

-- State the theorem for the total differential of z
theorem total_differential_z (x y : ℝ) (h : z x y ≠ 0) :
  ∃ (dz dx dy : ℝ), dz = (y / z x y) * dx + (x / z x y) * dy :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_differential_z_l1078_107823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_circumcenters_form_rhombus_l1078_107815

-- Define the points
variable (A B C D E P Q R S : EuclideanSpace ℝ (Fin 2))

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (B - A) = (C - D) ∧ (D - A) = (C - B)

-- Define the intersection point E of diagonals AC and BD
def is_intersection_point (E A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, E = A + t • (C - A) ∧ ∃ s : ℝ, E = B + s • (D - B)

-- Define the circumcenter of a triangle
def is_circumcenter (P A B E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (dist P A = dist P B) ∧ (dist P B = dist P E) ∧ (dist P E = dist P A)

-- Define a rhombus
def is_rhombus (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (dist P Q = dist Q R) ∧ (dist Q R = dist R S) ∧ (dist R S = dist S P) ∧ (dist S P = dist P Q)

-- Theorem statement
theorem parallelogram_circumcenters_form_rhombus 
  (h1 : is_parallelogram A B C D)
  (h2 : is_intersection_point E A B C D)
  (h3 : is_circumcenter P A B E)
  (h4 : is_circumcenter Q B C E)
  (h5 : is_circumcenter R C D E)
  (h6 : is_circumcenter S A D E) :
  is_rhombus P Q R S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_circumcenters_form_rhombus_l1078_107815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_II_must_be_true_l1078_107839

def Digit := Fin 10

structure Envelope :=
  (digit : Digit)

def Statement (e : Envelope) : Fin 5 → Prop
  | 0 => e.digit = ⟨5, by norm_num⟩
  | 1 => e.digit ≠ ⟨6, by norm_num⟩
  | 2 => e.digit = ⟨7, by norm_num⟩
  | 3 => e.digit ≠ ⟨8, by norm_num⟩
  | 4 => e.digit ≠ ⟨9, by norm_num⟩

theorem statement_II_must_be_true (e : Envelope) :
  (∃ (a b c : Fin 5), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    Statement e a ∧ Statement e b ∧ Statement e c ∧
    (∀ d : Fin 5, d ≠ a ∧ d ≠ b ∧ d ≠ c → ¬Statement e d)) →
  Statement e 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_II_must_be_true_l1078_107839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_g_symmetry_axis_correct_propositions_l1078_107867

-- Define the functions as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.cos ((2/3)*x + (7*Real.pi)/2)
noncomputable def g (x : ℝ) : ℝ := (1/2) * Real.sin (5*x + (7*Real.pi)/8)

-- State the theorems
theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

theorem g_symmetry_axis : 
  ∀ x, g (Real.pi/8 + x) = g (Real.pi/8 - x) := by sorry

-- Main theorem combining both results
theorem correct_propositions : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, g (Real.pi/8 + x) = g (Real.pi/8 - x)) := by
  constructor
  · exact f_is_odd
  · exact g_symmetry_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_g_symmetry_axis_correct_propositions_l1078_107867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bouncing_ball_height_l1078_107880

/-- The initial height of a bouncing ball -/
noncomputable def initial_height : ℝ := 1.8

/-- The ratio of the height the ball reaches after each bounce -/
noncomputable def bounce_ratio : ℝ := 0.8

/-- The total distance covered by the ball -/
noncomputable def total_distance : ℝ := 9

/-- The sum of an infinite geometric series -/
noncomputable def geometric_sum (a r : ℝ) : ℝ := a / (1 - r)

theorem bouncing_ball_height :
  geometric_sum initial_height bounce_ratio = total_distance :=
by
  -- Unfold the definitions
  unfold geometric_sum initial_height bounce_ratio total_distance
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bouncing_ball_height_l1078_107880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1078_107801

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 6) - 2 * Real.cos x

-- Define the domain
def domain : Set ℝ := { x | Real.pi / 2 ≤ x ∧ x ≤ Real.pi }

theorem f_properties :
  ∀ x ∈ domain,
    (Real.sin x = 4 / 5 → f x = (4 * Real.sqrt 3 + 3) / 5) ∧
    (1 ≤ f x ∧ f x ≤ 2) ∧
    (∀ y ∈ domain, f (2 * Real.pi / 3 - (y - 2 * Real.pi / 3)) = f y) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1078_107801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_arrangement_l1078_107898

theorem unique_digit_arrangement : ∃! (a b c d e f g h : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
   f ≠ g ∧ f ≠ h ∧
   g ≠ h) ∧
  (a ∈ (Finset.range 9).filter (· ≠ 8) ∧
   b ∈ (Finset.range 9).filter (· ≠ 8) ∧
   c ∈ (Finset.range 9).filter (· ≠ 8) ∧
   d ∈ (Finset.range 9).filter (· ≠ 8) ∧
   e ∈ (Finset.range 9).filter (· ≠ 8) ∧
   f ∈ (Finset.range 9).filter (· ≠ 8) ∧
   g ∈ (Finset.range 9).filter (· ≠ 8) ∧
   h ∈ (Finset.range 9).filter (· ≠ 8)) ∧
  (10 * a + b : ℚ) / (10 * c + d) = (10 * e + f) - (10 * g + h) ∧
  (10 * a + b : ℚ) / (10 * c + d) = 8 ∧
  (10 * e + f) - (10 * g + h) = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_arrangement_l1078_107898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tasty_pair_iff_isogonal_conjugate_l1078_107890

-- Define the triangle ABC
variable (A B C : EuclideanPlane) 

-- Define that ABC is acute and scalene
variable (h_acute : IsAcute A B C)
variable (h_scalene : IsScalene A B C)

-- Define a point P in the plane
variable (P : EuclideanPlane)

-- Define the function to get T_A given Q
noncomputable def get_T_A (Q : EuclideanPlane) : EuclideanPlane :=
  sorry

-- Define T_B and T_C similarly
noncomputable def get_T_B (Q : EuclideanPlane) : EuclideanPlane :=
  sorry

noncomputable def get_T_C (Q : EuclideanPlane) : EuclideanPlane :=
  sorry

-- Define the property of P having an isogonal conjugate
def has_isogonal_conjugate (P : EuclideanPlane) : Prop :=
  sorry

-- Define the property of T_A, T_B, T_C being on the circumcircle
def on_circumcircle (T_A T_B T_C : EuclideanPlane) : Prop :=
  sorry

-- State the theorem
theorem tasty_pair_iff_isogonal_conjugate :
  (∃ Q : EuclideanPlane, on_circumcircle (get_T_A Q) (get_T_B Q) (get_T_C Q)) ↔
  has_isogonal_conjugate P :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tasty_pair_iff_isogonal_conjugate_l1078_107890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_bottom_probability_l1078_107821

/-- Represents a vertex in a dodecahedron --/
inductive DodecahedronVertex
  | Top
  | Middle
  | Bottom

/-- Represents the dodecahedron structure --/
structure Dodecahedron where
  vertices : List DodecahedronVertex
  bottom_vertices : Nat
  h_bottom_vertices : bottom_vertices = 3

/-- Represents an ant's position on the dodecahedron --/
structure AntPosition where
  current : DodecahedronVertex

/-- Represents the probability of moving to each type of vertex --/
def transitionProbability : DodecahedronVertex → DodecahedronVertex → ℚ
  | DodecahedronVertex.Top, DodecahedronVertex.Middle => 1
  | DodecahedronVertex.Middle, DodecahedronVertex.Bottom => 1 / 3
  | DodecahedronVertex.Middle, DodecahedronVertex.Middle => 2 / 3
  | _, _ => 0

/-- The main theorem to prove --/
theorem ant_bottom_probability (d : Dodecahedron) :
  (AntPosition.mk DodecahedronVertex.Top).current = DodecahedronVertex.Top →
  (transitionProbability DodecahedronVertex.Top DodecahedronVertex.Middle = 1) →
  (transitionProbability DodecahedronVertex.Middle DodecahedronVertex.Bottom = 1 / 3) →
  (∃ (final : AntPosition), final.current = DodecahedronVertex.Bottom) →
  (1 / 3 : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_bottom_probability_l1078_107821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sqrt_sum_squares_l1078_107870

theorem min_value_sqrt_sum_squares (x y z : ℝ) (h : 2*x + y + 3*z = 32) :
  (Real.sqrt ((x - 1)^2 + (y + 2)^2 + z^2) : ℝ) ≥ 16 * Real.sqrt 14 / 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sqrt_sum_squares_l1078_107870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1078_107809

noncomputable def f (a : ℝ) (x : ℝ) := 2 * Real.sin x * Real.cos x + a * Real.cos x - 2 * x

theorem function_properties :
  (∃ (a : ℝ), (deriv (f a)) (π / 6) = -2) →
  (∃ (a : ℝ), a = 2 ∧
    ∀ x ∈ Set.Icc (-π / 6) (7 * π / 6), f a x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1078_107809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l1078_107856

-- Define the triangle ABC
structure Triangle where
  A : Real  -- Angle A
  B : Real  -- Angle B
  C : Real  -- Angle C
  a : Nat   -- Side opposite to A
  b : Nat   -- Side opposite to B
  c : Nat   -- Side opposite to C

-- Define the conditions
def isValidTriangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0

def hasConsecutiveEvenSides (t : Triangle) : Prop :=
  ∃ x : Nat, t.a = x - 2 ∧ t.b = x ∧ t.c = x + 2

def angleRelation (t : Triangle) : Prop :=
  t.C = 2 * t.A

-- State the theorem
theorem triangle_side_value (t : Triangle) 
  (h1 : isValidTriangle t)
  (h2 : hasConsecutiveEvenSides t)
  (h3 : angleRelation t) :
  t.a = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l1078_107856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_product_probability_l1078_107863

def S : Finset Int := {-5, 0, -8, 7, 4, -2, -3, 1}

theorem negative_product_probability :
  let pairs := (S.product S).filter (λ p => p.1 ≠ p.2)
  (pairs.filter (λ p => p.1 * p.2 < 0)).card / pairs.card = 3 / 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_product_probability_l1078_107863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_shift_l1078_107804

noncomputable def f : ℝ → ℝ := sorry

theorem domain_shift (a b : ℝ) (h : a < b) :
  (∀ x, x ∈ Set.Icc a b ↔ f (x - 1) ∈ Set.Icc 0 1) →
  (∀ x, x ∈ Set.Icc (-2) (-1) ↔ f (x + 2) ∈ Set.Icc 0 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_shift_l1078_107804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_in_circle_l1078_107859

/-- Helper function to calculate the area of a quadrilateral -/
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  abs ((A.1 - C.1) * (B.2 - D.2) - (B.1 - D.1) * (A.2 - C.2)) / 2

/-- Given a circle and a point, proves the maximum area of a quadrilateral formed by perpendicular lines through the point -/
theorem max_area_quadrilateral_in_circle (P : ℝ × ℝ) (h₁ : P = (2, 1)) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4}
  ∃ (A B C D : ℝ × ℝ),
    A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle ∧
    (∃ (t : ℝ), A = P + t • (B - P)) ∧
    (∃ (s : ℝ), D = P + s • (C - P)) ∧
    (B - P) • (C - P) = 0 ∧
    (∀ (A' B' C' D' : ℝ × ℝ),
      A' ∈ circle → B' ∈ circle → C' ∈ circle → D' ∈ circle →
      (∃ (t' : ℝ), A' = P + t' • (B' - P)) →
      (∃ (s' : ℝ), D' = P + s' • (C' - P)) →
      (B' - P) • (C' - P) = 0 →
      area_quadrilateral A B C D ≥ area_quadrilateral A' B' C' D') ∧
    area_quadrilateral A B C D = Real.sqrt 15 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_in_circle_l1078_107859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_friends_l1078_107866

/-- The set of friends who have Pokemon cards -/
def Friends : Finset String := {"Sam", "Dan", "Tom", "Keith"}

/-- Each friend has 14 Pokemon cards -/
def has_14_cards (f : String) : Prop := f ∈ Friends → (∃ n : ℕ, n = 14)

theorem count_friends : 
  (∀ f ∈ Friends, has_14_cards f) → Finset.card Friends = 4 := by
  intro h
  simp [Friends]
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_friends_l1078_107866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_M_l1078_107878

def M : ℕ := 2^5 * 3^4 * 5^3 * 7^3 * 11

theorem number_of_factors_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 960 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_M_l1078_107878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1078_107818

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 + x - 6 > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ≤ 2, y = 2^x - 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = Set.Ioc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1078_107818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangent_lines_l1078_107884

/-- A line that passes through a given point and intersects a parabola at only one point -/
structure TangentLine where
  slope : ℝ
  passes_through : ℝ × ℝ := (2, 4)
  intersects_once : ∀ x y : ℝ, y^2 = 8*x → (y - 4 = slope * (x - 2)) → (x, y) = (2, 4)

/-- The number of lines passing through (2,4) that intersect y² = 8x at only one point -/
def num_tangent_lines : ℕ := 2

/-- Theorem stating that there are exactly two lines passing through (2,4) 
    that intersect the parabola y² = 8x at only one point -/
theorem two_tangent_lines : 
  ∃! (lines : Finset TangentLine), lines.card = num_tangent_lines := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangent_lines_l1078_107884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_equal_diagonals_l1078_107808

/-- Definition of a convex polygon -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

/-- Definition of a diagonal in a polygon -/
noncomputable def ConvexPolygon.diagonal {n : ℕ} (P : ConvexPolygon n) (i j : Fin n) : ℝ :=
  let (x₁, y₁) := P.vertices i
  let (x₂, y₂) := P.vertices j
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- A convex polygon with all diagonals equal has either 4 or 5 sides -/
theorem convex_polygon_equal_diagonals (n : ℕ) (P : ConvexPolygon n) : 
  (∀ (i j k l : Fin n), i ≠ j ∧ k ≠ l ∧ i ≠ k ∧ j ≠ l → P.diagonal i j = P.diagonal k l) → 
  n = 4 ∨ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_equal_diagonals_l1078_107808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_five_fourths_l1078_107824

-- Define a circle type
structure Circle where
  radius : ℝ

-- Define circumference of a circle
noncomputable def circumference (c : Circle) : ℝ := 2 * Real.pi * c.radius

-- Define diameter of a circle
def diameter (c : Circle) : ℝ := 2 * c.radius

-- Define area of a circle
noncomputable def area (c : Circle) : ℝ := Real.pi * c.radius ^ 2

-- Theorem statement
theorem circle_area_is_five_fourths (c : Circle) 
  (h : 5 * (1 / circumference c) = diameter c) : 
  area c = 5 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_five_fourths_l1078_107824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1078_107862

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real
  area : Real

-- State the theorem
theorem triangle_side_length (abc : Triangle) 
  (h1 : abc.A = π / 3)
  (h2 : abc.AB = 2)
  (h3 : abc.area = Real.sqrt 3 / 2) :
  abc.AC = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1078_107862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1078_107820

def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 2

theorem problem_solution :
  (∀ x, x ∈ Set.Icc 0 4 → f 1 x ∈ Set.Icc 1 10) ∧
  (∀ a : ℝ, (∀ x, x ∈ Set.Icc a (a + 2) → f 1 x ≤ 5) → a ∈ Set.Icc (-1) 1) ∧
  (∀ t : ℝ, (∀ x₁ x₂, x₁ ∈ Set.Icc 0 4 → x₂ ∈ Set.Icc 0 4 → |f t x₁ - f t x₂| ≤ 8) →
    t ∈ Set.Icc (4 - 2*Real.sqrt 2) (2*Real.sqrt 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1078_107820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_N_exists_l1078_107814

/-- The probability Q(N) that at least 2/3 of N yellow balls are on the same side of one blue ball
    in a random arrangement of N yellow balls and one blue ball. -/
noncomputable def Q (N : ℕ) : ℚ :=
  (Int.floor (N / 3 : ℚ) + 1 + (N - Int.ceil (2 * N / 3 : ℚ) + 1)) / (N + 1 : ℚ)

/-- Theorem stating that there exists a smallest positive multiple of 6, N,
    such that Q(N) < 7/9. -/
theorem smallest_N_exists : ∃ N : ℕ,
  (∃ k : ℕ, N = 6 * k) ∧
  Q N < 7/9 ∧
  ∀ M : ℕ, (∃ j : ℕ, M = 6 * j) → M < N → Q M ≥ 7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_N_exists_l1078_107814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_result_l1078_107807

theorem power_equation_result (x : ℝ) (h : (16 : ℝ)^8 = (8 : ℝ)^x) : 
  (2 : ℝ)^(-x) = 1 / (2 : ℝ)^(32/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_result_l1078_107807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_most_appropriate_l1078_107838

structure Population where
  elderly : ℕ
  middleAged : ℕ
  young : ℕ

inductive SamplingMethod
  | SimpleRandom
  | DrawingLots
  | RandomNumberTable
  | Stratified

def totalPopulation (p : Population) : ℕ := p.elderly + p.middleAged + p.young

def sampleSize : ℕ := 41

def isAppropriateMethod (p : Population) (method : SamplingMethod) : Prop :=
  method = SamplingMethod.Stratified ∧
  totalPopulation p > sampleSize ∧
  p.elderly > 0 ∧ p.middleAged > 0 ∧ p.young > 0

theorem stratified_sampling_most_appropriate (p : Population) 
  (h1 : p.elderly = 28) (h2 : p.middleAged = 56) (h3 : p.young = 80) :
  isAppropriateMethod p SamplingMethod.Stratified := by
  sorry

#check stratified_sampling_most_appropriate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_most_appropriate_l1078_107838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1078_107872

noncomputable def eccentricity (m : ℝ) : ℝ := Real.sqrt (4 - m) / 2

theorem eccentricity_range :
  ∀ m : ℝ, m ∈ Set.Icc (-2) (-1) →
    eccentricity m ∈ Set.Icc (Real.sqrt 5 / 2) (Real.sqrt 6 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1078_107872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_collinear_of_collinear_and_not_collinear_l1078_107846

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are collinear if one is a scalar multiple of the other. -/
def collinear (u v : V) : Prop := ∃ k : ℝ, v = k • u

theorem not_collinear_of_collinear_and_not_collinear
  (a b c : V)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hab : ¬ collinear a b)
  (hac : collinear a c) :
  ¬ collinear b c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_collinear_of_collinear_and_not_collinear_l1078_107846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_lines_l1078_107849

noncomputable def f (x : ℝ) : ℝ := x + 1/x

noncomputable def line1 (x : ℝ) : ℝ := (1 + Real.sqrt 2) * x

noncomputable def line2 (x : ℝ) : ℝ := (1 - Real.sqrt 2) * x

theorem symmetry_lines :
  (∀ x : ℝ, x ≠ 0 → ∃ y : ℝ, f y = line1 x ∧ f (-y) = line1 (-x)) ∧
  (∀ x : ℝ, x ≠ 0 → ∃ y : ℝ, f y = line2 x ∧ f (-y) = line2 (-x)) := by
  sorry

#check symmetry_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_lines_l1078_107849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_length_approx_l1078_107845

/-- Perimeter of an isosceles trapezoid -/
noncomputable def trapezoid_perimeter (lower_base upper_base height : ℝ) : ℝ :=
  let horizontal_distance := (lower_base - upper_base) / 2
  let slant_height := Real.sqrt (horizontal_distance^2 + height^2)
  lower_base + upper_base + 2 * slant_height

/-- The perimeter of the specified isosceles trapezoid is approximately 88.28 feet -/
theorem fence_length_approx : 
  ∃ ε > 0, |trapezoid_perimeter 40 20 10 - 88.28| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_length_approx_l1078_107845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_COB_proof_l1078_107889

-- Define the points
def Q : ℚ × ℚ := (0, 15)
def O : ℚ × ℚ := (0, 0)
def B : ℚ × ℚ := (15, 0)
def C (p : ℚ) : ℚ × ℚ := (0, p)

-- Define the condition for p
def p_condition (p : ℚ) : Prop := 0 ≤ p ∧ p ≤ 15

-- Define the area of triangle COB
def area_COB (p : ℚ) : ℚ := (15 * p) / 2

-- Theorem statement
theorem area_COB_proof (p : ℚ) (h : p_condition p) : 
  area_COB p = (15 * p) / 2 := by
  -- Unfold the definition of area_COB
  unfold area_COB
  -- The proof is trivial as it's just the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_COB_proof_l1078_107889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_probability_twenty_l1078_107810

def fairCoin : Finset ℕ := {5, 15}
def standardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

def sumProbability (coin : Finset ℕ) (die : Finset ℕ) (target : ℕ) : ℚ :=
  (coin.filter (fun c => (die.filter (fun d => c + d = target)).card > 0)).card / 
  (2 * die.card)

theorem sum_probability_twenty :
  sumProbability fairCoin standardDie 20 = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_probability_twenty_l1078_107810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1078_107840

def multiset_identical (a b c d : ℝ) : Prop :=
  (a = c ∧ b = d) ∨ (a = d ∧ b = c)

theorem function_property (f : ℝ → ℝ) :
  (∀ x y : ℝ, multiset_identical (f (x * f y + 1)) (f (y * f x - 1)) (x * f (f y) + 1) (y * f (f x) - 1)) →
  (∀ x : ℝ, f x = -x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1078_107840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1078_107860

/-- Given an ellipse with the equation x²/a² + y²/b² = 1 where a > b > 0,
    center O, left focus F, and a point A on the ellipse satisfying certain conditions,
    prove that the eccentricity of the ellipse is (√10 - √2)/2 -/
theorem ellipse_eccentricity (a b c : ℝ) (O F A : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ ∃ t : ℝ, (x, y) = (a * Real.cos t, b * Real.sin t)) ∧
  F = (-c, 0) ∧
  (∃ t : ℝ, A = (a * Real.cos t, b * Real.sin t)) ∧
  (A.1 - O.1) * (A.1 - F.1) + (A.2 - O.2) * (A.2 - F.2) = 0 ∧
  (A.1 - O.1) * (F.1 - O.1) + (A.2 - O.2) * (F.2 - O.2) = (1/2) * ((F.1 - O.1)^2 + (F.2 - O.2)^2) →
  c/a = (Real.sqrt 10 - Real.sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1078_107860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1078_107888

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors and points
variable (a b : V)
variable (A B C D : V)

-- State the theorem
theorem vector_problem 
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : ¬ ∃ (r : ℝ), a = r • b)
  (h4 : B - A = a + b)
  (h5 : C - B = 2 • a + 8 • b)
  (h6 : D - C = 3 • (a - b)) :
  (∃ (r : ℝ), D - A = r • (B - A)) ∧
  (∃ (k : ℝ), k = -1 ∧ ∃ (t : ℝ), t < 0 ∧ k • a + b = t • (a + k • b)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1078_107888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_constant_product_l1078_107843

noncomputable section

/-- Definition of the ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of eccentricity -/
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Definition of the circle -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 3/4

/-- Definition of a point being on the x-axis -/
def on_x_axis (x : ℝ) : Prop :=
  true

/-- Definition of the dot product of two vectors -/
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ :=
  x1 * x2 + y1 * y2

theorem ellipse_and_constant_product :
  ∃ (a b : ℝ),
    (∀ x y, ellipse_C x y a b → (x^2 / 4 + y^2 / 3 = 1)) ∧
    (eccentricity a b = 1/2) ∧
    (∃ (xf yf : ℝ), ellipse_C xf yf a b ∧ circle_eq xf yf) ∧
    (∃ (xn : ℝ),
      on_x_axis xn ∧
      xn = 11/8 ∧
      ∀ (x1 y1 x2 y2 : ℝ),
        ellipse_C x1 y1 a b →
        ellipse_C x2 y2 a b →
        dot_product (x1 - xn) y1 (x2 - xn) y2 = -135/64) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_constant_product_l1078_107843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_groups_for_thirty_players_l1078_107811

/-- Given a total number of players and a maximum number of players per group,
    calculate the minimum number of equal-sized groups required. -/
def minGroupsRequired (totalPlayers : ℕ) (maxPlayersPerGroup : ℕ) : ℕ :=
  let validDivisors := (List.range (maxPlayersPerGroup + 1)).filter (fun d => d > 0 && totalPlayers % d == 0)
  match validDivisors.maximum? with
  | some size => totalPlayers / size
  | none => totalPlayers  -- If no valid group size found, each player is in their own group

/-- Theorem stating that for 30 players and a maximum of 12 players per group,
    the minimum number of equal-sized groups is 3. -/
theorem min_groups_for_thirty_players :
  minGroupsRequired 30 12 = 3 := by
  -- Proof goes here
  sorry

#eval minGroupsRequired 30 12  -- Should output 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_groups_for_thirty_players_l1078_107811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_real_f_is_odd_f_decreasing_on_positive_l1078_107871

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x)

-- State the theorems to be proven
theorem f_range_is_real : Set.range f = Set.univ := by sorry

theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

theorem f_decreasing_on_positive : ∀ x y, 0 < x → x < y → f y < f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_is_real_f_is_odd_f_decreasing_on_positive_l1078_107871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_pizza_slices_l1078_107834

theorem james_pizza_slices (initial_slices : ℕ) (friend_eats : ℕ) (james_fraction : ℚ) : ℕ :=
  by
  -- Define the conditions
  have h1 : initial_slices = 8 := by sorry
  have h2 : friend_eats = 2 := by sorry
  have h3 : james_fraction = 1/2 := by sorry

  -- Calculate the number of slices James eats
  let remaining_slices := initial_slices - friend_eats
  let james_eats := (remaining_slices : ℚ) * james_fraction

  -- Prove that James eats 3 slices
  have h4 : james_eats = 3 := by sorry

  -- Convert the rational number to a natural number
  exact Int.toNat (Int.floor james_eats)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_pizza_slices_l1078_107834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vector_scalar_l1078_107829

/-- Given vectors a and b, if (a - λb) is orthogonal to b, then λ = 3/5 -/
theorem orthogonal_vector_scalar (a b : ℝ × ℝ) (lambda : ℝ) :
  a = (1, 3) →
  b = (3, 4) →
  (a.1 - lambda * b.1) * b.1 + (a.2 - lambda * b.2) * b.2 = 0 →
  lambda = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vector_scalar_l1078_107829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_mpg_is_fourteen_l1078_107876

/-- Represents the fuel efficiency of a car -/
structure CarFuelEfficiency where
  highway_miles_per_tankful : ℚ
  city_miles_per_tankful : ℚ
  city_mpg_difference : ℚ

/-- Calculates the city miles per gallon given the car's fuel efficiency data -/
noncomputable def calculate_city_mpg (car : CarFuelEfficiency) : ℚ :=
  let highway_mpg := car.highway_miles_per_tankful / (car.city_miles_per_tankful / (car.highway_miles_per_tankful / car.city_miles_per_tankful - car.city_mpg_difference))
  highway_mpg - car.city_mpg_difference

/-- Theorem stating that for the given car specifications, the city mpg is 14 -/
theorem city_mpg_is_fourteen :
  let car := CarFuelEfficiency.mk 480 336 6
  calculate_city_mpg car = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_mpg_is_fourteen_l1078_107876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_rate_is_eight_l1078_107831

/-- Represents the rowing speeds of a man in different conditions -/
structure RowingSpeed where
  withStream : ℚ  -- Speed with the stream
  againstStream : ℚ  -- Speed against the stream

/-- Calculates the man's rate (rowing speed in still water) -/
def mansRate (speed : RowingSpeed) : ℚ :=
  (speed.withStream + speed.againstStream) / 2

/-- Theorem stating that given the specific rowing speeds, the man's rate is 8 km/h -/
theorem mans_rate_is_eight (speed : RowingSpeed) 
  (h1 : speed.withStream = 12)
  (h2 : speed.againstStream = 4) : 
  mansRate speed = 8 := by
  unfold mansRate
  rw [h1, h2]
  norm_num

#eval mansRate { withStream := 12, againstStream := 4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_rate_is_eight_l1078_107831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AP_is_half_l1078_107813

/-- A rectangle with an inscribed circle -/
structure RectangleWithCircle where
  /-- The width of the rectangle -/
  width : ℝ
  /-- The height of the rectangle -/
  height : ℝ
  /-- The center of the inscribed circle -/
  center : ℝ × ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ

/-- The specific rectangle described in the problem -/
noncomputable def problemRectangle : RectangleWithCircle where
  width := 2
  height := 1
  center := (0, 0)
  radius := 1/2

/-- Point A in the problem -/
noncomputable def A : ℝ × ℝ := (0, 1)

/-- Point M in the problem -/
noncomputable def M : ℝ × ℝ := (0, -1/2)

/-- Function to calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The length of AP is 1/2 -/
theorem length_AP_is_half (rect : RectangleWithCircle) 
    (h1 : rect = problemRectangle) 
    (h2 : A = (0, 1)) 
    (h3 : M = (0, -1/2)) : 
  ∃ P : ℝ × ℝ, P.1 = 0 ∧ P.2 = 1/2 ∧ distance A P = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AP_is_half_l1078_107813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_calculation_l1078_107837

/-- Calculates the total cost of purchasing an iPhone 12, iWatch, and iPad with given discounts, taxes, special promotion, and cashback. -/
theorem total_cost_calculation (iphone_price iwatch_price ipad_price : ℝ)
  (iphone_discount iwatch_discount ipad_discount : ℝ)
  (iphone_tax iwatch_tax ipad_tax : ℝ)
  (special_promotion cashback : ℝ)
  (h_iphone : iphone_price = 820)
  (h_iwatch : iwatch_price = 320)
  (h_ipad : ipad_price = 520)
  (h_iphone_discount : iphone_discount = 0.15)
  (h_iwatch_discount : iwatch_discount = 0.10)
  (h_ipad_discount : ipad_discount = 0.05)
  (h_iphone_tax : iphone_tax = 0.07)
  (h_iwatch_tax : iwatch_tax = 0.05)
  (h_ipad_tax : ipad_tax = 0.06)
  (h_special_promotion : special_promotion = 0.03)
  (h_cashback : cashback = 0.02) :
  (let discounted_iphone := iphone_price * (1 - iphone_discount)
   let discounted_iwatch := iwatch_price * (1 - iwatch_discount)
   let discounted_ipad := ipad_price * (1 - ipad_discount)
   let subtotal := discounted_iphone + discounted_iwatch + discounted_ipad
   let after_promotion := subtotal * (1 - special_promotion)
   let total_tax := discounted_iphone * iphone_tax + discounted_iwatch * iwatch_tax + discounted_ipad * ipad_tax
   let total_with_tax := after_promotion + total_tax
   let final_total := total_with_tax * (1 - cashback)
   final_total) = 1496.91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_calculation_l1078_107837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1078_107855

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := (3 - i) / (1 + i)

theorem modulus_of_z : Complex.abs z = Real.sqrt 5 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l1078_107855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_a_zero_l1078_107850

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) ^ (a * x^2 - 4 + 3)

-- State the theorem
theorem domain_implies_a_zero :
  (∀ x > 0, f a x > 0) → a = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_a_zero_l1078_107850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_6_deg_approx_l1078_107851

/-- The number of equal parts the circle is divided into -/
def n : ℕ := 60

/-- The angle in radians -/
noncomputable def θ : ℝ := 6 * Real.pi / 180

/-- Approximation of sin θ using the art of circle-cutting method -/
noncomputable def circle_cutting_approx (n : ℕ) (θ : ℝ) : ℝ := Real.pi / (2 * n)

/-- Theorem stating that the approximation of sin 6° using the art of circle-cutting
    method with 60 divisions is equal to π/30 -/
theorem sin_6_deg_approx :
  circle_cutting_approx n θ = Real.pi / 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_6_deg_approx_l1078_107851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1078_107881

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x + Real.pi / 3)

noncomputable def g (x : ℝ) := f (x - Real.pi / 4)

def symmetric_about (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x, f (a - x) = f (a + x)

theorem problem_statement :
  (¬ (∀ x y, x ∈ Set.Icc (-Real.pi/3) 0 → y ∈ Set.Icc (-Real.pi/3) 0 → x < y → g x < g y)) ∧
  (∀ f : ℝ → ℝ, (∀ x, f (-x) = f (3 + x)) → symmetric_about f (3/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1078_107881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_not_in_photo_simultaneously_l1078_107874

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℚ
  direction : Bool -- True for counterclockwise, False for clockwise

/-- Calculates the position of a runner at a given time -/
def position (r : Runner) (t : ℚ) : ℚ :=
  let laps := t / r.lapTime
  if r.direction then
    laps % 1
  else
    (1 - laps % 1) % 1

/-- Checks if a runner is within the photo range -/
def inPhoto (r : Runner) (t : ℚ) : Prop :=
  let pos := position r t
  pos ≤ 1/6 ∨ pos ≥ 5/6

theorem runners_not_in_photo_simultaneously
  (miranda : Runner)
  (milan : Runner)
  (h_miranda : miranda.lapTime = 100 ∧ miranda.direction = true)
  (h_milan : milan.lapTime = 60 ∧ milan.direction = false) :
  ∀ t, 480 ≤ t ∧ t ≤ 540 → ¬(inPhoto miranda t ∧ inPhoto milan t) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_not_in_photo_simultaneously_l1078_107874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_first_five_hours_l1078_107848

/-- Represents the average speed for the first 5 hours of the trip. -/
def S : ℝ := sorry

/-- The total trip duration in hours. -/
def total_time : ℝ := 15

/-- The average speed for the entire trip in miles per hour. -/
def avg_speed : ℝ := 38

/-- The speed for each additional hour after the first 5 hours, in miles per hour. -/
def additional_speed : ℝ := 42

/-- The duration of the initial part of the trip in hours. -/
def initial_time : ℝ := 5

theorem average_speed_first_five_hours :
  S * initial_time + additional_speed * (total_time - initial_time) = avg_speed * total_time → S = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_first_five_hours_l1078_107848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l1078_107827

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

-- State the theorem
theorem odd_function_and_inequality (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →  -- f is an odd function
  (a = 2 ∧ b = 1) ∧  -- Part 1: values of a and b
  (∀ k, (∀ t, f a b (t^2 - 2*t) + f a b (2*t^2 - k) < 0) → k < -1/3) :=  -- Part 2: range of k
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l1078_107827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_in_eighth_row_l1078_107832

/-- Represents a lattice with a given number of rows and columns -/
structure MyLattice where
  rows : Nat
  columns : Nat

/-- Returns the last number in a given row of the lattice -/
def lastNumberInRow (l : MyLattice) (row : Nat) : Nat :=
  row * l.columns

/-- Returns the third number in a given row of the lattice -/
def thirdNumberInRow (l : MyLattice) (row : Nat) : Nat :=
  lastNumberInRow l row - 3

/-- The main theorem to prove -/
theorem third_number_in_eighth_row :
  let l : MyLattice := { rows := 8, columns := 6 }
  thirdNumberInRow l 8 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_in_eighth_row_l1078_107832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_properties_l1078_107805

/-- The volume function of a tetrahedron with one edge of length x and all others of length 1 -/
noncomputable def F : ℝ → ℝ := sorry

/-- The domain of x for which the tetrahedron exists -/
def TetrahedronDomain : Set ℝ := {x : ℝ | x > 0}

theorem tetrahedron_volume_properties :
  ∃ (x_max : ℝ) (v_max : ℝ),
    x_max ∈ TetrahedronDomain ∧
    (∀ x ∈ TetrahedronDomain, F x ≤ v_max) ∧
    (F x_max = v_max) ∧
    (∃ x₁ x₂, x₁ ∈ TetrahedronDomain ∧ x₂ ∈ TetrahedronDomain ∧ x₁ < x₂ ∧ F x₁ > F x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_properties_l1078_107805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_not_twenty_l1078_107885

def jump_results : List ℕ := [150, 160, 165, 145, 150, 170]

theorem range_not_twenty : 
  let range := jump_results.maximum.getD 0 - jump_results.minimum.getD 0
  range ≠ 20 := by
  simp [jump_results]
  norm_num
  decide

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_not_twenty_l1078_107885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l1078_107833

def letters : List Char := ['P', 'R', 'O', 'B', 'L', 'E', 'M', 'S']

def isValidSequence (seq : List Char) : Bool :=
  seq.length = 5 &&
  seq.head? = some 'S' &&
  seq.getLast? ≠ some 'M' &&
  seq.toFinset ⊆ letters.toFinset &&
  seq.toFinset.card = seq.length

def countValidSequences : Nat :=
  (letters.permutations.filter isValidSequence).length

theorem valid_sequences_count :
  countValidSequences = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l1078_107833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concur_l1078_107853

-- Define the points as complex numbers
variable (X Y Z A B C D : ℂ)

-- Define the properties of the points and triangles
axiom points_on_line : X.re < Y.re ∧ Y.re < Z.re ∧ X.im = Y.im ∧ Y.im = Z.im

axiom triangle_XAB : Complex.abs (A - X) = Complex.abs (B - X) ∧ Complex.abs (B - A) = Complex.abs (A - X)
axiom triangle_YBC : Complex.abs (B - Y) = Complex.abs (C - Y) ∧ Complex.abs (C - B) = Complex.abs (B - Y)
axiom triangle_ZCD : Complex.abs (C - Z) = Complex.abs (D - Z) ∧ Complex.abs (D - C) = Complex.abs (C - Z)

axiom orientation_XAB : Complex.arg (A - X) - Complex.arg (B - X) = 2 * Real.pi / 3
axiom orientation_YBC : Complex.arg (C - Y) - Complex.arg (B - Y) = -2 * Real.pi / 3
axiom orientation_ZCD : Complex.arg (C - Z) - Complex.arg (D - Z) = 2 * Real.pi / 3

-- Define a helper function for a line through two points
def line_through (P Q : ℂ) : Set ℂ :=
  {z : ℂ | ∃ t : ℝ, z = (1 - t) • P + t • Q}

-- Define the theorem to be proved
theorem lines_concur :
  ∃ P : ℂ, P ∈ line_through A C ∧ P ∈ line_through B D ∧ P ∈ line_through X Y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concur_l1078_107853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_theorem_l1078_107812

/-- Represents a two-segment journey --/
structure Journey where
  distance1 : ℝ  -- Distance of first segment
  distance2 : ℝ  -- Distance of second segment
  speed1 : ℝ     -- Speed of first segment
  speed2 : ℝ     -- Speed of second segment

/-- Calculate the average speed of a journey --/
noncomputable def averageSpeed (j : Journey) : ℝ :=
  (j.distance1 + j.distance2) / (j.distance1 / j.speed1 + j.distance2 / j.speed2)

theorem journey_speed_theorem (j : Journey) :
  j.distance1 = 2 * j.distance2 →
  j.speed2 = 20 →
  averageSpeed j = 36 →
  j.speed1 = 60 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_theorem_l1078_107812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l1078_107858

def sequence_a : ℕ → ℚ
  | 0 => 5  -- Add a case for 0 to cover all natural numbers
  | 1 => 5
  | (n + 2) => (3 * sequence_a (n + 1) - 1) / (-sequence_a (n + 1) + 3)

theorem sequence_a_general_term (n : ℕ) (h : n ≥ 1) :
  sequence_a n = (2^n + 1) / (1 - 2^(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l1078_107858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l1078_107891

-- Define the two equations
def equation1 (x y : ℝ) : Prop := (x - y + 2) * (3 * x + y - 4) = 0
def equation2 (x y : ℝ) : Prop := (x + y - 2) * (2 * x - 5 * y + 7) = 0

-- Define an intersection point
def is_intersection_point (x y : ℝ) : Prop :=
  equation1 x y ∧ equation2 x y

-- Define distinct intersection points
def distinct_intersection_points (points : List (ℝ × ℝ)) : Prop :=
  (∀ p, p ∈ points → is_intersection_point p.1 p.2) ∧
  (∀ p q, p ∈ points → q ∈ points → p ≠ q → p.1 ≠ q.1 ∨ p.2 ≠ q.2)

-- Theorem statement
theorem intersection_count :
  ∃ (points : List (ℝ × ℝ)), distinct_intersection_points points ∧ points.length = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l1078_107891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expressions_l1078_107800

open Real

noncomputable def expression1 : ℝ := 1 / sin (10 * π / 180) - sqrt 3 / cos (10 * π / 180)

noncomputable def expression2 : ℝ := 
  (sin (50 * π / 180) * (1 + sqrt 3 * tan (10 * π / 180)) - cos (20 * π / 180)) /
  (cos (80 * π / 180) * sqrt (1 - cos (20 * π / 180)))

-- State the theorem
theorem trigonometric_expressions :
  expression1 = 4 ∧ expression2 = sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expressions_l1078_107800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_l1078_107873

theorem plane_equation_proof (x y z : ℝ) :
  let given_plane : ℝ → ℝ → ℝ → Prop := λ x y z ↦ 2*x - y - 3*z + 5 = 0
  let point_M : ℝ × ℝ × ℝ := (-2, 0, 3)
  let parallel_plane : ℝ → ℝ → ℝ → Prop := λ x y z ↦ 2*x - y - 3*z + 13 = 0
  (∀ (x y z : ℝ), parallel_plane x y z ↔ given_plane x y z) ∧ 
  parallel_plane point_M.1 point_M.2.1 point_M.2.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_l1078_107873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_permutation_l1078_107847

/-- A type representing the vertices of an octagon -/
inductive Vertex : Type
  | one | two | three | four | five | six | seven | eight

/-- A function representing a permutation of vertex numbers -/
def Permutation := Vertex → Vertex

/-- A function representing a rotation of the octagon -/
def Rotation := Vertex → Vertex

/-- Predicate to check if a permutation matches a vertex for a given rotation -/
def matchesVertex (p : Permutation) (r : Rotation) : Prop :=
  ∃ v : Vertex, p (r v) = v

/-- Theorem stating that no permutation satisfies the matching condition for all rotations -/
theorem no_valid_permutation :
  ¬ ∃ p : Permutation, ∀ r : Rotation, matchesVertex p r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_permutation_l1078_107847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1078_107861

/-- The area of a triangle with vertices at (-1, 0), (7, 4), and (7, -4) is 32 square units. -/
theorem triangle_area : ∃ (area : ℝ), area = 32 := by
  -- Define the vertices of the triangle
  let v1 : ℝ × ℝ := (-1, 0)
  let v2 : ℝ × ℝ := (7, 4)
  let v3 : ℝ × ℝ := (7, -4)

  -- Define the function to calculate the area of a triangle given its vertices
  let triangle_area_calc (a b c : ℝ × ℝ) : ℝ :=
    let (x1, y1) := a
    let (x2, y2) := b
    let (x3, y3) := c
    (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

  -- Calculate the area
  let area := triangle_area_calc v1 v2 v3

  -- Assert that the area is 32
  have h : area = 32 := by
    -- The proof would go here
    sorry

  -- Conclude the theorem
  exact ⟨area, h⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1078_107861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_15_l1078_107842

def S : Set ℤ := {-30, -5, -3, 1, 3, 10, 15}

theorem largest_quotient_is_15 :
  ∀ a b, a ∈ S → b ∈ S → a ≠ 0 → b ≠ 0 → (a : ℚ) / b ≤ 15 ∧ ∃ c d, c ∈ S ∧ d ∈ S ∧ c ≠ 0 ∧ d ≠ 0 ∧ (c : ℚ) / d = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_15_l1078_107842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_current_salary_l1078_107852

noncomputable def freds_salary_5_months_ago : ℝ := 1000

noncomputable def mikes_salary_5_months_ago (freds_salary : ℝ) : ℝ := 10 * freds_salary

def salary_increase_percentage : ℝ := 40

noncomputable def calculate_current_salary (initial_salary : ℝ) (increase_percentage : ℝ) : ℝ :=
  initial_salary * (1 + increase_percentage / 100)

theorem mikes_current_salary :
  calculate_current_salary (mikes_salary_5_months_ago freds_salary_5_months_ago) salary_increase_percentage = 14000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_current_salary_l1078_107852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_increase_is_six_percent_l1078_107887

/-- Calculates the percentage increase in property tax from 1995 to 1996 -/
noncomputable def tax_increase_percentage (tax_1995 tax_1996 surcharge : ℝ) : ℝ :=
  ((tax_1996 - surcharge - tax_1995) / tax_1995) * 100

/-- Theorem stating that the percentage increase in property tax from 1995 to 1996 is 6% -/
theorem tax_increase_is_six_percent (tax_1995 tax_1996 surcharge : ℝ)
  (h1 : tax_1995 = 1800)
  (h2 : tax_1996 = 2108)
  (h3 : surcharge = 200) :
  tax_increase_percentage tax_1995 tax_1996 surcharge = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_increase_is_six_percent_l1078_107887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_average_speed_approx_l1078_107844

/-- Represents a segment of Susan's trip -/
structure Segment where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a segment -/
noncomputable def time_for_segment (s : Segment) : ℝ := s.distance / s.speed

/-- Susan's trip segments -/
def susan_trip : List Segment := [
  ⟨40, 15⟩,
  ⟨20, 60⟩,
  ⟨30, 50⟩,
  ⟨10, 70⟩
]

/-- Total distance of Susan's trip -/
noncomputable def total_distance : ℝ := (susan_trip.map (λ s => s.distance)).sum

/-- Total time of Susan's trip -/
noncomputable def total_time : ℝ := (susan_trip.map time_for_segment).sum

/-- Susan's average speed for the entire trip -/
noncomputable def average_speed : ℝ := total_distance / total_time

theorem susan_average_speed_approx :
  |average_speed - 26.74| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_average_speed_approx_l1078_107844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contributions_before_johns_l1078_107830

/-- The number of contributions made before John's --/
def n : ℕ := 6

/-- The average contribution before John's donation --/
def A : ℚ := 50

/-- John's contribution amount --/
def john_contribution : ℚ := 225

/-- The new average contribution after John's donation --/
def new_average : ℚ := 75

theorem contributions_before_johns :
  (↑n * A + john_contribution) / (↑n + 1) = new_average ∧
  new_average = 1.5 * A ∧
  john_contribution = 225 ∧
  new_average = 75 →
  n = 6 := by
  sorry

#check contributions_before_johns

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contributions_before_johns_l1078_107830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_for_a_equals_one_range_for_necessary_not_sufficient_l1078_107826

-- Define the statements p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := Real.log (x - 2) < 0

-- Theorem for the first part of the problem
theorem range_for_a_equals_one :
  ∀ x : ℝ, (p 1 x ∧ q x) ↔ (2 < x ∧ x < 3) := by sorry

-- Theorem for the second part of the problem
theorem range_for_necessary_not_sufficient :
  ∀ a : ℝ, (a > 0 ∧ (∀ x : ℝ, q x → p a x) ∧ (∃ x : ℝ, p a x ∧ ¬q x)) ↔ (1 ≤ a ∧ a ≤ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_for_a_equals_one_range_for_necessary_not_sufficient_l1078_107826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_withdrawal_l1078_107877

def initialBalance : ℕ := 500
def withdrawAmount : ℕ := 300
def depositAmount : ℕ := 198

def isValidTransaction (balance : ℤ) (newBalance : ℤ) : Prop :=
  newBalance = balance - withdrawAmount ∨ newBalance = balance + depositAmount

def isAchievableBalance (amount : ℤ) : Prop :=
  ∃ (sequence : List ℤ), 
    sequence.head? = some initialBalance ∧
    sequence.getLast? = some amount ∧
    ∀ (i : ℕ), i < sequence.length - 1 → 
      isValidTransaction (sequence[i]!) (sequence[i + 1]!)

theorem max_withdrawal :
  (∀ (amount : ℤ), amount > 498 → ¬isAchievableBalance amount) ∧
  isAchievableBalance 498 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_withdrawal_l1078_107877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_disjoint_subsets_l1078_107825

/-- The set of numbers from 1 to 10 -/
def S : Finset ℕ := Finset.range 10

/-- The number of pairs of disjoint subsets of S -/
def num_disjoint_pairs : ℕ := 3^10

/-- Theorem stating that the number of pairs (A, B) of subsets of S 
    such that A ∩ B = ∅ is equal to 3^10 -/
theorem count_disjoint_subsets : 
  (Finset.filter (fun p : Finset ℕ × Finset ℕ => p.1 ⊆ S ∧ p.2 ⊆ S ∧ p.1 ∩ p.2 = ∅) (Finset.product (Finset.powerset S) (Finset.powerset S))).card = num_disjoint_pairs :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_disjoint_subsets_l1078_107825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1078_107896

noncomputable def f (x : ℝ) : ℝ := 1 / (2 * x - 12)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1078_107896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_array_coin_sum_l1078_107892

theorem triangular_array_coin_sum (n : ℕ) : 
  (n * (n + 1)) / 2 = 2485 → (n.repr.toList.map (λ c => c.toString.toNat!)).sum = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_array_coin_sum_l1078_107892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_theorem_l1078_107868

/-- A line passing through a point with a given angle of inclination -/
structure InclinedLine where
  point : ℝ × ℝ
  angle : ℝ

/-- A circle in polar form -/
structure PolarCircle where
  equation : ℝ → ℝ

/-- Intersection points of a line and a circle -/
def line_circle_intersection (l : InclinedLine) (c : PolarCircle) : Set (ℝ × ℝ) :=
  sorry

/-- The distance product theorem for a line and circle intersection -/
theorem distance_product_theorem
  (l : InclinedLine)
  (c : PolarCircle)
  (h1 : l.point = (-1, 2))
  (h2 : l.angle = 2 * Real.pi / 3)
  (h3 : c.equation = fun θ => 2 * Real.cos (θ + Real.pi / 3)) :
  ∃ (M N : ℝ × ℝ),
    (M ∈ line_circle_intersection l c) ∧
    (N ∈ line_circle_intersection l c) ∧
    M ≠ N ∧
    dist l.point M * dist l.point N = 6 + 2 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_theorem_l1078_107868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_range_l1078_107854

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + Real.cos x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x * Real.cos x

-- Theorem for the maximum value of f
theorem f_max_value : ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

-- Theorem for the range of g
theorem g_range : ∀ (y : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ g x = y) ↔ y ∈ Set.Icc 1 (3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_range_l1078_107854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_properties_l1078_107857

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

-- State the theorem
theorem sine_properties (ω : ℝ) (h_increasing : ∀ x y, 0 < x ∧ x < y ∧ y < π/6 → f ω x < f ω y) :
  (∀ x y, -π/6 < x ∧ x < y ∧ y < 0 → f ω x < f ω y) ∧ 
  (∀ n : ℕ, n > 0 ∧ (∀ x y, 0 < x ∧ x < y ∧ y < π/6 → f n x < f n y) → n ≤ 3) ∧
  f ω (π/4) ≥ f ω (π/12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_properties_l1078_107857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flu_spread_equation_l1078_107899

/-- Represents the average number of people infected by one person in each round -/
def x : Real := sorry

/-- The total number of people with the flu after two rounds of infection -/
def total_infected (x : Real) : Real := 1 + x + x * (1 + x)

/-- Theorem stating that the equation correctly represents the total number of infected people -/
theorem flu_spread_equation : total_infected x = 81 ↔ x = -2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flu_spread_equation_l1078_107899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_difference_for_1600_payment_discount_difference_for_1600_payment_exact_l1078_107835

/-- Calculates the discounted price based on the total purchase amount -/
noncomputable def discounted_price (x : ℝ) : ℝ :=
  if x ≤ 500 then x
  else if x ≤ 1000 then 500 + 0.8 * (x - 500)
  else 500 + 400 + 0.5 * (x - 1000)

/-- Theorem stating the difference between total purchase and discounted price for a specific case -/
theorem discount_difference_for_1600_payment :
  ∃ x : ℝ, discounted_price x = 1600 ∧ x - discounted_price x = 800 :=
by
  sorry

/-- Alternative formulation using exact values -/
theorem discount_difference_for_1600_payment_exact :
  discounted_price 2400 = 1600 ∧ 2400 - discounted_price 2400 = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_difference_for_1600_payment_discount_difference_for_1600_payment_exact_l1078_107835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1078_107817

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (2 * x - 1)) / (x^2 + x - 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ 1/2 ∧ x ≠ 1}

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1078_107817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_bridge_48_seconds_l1078_107841

/-- Calculates the time (in seconds) for a train to pass a bridge -/
noncomputable def train_pass_bridge_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

theorem train_pass_bridge_48_seconds :
  train_pass_bridge_time 460 45 140 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_bridge_48_seconds_l1078_107841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_property_l1078_107886

theorem lcm_property (x y : ℕ) (hx : x > 1) (hy : y > 1) :
  Nat.lcm (x + 2) (y + 2) - Nat.lcm (x + 1) (y + 1) = Nat.lcm (x + 1) (y + 1) - Nat.lcm x y →
  x ∣ y ∨ y ∣ x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_property_l1078_107886

import Mathlib

namespace NUMINAMATH_CALUDE_probability_four_ones_in_six_rolls_l1348_134809

theorem probability_four_ones_in_six_rolls (n : ℕ) (p : ℚ) : 
  n = 10 → p = 1 / n → 
  (Nat.choose 6 4 : ℚ) * p^4 * (1 - p)^2 = 243 / 200000 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_ones_in_six_rolls_l1348_134809


namespace NUMINAMATH_CALUDE_james_sodas_per_day_l1348_134830

/-- Calculates the number of sodas James drinks per day given the following conditions:
  * James buys 5 packs of sodas
  * Each pack contains 12 sodas
  * James already had 10 sodas
  * He finishes all the sodas in 1 week
-/
def sodas_per_day (packs : ℕ) (sodas_per_pack : ℕ) (initial_sodas : ℕ) (days_in_week : ℕ) : ℕ :=
  ((packs * sodas_per_pack + initial_sodas) / days_in_week)

theorem james_sodas_per_day :
  sodas_per_day 5 12 10 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_sodas_per_day_l1348_134830


namespace NUMINAMATH_CALUDE_even_fraction_integers_l1348_134827

theorem even_fraction_integers (a : ℤ) : 
  (∃ k : ℤ, a / (1011 - a) = 2 * k) ↔ 
  a ∈ ({1010, 1012, 1008, 1014, 674, 1348, 0, 2022} : Set ℤ) := by
sorry

end NUMINAMATH_CALUDE_even_fraction_integers_l1348_134827


namespace NUMINAMATH_CALUDE_curve_scaling_transformation_l1348_134856

/-- Given a curve C with equation x² + y² = 1 and a scaling transformation,
    prove that the resulting curve C'' has the equation x² + y²/4 = 1 -/
theorem curve_scaling_transformation (x y x'' y'' : ℝ) :
  (x^2 + y^2 = 1) →    -- Equation of curve C
  (x'' = x) →          -- x-coordinate transformation
  (y'' = 2*y) →        -- y-coordinate transformation
  (x''^2 + y''^2/4 = 1) -- Equation of curve C''
:= by sorry

end NUMINAMATH_CALUDE_curve_scaling_transformation_l1348_134856


namespace NUMINAMATH_CALUDE_factorization_equality_l1348_134828

theorem factorization_equality (m : ℝ) : 2 * m^2 - 12 * m + 18 = 2 * (m - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1348_134828


namespace NUMINAMATH_CALUDE_installation_charge_company_x_l1348_134849

/-- Represents a company's pricing for an air conditioner --/
structure CompanyPricing where
  price : ℝ
  surcharge_rate : ℝ
  installation_charge : ℝ

/-- Calculates the total cost for a company --/
def total_cost (c : CompanyPricing) : ℝ :=
  c.price + c.price * c.surcharge_rate + c.installation_charge

theorem installation_charge_company_x (
  company_x : CompanyPricing)
  (company_y : CompanyPricing)
  (h1 : company_x.price = 575)
  (h2 : company_x.surcharge_rate = 0.04)
  (h3 : company_y.price = 530)
  (h4 : company_y.surcharge_rate = 0.03)
  (h5 : company_y.installation_charge = 93)
  (h6 : total_cost company_x - total_cost company_y = 41.60) :
  company_x.installation_charge = 82.50 := by
  sorry

end NUMINAMATH_CALUDE_installation_charge_company_x_l1348_134849


namespace NUMINAMATH_CALUDE_plan_b_rate_l1348_134832

/-- Represents the cost of a call under Plan A -/
def costPlanA (minutes : ℕ) : ℚ :=
  if minutes ≤ 6 then 60/100
  else 60/100 + (minutes - 6) * (6/100)

/-- Represents the cost of a call under Plan B -/
def costPlanB (rate : ℚ) (minutes : ℕ) : ℚ :=
  rate * minutes

/-- The duration at which both plans charge the same amount -/
def equalDuration : ℕ := 12

theorem plan_b_rate : ∃ (rate : ℚ), 
  costPlanA equalDuration = costPlanB rate equalDuration ∧ rate = 8/100 := by
  sorry

end NUMINAMATH_CALUDE_plan_b_rate_l1348_134832


namespace NUMINAMATH_CALUDE_segment_ratio_l1348_134888

/-- Given two line segments with equally spaced points, prove the ratio of their lengths -/
theorem segment_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ x : ℝ, x > 0 ∧ a = 9*x ∧ b = 99*x) → b / a = 11 := by
  sorry

end NUMINAMATH_CALUDE_segment_ratio_l1348_134888


namespace NUMINAMATH_CALUDE_unique_assignment_l1348_134889

-- Define the friends and professions as enums
inductive Friend : Type
  | Ivanov | Petrenko | Sidorchuk | Grishin | Altman

inductive Profession : Type
  | Painter | Miller | Carpenter | Postman | Barber

-- Define the assignment of professions to friends
def assignment : Friend → Profession
  | Friend.Ivanov => Profession.Barber
  | Friend.Petrenko => Profession.Miller
  | Friend.Sidorchuk => Profession.Postman
  | Friend.Grishin => Profession.Carpenter
  | Friend.Altman => Profession.Painter

-- Define the conditions
def conditions (a : Friend → Profession) : Prop :=
  -- Each friend has a unique profession
  (∀ f1 f2, f1 ≠ f2 → a f1 ≠ a f2) ∧
  -- Petrenko and Grishin have never used a painter's brush
  (a Friend.Petrenko ≠ Profession.Painter ∧ a Friend.Grishin ≠ Profession.Painter) ∧
  -- Ivanov and Grishin visited the miller
  (a Friend.Ivanov ≠ Profession.Miller ∧ a Friend.Grishin ≠ Profession.Miller) ∧
  -- Petrenko and Altman live in the same house as the postman
  (a Friend.Petrenko ≠ Profession.Postman ∧ a Friend.Altman ≠ Profession.Postman) ∧
  -- Sidorchuk attended Petrenko's wedding and the wedding of his barber friend's daughter
  (a Friend.Sidorchuk ≠ Profession.Barber ∧ a Friend.Petrenko ≠ Profession.Barber) ∧
  -- Ivanov and Petrenko often play dominoes with the carpenter and the painter
  (a Friend.Ivanov ≠ Profession.Carpenter ∧ a Friend.Ivanov ≠ Profession.Painter ∧
   a Friend.Petrenko ≠ Profession.Carpenter ∧ a Friend.Petrenko ≠ Profession.Painter) ∧
  -- Grishin and Altman go to their barber friend's shop to get shaved
  (a Friend.Grishin ≠ Profession.Barber ∧ a Friend.Altman ≠ Profession.Barber) ∧
  -- The postman shaves himself
  (∀ f, a f = Profession.Postman → a f ≠ Profession.Barber)

-- Theorem statement
theorem unique_assignment : 
  ∀ a : Friend → Profession, conditions a → a = assignment :=
sorry

end NUMINAMATH_CALUDE_unique_assignment_l1348_134889


namespace NUMINAMATH_CALUDE_largest_angle_in_quadrilateral_with_ratio_l1348_134823

/-- 
Given a quadrilateral divided into two triangles by a diagonal,
with the measures of the angles around this diagonal in the ratio 2:3:4:5,
prove that the largest of these angles is 900°/7.
-/
theorem largest_angle_in_quadrilateral_with_ratio (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- Angles are positive
  a + b + c + d = 360 →  -- Sum of angles around a point is 360°
  ∃ (x : ℝ), a = 2*x ∧ b = 3*x ∧ c = 4*x ∧ d = 5*x →  -- Angles are in ratio 2:3:4:5
  (max a (max b (max c d))) = 900 / 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_quadrilateral_with_ratio_l1348_134823


namespace NUMINAMATH_CALUDE_johns_hair_growth_l1348_134845

/-- Represents John's hair growth and haircut information -/
structure HairGrowthInfo where
  cutFrom : ℝ  -- Length of hair before cut
  cutTo : ℝ    -- Length of hair after cut
  baseCost : ℝ -- Base cost of a haircut
  tipPercent : ℝ -- Tip percentage
  yearlySpend : ℝ -- Total spent on haircuts per year

/-- Calculates the monthly hair growth rate -/
def monthlyGrowthRate (info : HairGrowthInfo) : ℝ :=
  -- Definition of the function
  sorry

/-- Theorem stating that John's hair grows 1.5 inches per month -/
theorem johns_hair_growth (info : HairGrowthInfo) 
  (h1 : info.cutFrom = 9)
  (h2 : info.cutTo = 6)
  (h3 : info.baseCost = 45)
  (h4 : info.tipPercent = 0.2)
  (h5 : info.yearlySpend = 324) :
  monthlyGrowthRate info = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_johns_hair_growth_l1348_134845


namespace NUMINAMATH_CALUDE_set_union_problem_l1348_134861

theorem set_union_problem (a b : ℕ) :
  let A : Set ℕ := {5, a + 1}
  let B : Set ℕ := {a, b}
  A ∩ B = {2} → A ∪ B = {1, 2, 5} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l1348_134861


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1348_134803

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^2 + 3

-- State the theorem
theorem derivative_f_at_zero : 
  deriv f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1348_134803


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_prism_l1348_134883

/-- Given a right square prism with height 4 and volume 16, 
    where all vertices are on the surface of a sphere,
    prove that the surface area of the sphere is 24π -/
theorem sphere_surface_area_from_prism (h : ℝ) (v : ℝ) (r : ℝ) : 
  h = 4 →
  v = 16 →
  v = h * r^2 →
  r^2 + h^2 / 4 + r^2 = (2 * r)^2 →
  4 * π * ((r^2 + h^2 / 4 + r^2) / 4) = 24 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_prism_l1348_134883


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1348_134838

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1348_134838


namespace NUMINAMATH_CALUDE_nested_square_roots_simplification_l1348_134878

theorem nested_square_roots_simplification :
  Real.sqrt (36 * Real.sqrt (18 * Real.sqrt 9)) = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_roots_simplification_l1348_134878


namespace NUMINAMATH_CALUDE_equation_solution_l1348_134877

/-- The set of solutions for the equation x! + y! = 8z + 2017 -/
def SolutionSet : Set (ℕ × ℕ × ℤ) :=
  {(1, 4, -249), (4, 1, -249), (1, 5, -237), (5, 1, -237)}

/-- The equation x! + y! = 8z + 2017 -/
def Equation (x y : ℕ) (z : ℤ) : Prop :=
  Nat.factorial x + Nat.factorial y = 8 * z + 2017

/-- z is an odd integer -/
def IsOdd (z : ℤ) : Prop :=
  ∃ k : ℤ, z = 2 * k + 1

theorem equation_solution :
  ∀ x y : ℕ, ∀ z : ℤ,
    Equation x y z ∧ IsOdd z ↔ (x, y, z) ∈ SolutionSet :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l1348_134877


namespace NUMINAMATH_CALUDE_leftover_coins_value_l1348_134844

def quarters_per_roll : ℕ := 30
def dimes_per_roll : ℕ := 40
def sally_quarters : ℕ := 101
def sally_dimes : ℕ := 173
def ben_quarters : ℕ := 150
def ben_dimes : ℕ := 195
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftover_coins_value :
  let total_quarters := sally_quarters + ben_quarters
  let total_dimes := sally_dimes + ben_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  let leftover_value := (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value
  leftover_value = 3.55 := by sorry

end NUMINAMATH_CALUDE_leftover_coins_value_l1348_134844


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1348_134852

def M : Set ℝ := {y | ∃ x, y = 3 - x^2}
def N : Set ℝ := {y | ∃ x, y = 2*x^2 - 1}

theorem intersection_of_M_and_N : M ∩ N = Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1348_134852


namespace NUMINAMATH_CALUDE_solution_set_equivalence_a_range_when_f_nonnegative_l1348_134859

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Part 1
theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 - a*x + 1 > 0 ↔ x < 1/3 ∨ x > 1/2) :=
sorry

-- Part 2
theorem a_range_when_f_nonnegative :
  ∀ a, (∀ x, f a (3-a) x ≥ 0) → a ∈ Set.Icc (-6) 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_a_range_when_f_nonnegative_l1348_134859


namespace NUMINAMATH_CALUDE_blue_balls_count_l1348_134811

theorem blue_balls_count (red green : ℕ) (p : ℚ) (blue : ℕ) : 
  red = 4 → 
  green = 2 → 
  p = 4/30 → 
  (red / (red + blue + green : ℚ)) * ((red - 1) / (red + blue + green - 1 : ℚ)) = p → 
  blue = 4 :=
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l1348_134811


namespace NUMINAMATH_CALUDE_initial_files_correct_l1348_134839

/-- The number of files Nancy had initially -/
def initial_files : ℕ := 80

/-- The number of files Nancy deleted -/
def deleted_files : ℕ := 31

/-- The number of folders Nancy ended up with -/
def num_folders : ℕ := 7

/-- The number of files in each folder -/
def files_per_folder : ℕ := 7

/-- Theorem stating that the initial number of files is correct -/
theorem initial_files_correct : 
  initial_files = deleted_files + num_folders * files_per_folder := by
  sorry

end NUMINAMATH_CALUDE_initial_files_correct_l1348_134839


namespace NUMINAMATH_CALUDE_geometric_progression_special_ratio_l1348_134819

/-- A geometric progression where each term is positive and any term is equal to the sum of the next three following terms has a common ratio that satisfies r³ + r² + r - 1 = 0. -/
theorem geometric_progression_special_ratio :
  ∀ (a : ℝ) (r : ℝ),
  (a > 0) →  -- First term is positive
  (r > 0) →  -- Common ratio is positive
  (∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)) →  -- Any term equals sum of next three
  r^3 + r^2 + r - 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_special_ratio_l1348_134819


namespace NUMINAMATH_CALUDE_no_real_roots_l1348_134867

theorem no_real_roots : ¬∃ x : ℝ, |2*x - 5| + |3*x - 7| + |5*x - 11| = 2015/2016 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l1348_134867


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l1348_134899

theorem factorization_of_difference_of_squares (a b : ℝ) :
  3 * a^2 - 3 * b^2 = 3 * (a + b) * (a - b) := by sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l1348_134899


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l1348_134885

/-- The value of p for a parabola y² = 2px (p > 0) where the distance between (-2, 3) and the focus is 5 -/
theorem parabola_focus_distance (p : ℝ) : 
  p > 0 → 
  (∃ (x y : ℝ), y^2 = 2*p*x) → 
  let focus := (p, 0)
  Real.sqrt ((p - (-2))^2 + (0 - 3)^2) = 5 → 
  p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l1348_134885


namespace NUMINAMATH_CALUDE_square_plus_integer_equality_find_integer_l1348_134825

theorem square_plus_integer_equality (y : ℝ) : ∃ k : ℤ, y^2 + 12*y + 40 = (y + 6)^2 + k := by
  sorry

theorem find_integer : ∃ k : ℤ, ∀ y : ℝ, y^2 + 12*y + 40 = (y + 6)^2 + k ∧ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_integer_equality_find_integer_l1348_134825


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1348_134851

theorem negation_of_universal_statement :
  (¬∀ x : ℝ, x^2 - 3*x + 2 > 0) ↔ (∃ x : ℝ, x^2 - 3*x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1348_134851


namespace NUMINAMATH_CALUDE_total_cookies_count_l1348_134810

def cookies_eaten : ℕ := 4
def cookies_to_brother : ℕ := 6
def friends_count : ℕ := 3
def cookies_per_friend : ℕ := 2
def team_members : ℕ := 10
def first_team_member_cookies : ℕ := 2
def team_cookie_difference : ℕ := 2

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem total_cookies_count :
  cookies_eaten +
  cookies_to_brother +
  (friends_count * cookies_per_friend) +
  arithmetic_sum first_team_member_cookies team_cookie_difference team_members =
  126 := by sorry

end NUMINAMATH_CALUDE_total_cookies_count_l1348_134810


namespace NUMINAMATH_CALUDE_zero_in_M_l1348_134876

def M : Set ℕ := {0, 1, 2}

theorem zero_in_M : 0 ∈ M := by sorry

end NUMINAMATH_CALUDE_zero_in_M_l1348_134876


namespace NUMINAMATH_CALUDE_neg_f_squared_increasing_nonpos_neg_f_squared_decreasing_nonneg_a_range_l1348_134800

noncomputable section

variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_additive : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂
axiom f_increasing_nonneg : ∀ x₁ x₂ : ℝ, x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ < x₂ → f x₁ < f x₂
axiom f_one_eq_two : f 1 = 2

-- State the theorems to be proved
theorem neg_f_squared_increasing_nonpos :
  ∀ x₁ x₂ : ℝ, x₁ ≤ 0 ∧ x₂ ≤ 0 ∧ x₁ < x₂ → -(f x₁)^2 < -(f x₂)^2 := by sorry

theorem neg_f_squared_decreasing_nonneg :
  ∀ x₁ x₂ : ℝ, x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ < x₂ → -(f x₁)^2 > -(f x₂)^2 := by sorry

theorem a_range :
  ∀ a : ℝ, f (2*a^2 - 1) + 2*f a - 6 < 0 ↔ -2 < a ∧ a < 1 := by sorry

end

end NUMINAMATH_CALUDE_neg_f_squared_increasing_nonpos_neg_f_squared_decreasing_nonneg_a_range_l1348_134800


namespace NUMINAMATH_CALUDE_marnie_bracelets_l1348_134858

/-- The number of bracelets that can be made from a given number of beads -/
def bracelets_from_beads (total_beads : ℕ) (beads_per_bracelet : ℕ) : ℕ :=
  total_beads / beads_per_bracelet

/-- The total number of beads from multiple bags -/
def total_beads_from_bags (bags_of_50 : ℕ) (bags_of_100 : ℕ) : ℕ :=
  bags_of_50 * 50 + bags_of_100 * 100

theorem marnie_bracelets : 
  let bags_of_50 : ℕ := 5
  let bags_of_100 : ℕ := 2
  let beads_per_bracelet : ℕ := 50
  let total_beads := total_beads_from_bags bags_of_50 bags_of_100
  bracelets_from_beads total_beads beads_per_bracelet = 9 := by
  sorry

end NUMINAMATH_CALUDE_marnie_bracelets_l1348_134858


namespace NUMINAMATH_CALUDE_triangle_value_l1348_134826

theorem triangle_value (triangle q p : ℤ) 
  (eq1 : triangle + q = 73)
  (eq2 : 2 * (triangle + q) + p = 172)
  (eq3 : p = 26) : 
  triangle = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l1348_134826


namespace NUMINAMATH_CALUDE_playground_to_landscape_ratio_l1348_134836

def rectangular_landscape (length breadth : ℝ) : Prop :=
  length = 8 * breadth ∧ length = 240

def playground_area : ℝ := 1200

theorem playground_to_landscape_ratio 
  (length breadth : ℝ) 
  (h : rectangular_landscape length breadth) : 
  playground_area / (length * breadth) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_playground_to_landscape_ratio_l1348_134836


namespace NUMINAMATH_CALUDE_alice_oranges_sold_l1348_134874

/-- Given that Alice sold twice as many oranges as Emily, and they sold 180 oranges in total,
    prove that Alice sold 120 oranges. -/
theorem alice_oranges_sold (emily : ℕ) (h1 : emily + 2 * emily = 180) : 2 * emily = 120 := by
  sorry

end NUMINAMATH_CALUDE_alice_oranges_sold_l1348_134874


namespace NUMINAMATH_CALUDE_z_is_real_z_is_imaginary_z_is_pure_imaginary_l1348_134801

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 7*a + 12) (a^2 - 5*a + 6)

-- Theorem for real values of z
theorem z_is_real (a : ℝ) : (z a).im = 0 ↔ a = 2 ∨ a = 3 := by sorry

-- Theorem for imaginary values of z
theorem z_is_imaginary (a : ℝ) : (z a).im ≠ 0 ↔ a ≠ 2 ∧ a ≠ 3 := by sorry

-- Theorem for pure imaginary values of z
theorem z_is_pure_imaginary (a : ℝ) : (z a).re = 0 ∧ (z a).im ≠ 0 ↔ a = 4 := by sorry

end NUMINAMATH_CALUDE_z_is_real_z_is_imaginary_z_is_pure_imaginary_l1348_134801


namespace NUMINAMATH_CALUDE_polynomial_value_at_one_l1348_134873

-- Define the polynomial P(x)
def P (r : ℝ) (x : ℝ) : ℝ := x^3 + x^2 - r^2*x - 2020

-- Define the roots of P(x)
variable (r s t : ℝ)

-- State the theorem
theorem polynomial_value_at_one (hr : P r r = 0) (hs : P r s = 0) (ht : P r t = 0) :
  P r 1 = -4038 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_one_l1348_134873


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l1348_134853

theorem smallest_five_digit_divisible_by_first_five_primes :
  (∀ n : ℕ, n ≥ 10000 ∧ n < 11550 → ¬(2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n)) ∧
  (11550 ≥ 10000) ∧
  (2 ∣ 11550) ∧ (3 ∣ 11550) ∧ (5 ∣ 11550) ∧ (7 ∣ 11550) ∧ (11 ∣ 11550) :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l1348_134853


namespace NUMINAMATH_CALUDE_triangle_properties_triangle_is_equilateral_l1348_134868

-- Define the triangle ABC
structure Triangle where
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C
  angleA : ℝ  -- measure of angle A
  angleB : ℝ  -- measure of angle B
  angleC : ℝ  -- measure of angle C

-- Define the theorem
theorem triangle_properties (t : Triangle)
  (h1 : (t.a + t.b + t.c) * (t.a - t.b - t.c) + 3 * t.b * t.c = 0)
  (h2 : t.a = 2 * t.c * Real.cos t.angleB) :
  t.angleA = π / 3 ∧ t.angleB = t.angleC := by
  sorry

-- Define what it means for a triangle to be equilateral
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Prove that the triangle is equilateral
theorem triangle_is_equilateral (t : Triangle)
  (h1 : (t.a + t.b + t.c) * (t.a - t.b - t.c) + 3 * t.b * t.c = 0)
  (h2 : t.a = 2 * t.c * Real.cos t.angleB) :
  is_equilateral t := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_triangle_is_equilateral_l1348_134868


namespace NUMINAMATH_CALUDE_least_k_factorial_divisible_by_315_l1348_134829

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

theorem least_k_factorial_divisible_by_315 :
  ∀ k : ℕ, k > 1 → (factorial k) % 315 = 0 → k ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_least_k_factorial_divisible_by_315_l1348_134829


namespace NUMINAMATH_CALUDE_company_profit_l1348_134817

theorem company_profit (a : ℝ) : 
  let october_profit := a
  let november_decrease := 0.06
  let december_increase := 0.10
  let november_profit := october_profit * (1 - november_decrease)
  let december_profit := november_profit * (1 + december_increase)
  december_profit = (1 - 0.06) * (1 + 0.10) * a := by
sorry

end NUMINAMATH_CALUDE_company_profit_l1348_134817


namespace NUMINAMATH_CALUDE_vector_dot_product_sum_l1348_134841

theorem vector_dot_product_sum (a b : ℝ × ℝ) : 
  a = (1/2, Real.sqrt 3/2) → 
  b = (-Real.sqrt 3/2, 1/2) → 
  (a.1 + b.1, a.2 + b.2) • a = 1 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_sum_l1348_134841


namespace NUMINAMATH_CALUDE_dining_group_size_l1348_134896

theorem dining_group_size (total_bill : ℝ) (tip_percentage : ℝ) (individual_payment : ℝ) : 
  total_bill = 139 ∧ tip_percentage = 0.1 ∧ individual_payment = 50.97 →
  ∃ n : ℕ, n = 3 ∧ n * individual_payment = total_bill * (1 + tip_percentage) := by
sorry

end NUMINAMATH_CALUDE_dining_group_size_l1348_134896


namespace NUMINAMATH_CALUDE_max_table_height_specific_triangle_l1348_134875

/-- Triangle ABC with sides a, b, and c -/
structure Triangle (α : Type*) [LinearOrderedField α] where
  a : α
  b : α
  c : α

/-- The maximum height of a table constructed from a triangle -/
def maxTableHeight {α : Type*} [LinearOrderedField α] (t : Triangle α) : α :=
  sorry

/-- The theorem stating the maximum height of the table -/
theorem max_table_height_specific_triangle :
  let t : Triangle ℝ := ⟨25, 29, 32⟩
  maxTableHeight t = 84 * Real.sqrt 1547 / 57 := by
  sorry

end NUMINAMATH_CALUDE_max_table_height_specific_triangle_l1348_134875


namespace NUMINAMATH_CALUDE_perpendicular_line_plane_necessary_not_sufficient_l1348_134897

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields

structure Plane3D where
  -- Add necessary fields

-- Define perpendicularity relations
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def perpendicular_plane_plane (p1 p2 : Plane3D) : Prop :=
  sorry

def plane_contains_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_line_plane_necessary_not_sufficient 
  (l : Line3D) (α : Plane3D) :
  (perpendicular_line_plane l α → 
    ∃ (p : Plane3D), plane_contains_line p l ∧ perpendicular_plane_plane p α) ∧
  ¬(∀ (l : Line3D) (α : Plane3D), 
    (∃ (p : Plane3D), plane_contains_line p l ∧ perpendicular_plane_plane p α) → 
    perpendicular_line_plane l α) :=
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_plane_necessary_not_sufficient_l1348_134897


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1348_134815

/-- Acme T-Shirt Company's pricing function -/
def acme_cost (x : ℕ) : ℝ := 60 + 8 * x

/-- Gamma T-Shirt Company's pricing function -/
def gamma_cost (x : ℕ) : ℝ := 12 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_for_acme : ℕ := 16

theorem acme_cheaper_at_min_shirts :
  acme_cost min_shirts_for_acme < gamma_cost min_shirts_for_acme ∧
  ∀ n : ℕ, n < min_shirts_for_acme → acme_cost n ≥ gamma_cost n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1348_134815


namespace NUMINAMATH_CALUDE_bernardo_always_wins_l1348_134894

def even_set : Finset ℕ := {2, 4, 6, 8, 10}
def odd_set : Finset ℕ := {1, 3, 5, 7, 9}

def form_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem bernardo_always_wins :
  ∀ (a b c : ℕ) (d e f : ℕ),
    a ∈ even_set → b ∈ even_set → c ∈ even_set →
    d ∈ odd_set → e ∈ odd_set → f ∈ odd_set →
    a ≠ b → b ≠ c → a ≠ c →
    d ≠ e → e ≠ f → d ≠ f →
    a > b → b > c →
    d > e → e > f →
    form_number a b c > form_number d e f :=
by sorry

end NUMINAMATH_CALUDE_bernardo_always_wins_l1348_134894


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_B_in_A_union_A_l1348_134872

-- Define set A as positive integers less than 9
def A : Set ℕ := {x | x > 0 ∧ x < 9}

-- Define set B
def B : Set ℕ := {1, 2, 3}

-- Define set A for the second part
def A' : Set ℝ := {x | -3 < x ∧ x < 1}

-- Define set B for the second part
def B' : Set ℝ := {x | 2 < x ∧ x < 10}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {1, 2, 3} := by sorry

-- Theorem for complement of B in A
theorem complement_B_in_A : A \ B = {4, 5, 6, 7, 8} := by sorry

-- Theorem for A' ∪ B'
theorem union_A'_B' : A' ∪ B' = {x | -3 < x ∧ x < 1 ∨ 2 < x ∧ x < 10} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_B_in_A_union_A_l1348_134872


namespace NUMINAMATH_CALUDE_parallel_lines_length_l1348_134818

/-- Represents a line segment with a length -/
structure Segment where
  length : ℝ

/-- Represents a geometric figure with parallel lines -/
structure GeometricFigure where
  AB : Segment
  CD : Segment
  EF : Segment
  GH : Segment
  parallel : AB.length / CD.length = CD.length / EF.length ∧ 
             CD.length / EF.length = EF.length / GH.length

theorem parallel_lines_length (fig : GeometricFigure) 
  (h1 : fig.AB.length = 180)
  (h2 : fig.CD.length = 120)
  (h3 : fig.GH.length = 80) :
  fig.EF.length = 24 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_length_l1348_134818


namespace NUMINAMATH_CALUDE_sum_even_coefficients_l1348_134813

open BigOperators

theorem sum_even_coefficients (n : ℕ) (a : ℕ → ℝ) :
  (∀ x, (1 + x + x^2)^n = ∑ i in Finset.range (2*n + 1), a i * x^i) →
  ∑ i in Finset.range (n + 1), a (2*i) = (3^n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_coefficients_l1348_134813


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1348_134850

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1348_134850


namespace NUMINAMATH_CALUDE_f_min_value_l1348_134847

/-- The function f(x) = -x³ + 3x² + 9x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

/-- The theorem stating that if f(x) has a maximum value of 2 on [-2, -1], 
    then its minimum value on this interval is -5 -/
theorem f_min_value (a : ℝ) : 
  (∃ x ∈ Set.Icc (-2) (-1), ∀ y ∈ Set.Icc (-2) (-1), f a y ≤ f a x) ∧ 
  (∃ x ∈ Set.Icc (-2) (-1), f a x = 2) →
  (∃ x ∈ Set.Icc (-2) (-1), ∀ y ∈ Set.Icc (-2) (-1), f a x ≤ f a y) ∧
  (∃ x ∈ Set.Icc (-2) (-1), f a x = -5) :=
by sorry


end NUMINAMATH_CALUDE_f_min_value_l1348_134847


namespace NUMINAMATH_CALUDE_infinite_solutions_l1348_134835

/-- A system of linear equations -/
structure LinearSystem where
  eq1 : ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ
  eq3 : ℝ → ℝ → ℝ

/-- The specific system of equations from the problem -/
def problemSystem : LinearSystem where
  eq1 := fun x y => 3 * x - 4 * y - 5
  eq2 := fun x y => 6 * x - 8 * y - 10
  eq3 := fun x y => 9 * x - 12 * y - 15

/-- A solution to the system is a pair (x, y) that satisfies all equations -/
def isSolution (system : LinearSystem) (x y : ℝ) : Prop :=
  system.eq1 x y = 0 ∧ system.eq2 x y = 0 ∧ system.eq3 x y = 0

/-- The theorem stating that the system has infinitely many solutions -/
theorem infinite_solutions (system : LinearSystem) 
  (h1 : ∀ x y, system.eq2 x y = 2 * system.eq1 x y)
  (h2 : ∀ x y, system.eq3 x y = 3 * system.eq1 x y) :
  ∃ f : ℝ → ℝ × ℝ, ∀ t, isSolution system (f t).1 (f t).2 :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_l1348_134835


namespace NUMINAMATH_CALUDE_triple_cheese_ratio_undetermined_l1348_134808

/-- Represents the types of pizzas available --/
inductive PizzaType
| TripleCheese
| MeatLovers

/-- Represents the pricing structure for pizzas --/
structure PizzaPricing where
  standardPrice : ℕ
  meatLoversOffer : ℕ → ℕ  -- Function that takes number of pizzas and returns number to pay for
  tripleCheesePricing : ℕ → ℕ  -- Function for triple cheese pizzas (unknown specifics)

/-- Represents the order details --/
structure Order where
  tripleCheeseCount : ℕ
  meatLoversCount : ℕ

/-- Calculates the total cost of an order --/
def calculateTotalCost (pricing : PizzaPricing) (order : Order) : ℕ :=
  pricing.tripleCheesePricing order.tripleCheeseCount * pricing.standardPrice +
  pricing.meatLoversOffer order.meatLoversCount * pricing.standardPrice

/-- Theorem stating that the ratio for triple cheese pizzas cannot be determined --/
theorem triple_cheese_ratio_undetermined (pricing : PizzaPricing) (order : Order) :
  pricing.standardPrice = 5 ∧
  order.meatLoversCount = 9 ∧
  calculateTotalCost pricing order = 55 →
  ¬ ∃ (r : ℚ), r > 0 ∧ r < 1 ∧ ∀ (n : ℕ), pricing.tripleCheesePricing n = n - ⌊n * r⌋ :=
by sorry

end NUMINAMATH_CALUDE_triple_cheese_ratio_undetermined_l1348_134808


namespace NUMINAMATH_CALUDE_range_of_k_for_inequality_l1348_134895

/-- Given functions f and g, prove the range of k for which g(x) ≥ k(x) holds. -/
theorem range_of_k_for_inequality (f g : ℝ → ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f x = Real.log x) →
  (∀ x : ℝ, g x = x - 1) →
  (∀ x : ℝ, x ≥ 0 → g x ≥ k * x) ↔ k ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_for_inequality_l1348_134895


namespace NUMINAMATH_CALUDE_min_sum_given_product_l1348_134804

theorem min_sum_given_product (x y : ℝ) : 
  x > 0 → y > 0 → (x - 1) * (y - 1) = 1 → x + y ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (x - 1) * (y - 1) = 1 ∧ x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l1348_134804


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficients_l1348_134843

theorem polynomial_factor_coefficients :
  ∀ (a b : ℤ),
  (∃ (c d : ℤ), ∀ (x : ℚ),
    a * x^4 + b * x^3 + 32 * x^2 - 16 * x + 6 = (3 * x^2 - 2 * x + 1) * (c * x^2 + d * x + 6)) →
  a = 18 ∧ b = -24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficients_l1348_134843


namespace NUMINAMATH_CALUDE_billy_lemon_heads_l1348_134802

theorem billy_lemon_heads (total_lemon_heads : ℕ) (lemon_heads_per_friend : ℕ) (h1 : total_lemon_heads = 72) (h2 : lemon_heads_per_friend = 12) :
  total_lemon_heads / lemon_heads_per_friend = 6 := by
sorry

end NUMINAMATH_CALUDE_billy_lemon_heads_l1348_134802


namespace NUMINAMATH_CALUDE_problem_solution_l1348_134848

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1348_134848


namespace NUMINAMATH_CALUDE_books_printing_sheets_l1348_134807

/-- The number of sheets of paper used to print books with the given specifications. -/
def sheets_used (num_books : ℕ) (pages_per_book : ℕ) (pages_per_sheet : ℕ) : ℕ :=
  (num_books * pages_per_book) / pages_per_sheet

/-- Theorem stating the number of sheets used for printing the books. -/
theorem books_printing_sheets :
  sheets_used 2 600 8 = 150 := by
  sorry

end NUMINAMATH_CALUDE_books_printing_sheets_l1348_134807


namespace NUMINAMATH_CALUDE_vector_computation_l1348_134865

theorem vector_computation :
  let v1 : Fin 3 → ℝ := ![3, -5, 1]
  let v2 : Fin 3 → ℝ := ![-1, 4, -2]
  let v3 : Fin 3 → ℝ := ![2, -1, 3]
  2 • v1 + 3 • v2 - v3 = ![1, 3, -7] := by
  sorry

end NUMINAMATH_CALUDE_vector_computation_l1348_134865


namespace NUMINAMATH_CALUDE_tree_planting_campaign_l1348_134842

theorem tree_planting_campaign (february_trees : ℕ) (planned_trees : ℕ) : 
  february_trees = planned_trees * 19 / 20 →
  (planned_trees * 11 / 10 : ℚ) = 
    (planned_trees * 11 / 10 : ℕ) ∧ planned_trees > 0 →
  ∃ (total_trees : ℕ), total_trees = planned_trees * 11 / 10 :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_campaign_l1348_134842


namespace NUMINAMATH_CALUDE_sqrt_sum_fourth_power_divided_by_two_l1348_134831

theorem sqrt_sum_fourth_power_divided_by_two :
  Real.sqrt (4^4 + 4^4 + 4^4) / 2 = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fourth_power_divided_by_two_l1348_134831


namespace NUMINAMATH_CALUDE_arman_two_week_earnings_l1348_134805

/-- Calculates Arman's earnings for two weeks given his work hours and rates --/
theorem arman_two_week_earnings : 
  let first_week_hours : ℕ := 35
  let first_week_rate : ℚ := 10
  let second_week_hours : ℕ := 40
  let second_week_rate_increase : ℚ := 0.5
  let second_week_rate : ℚ := first_week_rate + second_week_rate_increase
  let first_week_earnings : ℚ := first_week_hours * first_week_rate
  let second_week_earnings : ℚ := second_week_hours * second_week_rate
  let total_earnings : ℚ := first_week_earnings + second_week_earnings
  total_earnings = 770 := by sorry

end NUMINAMATH_CALUDE_arman_two_week_earnings_l1348_134805


namespace NUMINAMATH_CALUDE_scientific_notation_86560_l1348_134821

theorem scientific_notation_86560 : 
  86560 = 8.656 * (10 ^ 4) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_86560_l1348_134821


namespace NUMINAMATH_CALUDE_twenty_mps_equals_72_kmph_l1348_134816

/-- Conversion from meters per second to kilometers per hour -/
def mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * 3.6

/-- Theorem: 20 mps is equal to 72 kmph -/
theorem twenty_mps_equals_72_kmph :
  mps_to_kmph 20 = 72 := by
  sorry

#eval mps_to_kmph 20

end NUMINAMATH_CALUDE_twenty_mps_equals_72_kmph_l1348_134816


namespace NUMINAMATH_CALUDE_game_attendance_l1348_134869

theorem game_attendance : ∃ (total : ℕ), 
  (total : ℚ) * (40 / 100) + (total : ℚ) * (34 / 100) + 3 = total ∧ total = 12 := by
  sorry

end NUMINAMATH_CALUDE_game_attendance_l1348_134869


namespace NUMINAMATH_CALUDE_sqrt_13_minus_3_bounds_l1348_134871

theorem sqrt_13_minus_3_bounds : 0 < Real.sqrt 13 - 3 ∧ Real.sqrt 13 - 3 < 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_13_minus_3_bounds_l1348_134871


namespace NUMINAMATH_CALUDE_number_of_divisors_36_l1348_134840

theorem number_of_divisors_36 : Finset.card (Nat.divisors 36) = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_36_l1348_134840


namespace NUMINAMATH_CALUDE_function_value_at_symmetry_point_l1348_134887

/-- Given a function f(x) = 3cos(ωx + φ) that satisfies f(π/6 + x) = f(π/6 - x) for all x,
    prove that f(π/6) equals either 3 or -3 -/
theorem function_value_at_symmetry_point 
  (ω φ : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = 3 * Real.cos (ω * x + φ))
  (h2 : ∀ x, f (π/6 + x) = f (π/6 - x)) :
  f (π/6) = 3 ∨ f (π/6) = -3 :=
sorry

end NUMINAMATH_CALUDE_function_value_at_symmetry_point_l1348_134887


namespace NUMINAMATH_CALUDE_walking_days_problem_l1348_134892

/-- 
Given:
- Jackie walks 2 miles per day
- Jessie walks 1.5 miles per day
- Over d days, Jackie walks 3 miles more than Jessie

Prove that d = 6
-/
theorem walking_days_problem (d : ℝ) 
  (h1 : 2 * d = 1.5 * d + 3) : d = 6 := by
  sorry

end NUMINAMATH_CALUDE_walking_days_problem_l1348_134892


namespace NUMINAMATH_CALUDE_net_rate_of_pay_l1348_134882

/-- Calculates the net rate of pay for a driver given travel conditions and expenses. -/
theorem net_rate_of_pay
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (gasoline_cost : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_rate = 0.60)
  (h5 : gasoline_cost = 2.50) :
  (pay_rate * speed * travel_time - (speed * travel_time / fuel_efficiency) * gasoline_cost) / travel_time = 25 := by
  sorry


end NUMINAMATH_CALUDE_net_rate_of_pay_l1348_134882


namespace NUMINAMATH_CALUDE_last_digit_of_2_pow_2010_l1348_134898

-- Define the function that gives the last digit of 2^n
def lastDigitOf2Pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => unreachable!

-- Theorem statement
theorem last_digit_of_2_pow_2010 : lastDigitOf2Pow 2010 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_2_pow_2010_l1348_134898


namespace NUMINAMATH_CALUDE_vector_sum_equality_l1348_134880

variable (V : Type*) [AddCommGroup V]

theorem vector_sum_equality (a : V) : a + 2 • a = 3 • a := by sorry

end NUMINAMATH_CALUDE_vector_sum_equality_l1348_134880


namespace NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l1348_134846

theorem sum_of_numbers_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a →
  c = 4 * a →
  a^2 + b^2 + c^2 = 4725 →
  a + b + c = 105 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l1348_134846


namespace NUMINAMATH_CALUDE_dining_bill_proof_l1348_134884

theorem dining_bill_proof (num_people : ℕ) (individual_payment : ℚ) (tip_percentage : ℚ) 
  (h1 : num_people = 7)
  (h2 : individual_payment = 21.842857142857145)
  (h3 : tip_percentage = 1/10) :
  (num_people : ℚ) * individual_payment / (1 + tip_percentage) = 139 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_proof_l1348_134884


namespace NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l1348_134806

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem: The y-intercept of the common external tangent of two specific circles is 11 -/
theorem common_external_tangent_y_intercept :
  let c1 : Circle := { center := (1, 3), radius := 3 }
  let c2 : Circle := { center := (10, 6), radius := 5 }
  ∃ m b : ℝ, b = 11 ∧ 
    (∀ x y : ℝ, y = m * x + b → 
      (((x - c1.center.1) ^ 2 + (y - c1.center.2) ^ 2 = c1.radius ^ 2) ∨
       ((x - c2.center.1) ^ 2 + (y - c2.center.2) ^ 2 = c2.radius ^ 2))) :=
by
  sorry


end NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l1348_134806


namespace NUMINAMATH_CALUDE_correct_regression_sequence_l1348_134820

-- Define the steps of linear regression analysis
inductive RegressionStep
  | InterpretEquation
  | CollectData
  | CalculateEquation
  | CalculateCorrelation
  | DrawScatterPlot

-- Define a type for sequences of regression steps
def RegressionSequence := List RegressionStep

-- Define the correct sequence
def correctSequence : RegressionSequence :=
  [RegressionStep.CollectData,
   RegressionStep.DrawScatterPlot,
   RegressionStep.CalculateCorrelation,
   RegressionStep.CalculateEquation,
   RegressionStep.InterpretEquation]

-- Theorem stating that the defined sequence is correct
theorem correct_regression_sequence :
  correctSequence = [RegressionStep.CollectData,
                     RegressionStep.DrawScatterPlot,
                     RegressionStep.CalculateCorrelation,
                     RegressionStep.CalculateEquation,
                     RegressionStep.InterpretEquation] :=
by sorry

end NUMINAMATH_CALUDE_correct_regression_sequence_l1348_134820


namespace NUMINAMATH_CALUDE_first_group_size_l1348_134870

/-- The number of men in the first group -/
def M : ℕ := sorry

/-- The length of the wall built by the first group -/
def length1 : ℝ := 66

/-- The number of days taken by the first group -/
def days1 : ℕ := 8

/-- The number of men in the second group -/
def men2 : ℕ := 86

/-- The length of the wall built by the second group -/
def length2 : ℝ := 283.8

/-- The number of days taken by the second group -/
def days2 : ℕ := 8

/-- The work done is directly proportional to the number of men and the length of the wall -/
axiom work_proportional : ∀ (men : ℕ) (length : ℝ) (days : ℕ), 
  (men : ℝ) * length / days = (M : ℝ) * length1 / days1

theorem first_group_size : 
  ∃ (m : ℕ), (m : ℝ) ≥ 368.5 ∧ (m : ℝ) < 369.5 ∧ M = m :=
sorry

end NUMINAMATH_CALUDE_first_group_size_l1348_134870


namespace NUMINAMATH_CALUDE_parallelogram_D_coordinates_l1348_134862

structure Point where
  x : ℝ
  y : ℝ

def Parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x, B.y - A.y) = (D.x - C.x, D.y - C.y) ∧
  (C.x - B.x, C.y - B.y) = (A.x - D.x, A.y - D.y)

theorem parallelogram_D_coordinates :
  let A : Point := ⟨-1, 2⟩
  let B : Point := ⟨0, 0⟩
  let C : Point := ⟨1, 7⟩
  let D : Point := ⟨0, 9⟩
  Parallelogram A B C D → D = ⟨0, 9⟩ := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_D_coordinates_l1348_134862


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l1348_134893

theorem factorial_equation_solutions :
  ∀ (x y : ℕ) (z : ℤ),
    (Odd z) →
    (Nat.factorial x + Nat.factorial y = 24 * z + 2017) →
    ((x = 1 ∧ y = 4 ∧ z = -83) ∨
     (x = 4 ∧ y = 1 ∧ z = -83) ∨
     (x = 1 ∧ y = 5 ∧ z = -79) ∨
     (x = 5 ∧ y = 1 ∧ z = -79)) :=
by sorry


end NUMINAMATH_CALUDE_factorial_equation_solutions_l1348_134893


namespace NUMINAMATH_CALUDE_total_eggs_collected_l1348_134879

/-- The number of dozen eggs Benjamin collects per day -/
def benjamin_eggs : ℕ := 6

/-- The number of dozen eggs Carla collects per day -/
def carla_eggs : ℕ := 3 * benjamin_eggs

/-- The number of dozen eggs Trisha collects per day -/
def trisha_eggs : ℕ := benjamin_eggs - 4

/-- The total number of dozen eggs collected by Benjamin, Carla, and Trisha -/
def total_eggs : ℕ := benjamin_eggs + carla_eggs + trisha_eggs

theorem total_eggs_collected :
  total_eggs = 26 := by sorry

end NUMINAMATH_CALUDE_total_eggs_collected_l1348_134879


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1348_134822

-- System 1
theorem system_one_solution (x y : ℝ) : 
  x = 2 * y ∧ 3 * x - 2 * y = 8 → x = 4 ∧ y = 2 := by sorry

-- System 2
theorem system_two_solution (x y : ℝ) : 
  3 * x + 2 * y = 4 ∧ x / 2 - (y + 1) / 3 = 1 → x = 2 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1348_134822


namespace NUMINAMATH_CALUDE_additional_as_needed_l1348_134881

/-- Given initial grades and A's, and subsequent increases in A proportion,
    calculate additional A's needed for a further increase. -/
theorem additional_as_needed
  (n k : ℕ)  -- Initial number of grades and A's
  (h1 : (k + 1 : ℚ) / (n + 1) - k / n = 15 / 100)  -- First increase
  (h2 : (k + 2 : ℚ) / (n + 2) - (k + 1) / (n + 1) = 1 / 10)  -- Second increase
  (h3 : (k + 2 : ℚ) / (n + 2) = 2 / 3)  -- Current proportion
  : ∃ m : ℕ, (k + 2 + m : ℚ) / (n + 2 + m) = 7 / 10 ∧ m = 4 := by
  sorry


end NUMINAMATH_CALUDE_additional_as_needed_l1348_134881


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1348_134814

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 1 = 0 →
  q^3 - 8*q^2 + 10*q - 1 = 0 →
  r^3 - 8*r^2 + 10*r - 1 = 0 →
  p/(q*r + 1) + q/(p*r + 1) + r/(p*q + 1) = 113/20 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1348_134814


namespace NUMINAMATH_CALUDE_bicycle_speeds_l1348_134860

/-- Represents a bicycle with front and rear gears -/
structure Bicycle where
  front_gears : Nat
  rear_gears : Nat

/-- Calculates the number of unique speeds for a bicycle -/
def unique_speeds (b : Bicycle) : Nat :=
  b.front_gears * b.rear_gears - b.rear_gears

/-- Theorem stating that a bicycle with 3 front gears and 4 rear gears has 8 unique speeds -/
theorem bicycle_speeds :
  ∃ (b : Bicycle), b.front_gears = 3 ∧ b.rear_gears = 4 ∧ unique_speeds b = 8 :=
by
  sorry

#eval unique_speeds ⟨3, 4⟩

end NUMINAMATH_CALUDE_bicycle_speeds_l1348_134860


namespace NUMINAMATH_CALUDE_sum_of_equal_expressions_l1348_134864

theorem sum_of_equal_expressions 
  (a b c d e f g h i : ℤ) 
  (eq1 : a + b + c + d = d + e + f + g) 
  (eq2 : d + e + f + g = g + h + i) 
  (ha : a = 4) 
  (hg : g = 13) 
  (hh : h = 6) : 
  ∃ S : ℤ, (a + b + c + d = S) ∧ (d + e + f + g = S) ∧ (g + h + i = S) ∧ (S = 19 + i) :=
sorry

end NUMINAMATH_CALUDE_sum_of_equal_expressions_l1348_134864


namespace NUMINAMATH_CALUDE_jennifer_fruit_count_l1348_134891

def fruit_problem (pears oranges : ℕ) : Prop :=
  let apples := 2 * pears
  let cherries := oranges / 2
  let grapes := 3 * apples
  let initial_total := pears + oranges + apples + cherries + grapes
  let pineapples := initial_total
  let remaining_pears := pears - 3
  let remaining_oranges := oranges - 5
  let remaining_apples := apples - 5
  let remaining_cherries := cherries - 7
  let remaining_grapes := grapes - 3
  let remaining_before_pineapples := remaining_pears + remaining_oranges + remaining_apples + remaining_cherries + remaining_grapes
  let remaining_pineapples := pineapples - (pineapples / 2)
  remaining_before_pineapples + remaining_pineapples = 247

theorem jennifer_fruit_count : fruit_problem 15 30 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_fruit_count_l1348_134891


namespace NUMINAMATH_CALUDE_two_lines_properties_l1348_134857

/-- Two lines l₁ and l₂ in the xy-plane -/
structure TwoLines (m n : ℝ) :=
  (l₁ : ℝ → ℝ → Prop)
  (l₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ m * x + 8 * y + n = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 2 * x + m * y - 1 = 0)

/-- The lines intersect at point P(m, -1) -/
def intersect_at_P (l : TwoLines m n) : Prop :=
  l.l₁ m (-1) ∧ l.l₂ m (-1)

/-- The lines are parallel -/
def parallel (l : TwoLines m n) : Prop :=
  m / 2 = 8 / m ∧ m / 2 ≠ n / (-1)

/-- The lines are perpendicular -/
def perpendicular (l : TwoLines m n) : Prop :=
  m = 0 ∨ (m ≠ 0 ∧ (-m / 8) * (1 / m) = -1)

/-- Main theorem about the properties of the two lines -/
theorem two_lines_properties (m n : ℝ) (l : TwoLines m n) :
  (intersect_at_P l → m = 1 ∧ n = 7) ∧
  (parallel l → (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2)) ∧
  (perpendicular l → m = 0) :=
sorry

end NUMINAMATH_CALUDE_two_lines_properties_l1348_134857


namespace NUMINAMATH_CALUDE_email_difference_l1348_134812

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 9

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 10

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 7

/-- The difference between the number of emails Jack received in the morning and evening -/
theorem email_difference : morning_emails - evening_emails = 2 := by
  sorry

end NUMINAMATH_CALUDE_email_difference_l1348_134812


namespace NUMINAMATH_CALUDE_tan_30_plus_2cos_30_l1348_134890

theorem tan_30_plus_2cos_30 : Real.tan (π / 6) + 2 * Real.cos (π / 6) = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_plus_2cos_30_l1348_134890


namespace NUMINAMATH_CALUDE_no_arithmetic_sequence_with_arithmetic_digit_sum_l1348_134855

/-- An arithmetic sequence of positive integers. -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ (a₀ d : ℕ), ∀ n, a n = a₀ + n * d

/-- The sum of digits of a natural number. -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that no infinite arithmetic sequence of distinct positive integers
    exists such that the sum of digits of each term also forms an arithmetic sequence. -/
theorem no_arithmetic_sequence_with_arithmetic_digit_sum :
  ¬ ∃ (a : ℕ → ℕ),
    ArithmeticSequence a ∧
    (∀ n m, n ≠ m → a n ≠ a m) ∧
    ArithmeticSequence (λ n => sumOfDigits (a n)) :=
sorry

end NUMINAMATH_CALUDE_no_arithmetic_sequence_with_arithmetic_digit_sum_l1348_134855


namespace NUMINAMATH_CALUDE_power_division_multiplication_l1348_134863

theorem power_division_multiplication (x : ℕ) : (3^18 / 27^2) * 7 = 3720087 := by
  sorry

end NUMINAMATH_CALUDE_power_division_multiplication_l1348_134863


namespace NUMINAMATH_CALUDE_part_one_part_two_l1348_134824

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (a : ℝ) : (∀ x, f a x ≤ 3 ↔ x ∈ Set.Icc (-1) 5) → a = 2 := by sorry

-- Part II
theorem part_two : 
  (∀ x : ℝ, f 2 x + f 2 (x + 5) ≥ 5) ∧ 
  (∀ m : ℝ, (∀ x : ℝ, f 2 x + f 2 (x + 5) ≥ m) → m ≤ 5) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1348_134824


namespace NUMINAMATH_CALUDE_base_conversion_and_arithmetic_l1348_134837

-- Define the base conversion functions
def to_base_10 (digits : List Nat) (base : Nat) : Rat :=
  (digits.reverse.enum.map (λ (i, d) => d * base^i)).sum

-- Define the given numbers in their respective bases
def num1 : Rat := 2468
def num2 : Rat := to_base_10 [1, 2, 1] 3
def num3 : Rat := to_base_10 [6, 5, 4, 3] 7
def num4 : Rat := to_base_10 [6, 7, 8, 9] 9

-- State the theorem
theorem base_conversion_and_arithmetic :
  num1 / num2 + num3 - num4 = -5857.75 := by sorry

end NUMINAMATH_CALUDE_base_conversion_and_arithmetic_l1348_134837


namespace NUMINAMATH_CALUDE_complex_fraction_difference_l1348_134834

theorem complex_fraction_difference (i : ℂ) (h : i^2 = -1) :
  (1 + 2*i) / (1 - 2*i) - (1 - 2*i) / (1 + 2*i) = 8/5 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_difference_l1348_134834


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1348_134854

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem sixth_term_of_geometric_sequence (a₁ a₄ : ℝ) (h₁ : a₁ = 8) (h₂ : a₄ = 64) :
  ∃ r : ℝ, geometric_sequence a₁ r 4 = a₄ ∧ geometric_sequence a₁ r 6 = 256 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1348_134854


namespace NUMINAMATH_CALUDE_xy_value_l1348_134833

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 16)
  (h2 : (27:ℝ)^(x+y) / (9:ℝ)^(5*y) = 729) : 
  x * y = 96 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l1348_134833


namespace NUMINAMATH_CALUDE_equation_holds_l1348_134866

theorem equation_holds : (8 - 2) + 5 - (3 - 1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l1348_134866


namespace NUMINAMATH_CALUDE_no_simultaneous_solution_l1348_134886

theorem no_simultaneous_solution : ¬∃ x : ℝ, (5 * x^2 - 7 * x + 1 < 0) ∧ (x^2 - 9 * x + 30 < 0) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_solution_l1348_134886

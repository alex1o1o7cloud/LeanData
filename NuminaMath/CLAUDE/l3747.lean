import Mathlib

namespace NUMINAMATH_CALUDE_vector_properties_l3747_374729

/-- Given vectors in R^2 -/
def a : ℝ × ℝ := (2, -1)
def b (m : ℝ) : ℝ × ℝ := (m, 2)

/-- Parallel vectors condition -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

/-- Perpendicular vectors condition -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Vector addition -/
def add (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2)

/-- Scalar multiplication -/
def smul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

/-- Vector subtraction -/
def sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  add v (smul (-1) w)

/-- Squared norm of a vector -/
def norm_sq (v : ℝ × ℝ) : ℝ :=
  v.1^2 + v.2^2

theorem vector_properties (m : ℝ) :
  (parallel a (b m) ↔ m = -4) ∧
  (perpendicular a (b m) ↔ m = 1) ∧
  ¬(norm_sq (sub (smul 2 a) (b m)) = norm_sq (add a (b m)) → m = 1) ∧
  ¬(norm_sq (add a (b m)) = norm_sq a → m = -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l3747_374729


namespace NUMINAMATH_CALUDE_round_trip_percentage_is_25_percent_l3747_374764

/-- Represents the percentage of ship passengers holding round-trip tickets -/
def round_trip_percentage : ℝ := 25

/-- Represents the percentage of all passengers who held round-trip tickets and took their cars aboard -/
def round_trip_with_car_percentage : ℝ := 20

/-- Represents the percentage of round-trip ticket holders who did not take their cars aboard -/
def round_trip_without_car_percentage : ℝ := 20

/-- Proves that the percentage of ship passengers holding round-trip tickets is 25% -/
theorem round_trip_percentage_is_25_percent :
  round_trip_percentage = 25 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_percentage_is_25_percent_l3747_374764


namespace NUMINAMATH_CALUDE_george_second_half_correct_l3747_374761

def trivia_game (first_half_correct : ℕ) (points_per_question : ℕ) (final_score : ℕ) : ℕ :=
  (final_score - first_half_correct * points_per_question) / points_per_question

theorem george_second_half_correct :
  trivia_game 6 3 30 = 4 :=
sorry

end NUMINAMATH_CALUDE_george_second_half_correct_l3747_374761


namespace NUMINAMATH_CALUDE_student_council_selections_l3747_374785

-- Define the number of students
def n : ℕ := 6

-- Define the number of ways to select a two-person team
def two_person_selections : ℕ := 15

-- Define the number of ways to select a three-person team
def three_person_selections : ℕ := 20

-- Theorem statement
theorem student_council_selections :
  (Nat.choose n 2 = two_person_selections) →
  (Nat.choose n 3 = three_person_selections) :=
by sorry

end NUMINAMATH_CALUDE_student_council_selections_l3747_374785


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3747_374787

-- Define the ellipse C
def C (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the points
def O : ℝ × ℝ := (0, 0)
def F (c : ℝ) : ℝ × ℝ := (c, 0)
def A (a : ℝ) : ℝ × ℝ := (-a, 0)
def B (a : ℝ) : ℝ × ℝ := (a, 0)
def P (x y : ℝ) : ℝ × ℝ := (x, y)
def M (x y : ℝ) : ℝ × ℝ := (x, y)
def E (y : ℝ) : ℝ × ℝ := (0, y)

-- Define the lines
def line_l (a c : ℝ) (x y : ℝ) : Prop := 
  ∃ (y_M : ℝ), y - y_M = (y_M / (c + a)) * (x - c)

def line_BM (a c : ℝ) (x y y_E : ℝ) : Prop := 
  y - y_E/2 = -(y_E/2) / (a + c) * x

-- Main theorem
theorem ellipse_eccentricity 
  (a b c : ℝ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : c < a)
  (h3 : ∃ (x y : ℝ), C a b x y ∧ P x y = (x, y) ∧ x = c)
  (h4 : ∃ (x y y_E : ℝ), line_l a c x y ∧ line_BM a c x y y_E ∧ M x y = (x, y))
  : c/a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3747_374787


namespace NUMINAMATH_CALUDE_intersection_point_correct_l3747_374738

/-- The point of intersection of two lines -/
def intersection_point : ℚ × ℚ := (-9/13, 32/13)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: 2y = -10x - 2 -/
def line2 (x y : ℚ) : Prop := 2 * y = -10 * x - 2

theorem intersection_point_correct :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l3747_374738


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l3747_374724

theorem circle_line_distance_range (b : ℝ) : 
  (∃! (p q : ℝ × ℝ), (p.1 - 1)^2 + (p.2 - 1)^2 = 4 ∧ 
                      (q.1 - 1)^2 + (q.2 - 1)^2 = 4 ∧ 
                      p ≠ q ∧
                      (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 4 → 
                        (|y - (x + b)| / Real.sqrt 2 = 1 → (x, y) = p ∨ (x, y) = q))) →
  b ∈ Set.union (Set.Ioo (-3 * Real.sqrt 2) (-Real.sqrt 2)) 
                (Set.Ioo (Real.sqrt 2) (3 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_range_l3747_374724


namespace NUMINAMATH_CALUDE_camp_wonka_marshmallows_l3747_374793

theorem camp_wonka_marshmallows : 
  ∀ (total_campers : ℕ) 
    (boys_fraction girls_fraction : ℚ) 
    (boys_toast_percent girls_toast_percent : ℚ),
  total_campers = 96 →
  boys_fraction = 2/3 →
  girls_fraction = 1/3 →
  boys_toast_percent = 1/2 →
  girls_toast_percent = 3/4 →
  (boys_fraction * ↑total_campers * boys_toast_percent + 
   girls_fraction * ↑total_campers * girls_toast_percent : ℚ) = 56 := by
sorry

end NUMINAMATH_CALUDE_camp_wonka_marshmallows_l3747_374793


namespace NUMINAMATH_CALUDE_equation_solutions_l3747_374748

theorem equation_solutions : ∃ (x₁ x₂ : ℝ) (z₁ z₂ : ℂ),
  (∀ x : ℝ, (15*x - x^2)/(x + 2) * (x + (15 - x)/(x + 2)) = 48 ↔ x = x₁ ∨ x = x₂) ∧
  (∀ z : ℂ, (15*z - z^2)/(z + 2) * (z + (15 - z)/(z + 2)) = 48 ↔ z = z₁ ∨ z = z₂) ∧
  x₁ = 4 ∧ x₂ = -3 ∧ z₁ = 3 + Complex.I * Real.sqrt 2 ∧ z₂ = 3 - Complex.I * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3747_374748


namespace NUMINAMATH_CALUDE_max_g_value_l3747_374756

theorem max_g_value (t : Real) (h : t ∈ Set.Icc 0 Real.pi) : 
  let g := fun (t : Real) => (4 * Real.cos t + 5) * (1 - Real.cos t)^2
  ∃ (max_val : Real), max_val = 27/4 ∧ 
    (∀ s, s ∈ Set.Icc 0 Real.pi → g s ≤ max_val) ∧
    g (2 * Real.pi / 3) = max_val :=
by sorry

end NUMINAMATH_CALUDE_max_g_value_l3747_374756


namespace NUMINAMATH_CALUDE_work_done_equals_21_l3747_374778

def force : ℝ × ℝ := (5, 2)
def point_A : ℝ × ℝ := (-1, 3)
def point_B : ℝ × ℝ := (2, 6)

theorem work_done_equals_21 : 
  let displacement := (point_B.1 - point_A.1, point_B.2 - point_A.2)
  (force.1 * displacement.1 + force.2 * displacement.2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_work_done_equals_21_l3747_374778


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l3747_374760

/-- The rate per kg for grapes that Andrew purchased -/
def grape_rate : ℝ := 74

/-- The amount of grapes Andrew purchased in kg -/
def grape_amount : ℝ := 6

/-- The rate per kg for mangoes that Andrew purchased -/
def mango_rate : ℝ := 59

/-- The amount of mangoes Andrew purchased in kg -/
def mango_amount : ℝ := 9

/-- The total amount Andrew paid to the shopkeeper -/
def total_paid : ℝ := 975

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l3747_374760


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l3747_374705

/-- Represents a parabola in the form y = (x + a)² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (right : ℝ) (down : ℝ) : Parabola :=
  { a := p.a - right,
    b := p.b - down }

theorem parabola_shift_theorem (p : Parabola) :
  shift_parabola { a := 2, b := 3 } 3 2 = { a := -1, b := 1 } :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l3747_374705


namespace NUMINAMATH_CALUDE_semicircle_radius_in_specific_triangle_l3747_374706

/-- An isosceles triangle with a semicircle inscribed on its base -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base is positive -/
  base_pos : 0 < base
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The diameter of the semicircle is equal to the base of the triangle -/
  diameter_eq_base : 2 * radius = base
  /-- The radius plus the height of the triangle equals the length of the equal sides -/
  radius_plus_height_eq_side : radius + height = Real.sqrt ((base / 2) ^ 2 + height ^ 2)

/-- The radius of the inscribed semicircle in the specific isosceles triangle -/
theorem semicircle_radius_in_specific_triangle :
  ∃ (t : IsoscelesTriangleWithSemicircle), t.base = 20 ∧ t.height = 12 ∧ t.radius = 12 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_in_specific_triangle_l3747_374706


namespace NUMINAMATH_CALUDE_total_outlets_is_42_l3747_374771

/-- The number of rooms in the house -/
def num_rooms : ℕ := 7

/-- The number of outlets required per room -/
def outlets_per_room : ℕ := 6

/-- The total number of outlets needed for the house -/
def total_outlets : ℕ := num_rooms * outlets_per_room

/-- Theorem stating that the total number of outlets needed is 42 -/
theorem total_outlets_is_42 : total_outlets = 42 := by
  sorry

end NUMINAMATH_CALUDE_total_outlets_is_42_l3747_374771


namespace NUMINAMATH_CALUDE_min_p_for_quadratic_roots_in_unit_interval_l3747_374731

theorem min_p_for_quadratic_roots_in_unit_interval :
  (∃ (p : ℕ+),
    (∀ p' : ℕ+, p' < p →
      ¬∃ (q r : ℕ+),
        (∃ (x y : ℝ),
          0 < x ∧ x < 1 ∧
          0 < y ∧ y < 1 ∧
          x ≠ y ∧
          p' * x^2 - q * x + r = 0 ∧
          p' * y^2 - q * y + r = 0)) ∧
    (∃ (q r : ℕ+),
      ∃ (x y : ℝ),
        0 < x ∧ x < 1 ∧
        0 < y ∧ y < 1 ∧
        x ≠ y ∧
        p * x^2 - q * x + r = 0 ∧
        p * y^2 - q * y + r = 0)) ∧
  (∀ p : ℕ+,
    (∀ p' : ℕ+, p' < p →
      ¬∃ (q r : ℕ+),
        (∃ (x y : ℝ),
          0 < x ∧ x < 1 ∧
          0 < y ∧ y < 1 ∧
          x ≠ y ∧
          p' * x^2 - q * x + r = 0 ∧
          p' * y^2 - q * y + r = 0)) ∧
    (∃ (q r : ℕ+),
      ∃ (x y : ℝ),
        0 < x ∧ x < 1 ∧
        0 < y ∧ y < 1 ∧
        x ≠ y ∧
        p * x^2 - q * x + r = 0 ∧
        p * y^2 - q * y + r = 0) →
    p = 5) :=
by sorry

end NUMINAMATH_CALUDE_min_p_for_quadratic_roots_in_unit_interval_l3747_374731


namespace NUMINAMATH_CALUDE_larger_number_proof_l3747_374704

theorem larger_number_proof (S L : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) : 
  L = 1635 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3747_374704


namespace NUMINAMATH_CALUDE_helen_made_56_pies_l3747_374711

/-- The number of pies Helen made -/
def helen_pies (pinky_pies total_pies : ℕ) : ℕ := total_pies - pinky_pies

/-- Proof that Helen made 56 pies -/
theorem helen_made_56_pies : helen_pies 147 203 = 56 := by
  sorry

end NUMINAMATH_CALUDE_helen_made_56_pies_l3747_374711


namespace NUMINAMATH_CALUDE_shelter_final_count_l3747_374783

/-- Represents the number of cats in the shelter at different points in time. -/
structure CatCount where
  initial : Nat
  afterDoubling : Nat
  afterMonday : Nat
  afterTuesday : Nat
  afterWednesday : Nat
  afterThursday : Nat
  afterFriday : Nat
  afterReclaiming : Nat
  final : Nat

/-- Represents the events that occurred during the week at the animal shelter. -/
def shelterWeek (c : CatCount) : Prop :=
  c.afterDoubling = c.initial * 2 ∧
  c.afterDoubling = 48 ∧
  c.afterMonday = c.afterDoubling - 3 ∧
  c.afterTuesday = c.afterMonday + 5 ∧
  c.afterWednesday = c.afterTuesday - 3 ∧
  c.afterThursday = c.afterWednesday + 5 ∧
  c.afterFriday = c.afterThursday - 3 ∧
  c.afterReclaiming = c.afterFriday - 3 ∧
  c.final = c.afterReclaiming - 5

/-- Theorem stating that after the events of the week, the shelter has 41 cats. -/
theorem shelter_final_count (c : CatCount) :
  shelterWeek c → c.final = 41 := by
  sorry


end NUMINAMATH_CALUDE_shelter_final_count_l3747_374783


namespace NUMINAMATH_CALUDE_strawberry_harvest_l3747_374735

theorem strawberry_harvest (garden_length : ℝ) (garden_width : ℝ) 
  (plantable_percentage : ℝ) (plants_per_sqft : ℝ) (strawberries_per_plant : ℝ) : ℝ :=
  by
  have garden_length_eq : garden_length = 10 := by sorry
  have garden_width_eq : garden_width = 12 := by sorry
  have plantable_percentage_eq : plantable_percentage = 0.9 := by sorry
  have plants_per_sqft_eq : plants_per_sqft = 4 := by sorry
  have strawberries_per_plant_eq : strawberries_per_plant = 8 := by sorry
  
  have total_area : ℝ := garden_length * garden_width
  have plantable_area : ℝ := total_area * plantable_percentage
  have total_plants : ℝ := plantable_area * plants_per_sqft
  have total_strawberries : ℝ := total_plants * strawberries_per_plant
  
  exact total_strawberries

end NUMINAMATH_CALUDE_strawberry_harvest_l3747_374735


namespace NUMINAMATH_CALUDE_smallest_x_equals_f_2003_l3747_374746

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The problem statement -/
theorem smallest_x_equals_f_2003 :
  (∀ x > 0, f (3 * x) = 3 * f x) →
  (∀ x ∈ Set.Icc 2 4, f x = 1 - |x - 2|) →
  (∃ x₀ > 0, f x₀ = f 2003 ∧ ∀ x > 0, f x = f 2003 → x ≥ x₀) →
  (∃ x₀ > 0, f x₀ = f 2003 ∧ ∀ x > 0, f x = f 2003 → x ≥ x₀ ∧ x₀ = 1422817) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_equals_f_2003_l3747_374746


namespace NUMINAMATH_CALUDE_boat_speed_l3747_374710

theorem boat_speed (t : ℝ) (h : t > 0) : 
  let v_s : ℝ := 21
  let upstream_time : ℝ := 2 * t
  let downstream_time : ℝ := t
  let v_b : ℝ := (v_s * (upstream_time + downstream_time)) / (upstream_time - downstream_time)
  v_b = 63 := by sorry

end NUMINAMATH_CALUDE_boat_speed_l3747_374710


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_reciprocal_l3747_374745

theorem imaginary_part_of_complex_reciprocal (z : ℂ) (h : z = -2 + I) :
  (1 / z).im = -1 / 5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_reciprocal_l3747_374745


namespace NUMINAMATH_CALUDE_bond_value_after_eight_years_l3747_374742

/-- Represents the simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- The interest rate as a decimal -/
def interest_rate : ℝ := 0.08333333333333332

theorem bond_value_after_eight_years :
  ∀ initial_investment : ℝ,
  simple_interest initial_investment interest_rate 3 = 300 →
  simple_interest initial_investment interest_rate 8 = 400 :=
by
  sorry


end NUMINAMATH_CALUDE_bond_value_after_eight_years_l3747_374742


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l3747_374743

/-- Given a bowl of marbles with the following properties:
  - Total number of marbles is 19
  - Marbles are split into yellow, blue, and red
  - Ratio of blue to red marbles is 3:4
  - There are 3 more red marbles than yellow marbles
  Prove that the number of yellow marbles is 5. -/
theorem yellow_marbles_count :
  ∀ (yellow blue red : ℕ),
  yellow + blue + red = 19 →
  4 * blue = 3 * red →
  red = yellow + 3 →
  yellow = 5 := by
sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l3747_374743


namespace NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l3747_374726

/-- Given an arithmetic sequence with first term 2 and common difference 4,
    the 75th term of this sequence is 298. -/
theorem arithmetic_sequence_75th_term : 
  let a₁ : ℕ := 2  -- First term
  let d : ℕ := 4   -- Common difference
  let n : ℕ := 75  -- Term number we're looking for
  a₁ + (n - 1) * d = 298 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l3747_374726


namespace NUMINAMATH_CALUDE_not_multiple_of_five_l3747_374734

theorem not_multiple_of_five : ∃ n : ℕ, (2015^2 / 5^2 = n) ∧ ¬(∃ k : ℕ, n = 5 * k) ∧
  (∃ k₁ : ℕ, 2019^2 - 2014^2 = 5 * k₁) ∧
  (∃ k₂ : ℕ, 2019^2 * 10^2 = 5 * k₂) ∧
  (∃ k₃ : ℕ, 2020^2 / 101^2 = 5 * k₃) ∧
  (∃ k₄ : ℕ, 2010^2 - 2005^2 = 5 * k₄) :=
by sorry

#check not_multiple_of_five

end NUMINAMATH_CALUDE_not_multiple_of_five_l3747_374734


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3747_374772

theorem rectangular_solid_surface_area (a b c : ℕ) : 
  Prime a → Prime b → Prime c → 
  a * b * c = 273 → 
  2 * (a * b + b * c + c * a) = 302 := by sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3747_374772


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3747_374712

theorem exponent_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3747_374712


namespace NUMINAMATH_CALUDE_dish_price_proof_l3747_374717

theorem dish_price_proof (discount_rate : Real) (tip_rate : Real) (price_difference : Real) :
  let original_price : Real := 36
  let john_payment := original_price * (1 - discount_rate) + original_price * tip_rate
  let jane_payment := original_price * (1 - discount_rate) * (1 + tip_rate)
  discount_rate = 0.1 ∧ tip_rate = 0.15 ∧ price_difference = 0.54 →
  john_payment - jane_payment = price_difference :=
by
  sorry

end NUMINAMATH_CALUDE_dish_price_proof_l3747_374717


namespace NUMINAMATH_CALUDE_tan_angle_equality_l3747_374727

theorem tan_angle_equality (n : Int) :
  -90 < n ∧ n < 90 →
  Real.tan (n * Real.pi / 180) = Real.tan (75 * Real.pi / 180) →
  n = 75 := by
sorry

end NUMINAMATH_CALUDE_tan_angle_equality_l3747_374727


namespace NUMINAMATH_CALUDE_bond_energy_OF_bond_energy_OF_proof_l3747_374740

-- Define the molecules and atoms
inductive Molecule | OF₂ | O₂ | F₂
inductive Atom | O | F

-- Define the enthalpy of formation for OF₂
def enthalpy_formation_OF₂ : ℝ := 22

-- Define the bond energies for O₂ and F₂
def bond_energy_O₂ : ℝ := 498
def bond_energy_F₂ : ℝ := 159

-- Define the thermochemical equations
def thermochem_OF₂ (x : ℝ) : Prop :=
  x = 1 * bond_energy_F₂ + 0.5 * bond_energy_O₂ - enthalpy_formation_OF₂

-- Theorem: The bond energy of O-F in OF₂ is 215 kJ/mol
theorem bond_energy_OF : ℝ :=
  215

-- Proof of the theorem
theorem bond_energy_OF_proof : 
  thermochem_OF₂ bond_energy_OF := by
  sorry


end NUMINAMATH_CALUDE_bond_energy_OF_bond_energy_OF_proof_l3747_374740


namespace NUMINAMATH_CALUDE_shared_circles_existence_l3747_374797

-- Define the structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the structure for a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to check if a point is on a circle
def isPointOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define a function to check if a circle is the circumcircle of a triangle
def isCircumcircle (c : Circle) (t : Triangle) : Prop :=
  isPointOnCircle t.A c ∧ isPointOnCircle t.B c ∧ isPointOnCircle t.C c

-- Define a function to check if a circle is the inscribed circle of a triangle
def isInscribedCircle (c : Circle) (t : Triangle) : Prop :=
  -- This is a simplified condition; in reality, it would involve more complex geometric relationships
  true

-- The main theorem
theorem shared_circles_existence 
  (ABC : Triangle) 
  (O : Circle) 
  (I : Circle) 
  (h1 : isCircumcircle O ABC) 
  (h2 : isInscribedCircle I ABC) 
  (D : ℝ × ℝ) 
  (h3 : isPointOnCircle D O) : 
  ∃ (DEF : Triangle), isCircumcircle O DEF ∧ isInscribedCircle I DEF :=
sorry

end NUMINAMATH_CALUDE_shared_circles_existence_l3747_374797


namespace NUMINAMATH_CALUDE_sin_of_arcsin_plus_arctan_l3747_374784

theorem sin_of_arcsin_plus_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan 1) = 7 * Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_of_arcsin_plus_arctan_l3747_374784


namespace NUMINAMATH_CALUDE_february_highest_percentage_l3747_374766

-- Define the months
inductive Month
| January
| February
| March
| April
| May

-- Define the sales data for each month
def sales_data (m : Month) : (Nat × Nat × Nat) :=
  match m with
  | Month.January => (5, 4, 6)
  | Month.February => (6, 5, 7)
  | Month.March => (5, 5, 8)
  | Month.April => (4, 6, 7)
  | Month.May => (3, 4, 5)

-- Calculate the percentage difference
def percentage_difference (m : Month) : Rat :=
  let (d, b, f) := sales_data m
  let c := d + b
  (c - f : Rat) / f * 100

-- Theorem statement
theorem february_highest_percentage :
  ∀ m : Month, m ≠ Month.February →
  percentage_difference Month.February ≥ percentage_difference m :=
by sorry

end NUMINAMATH_CALUDE_february_highest_percentage_l3747_374766


namespace NUMINAMATH_CALUDE_pebble_difference_l3747_374768

/-- Given Shawn's pebble collection and painting process, prove the difference between blue and yellow pebbles. -/
theorem pebble_difference (total : ℕ) (red : ℕ) (blue : ℕ) (groups : ℕ) : 
  total = 40 →
  red = 9 →
  blue = 13 →
  groups = 3 →
  let remaining := total - red - blue
  let yellow := remaining / groups
  blue - yellow = 7 := by
  sorry

end NUMINAMATH_CALUDE_pebble_difference_l3747_374768


namespace NUMINAMATH_CALUDE_exactly_one_hit_probability_l3747_374730

def probability_exactly_one_hit (p_a p_b : ℝ) : ℝ :=
  p_a * (1 - p_b) + (1 - p_a) * p_b

theorem exactly_one_hit_probability :
  let p_a : ℝ := 1/2
  let p_b : ℝ := 1/3
  probability_exactly_one_hit p_a p_b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_hit_probability_l3747_374730


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l3747_374722

theorem smallest_solution_congruence :
  ∃! x : ℕ+, (5 * x.val ≡ 14 [ZMOD 26]) ∧
    ∀ y : ℕ+, (5 * y.val ≡ 14 [ZMOD 26]) → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l3747_374722


namespace NUMINAMATH_CALUDE_boat_against_stream_distance_l3747_374769

/-- Proves the distance traveled against the stream given boat speed and along-stream distance --/
theorem boat_against_stream_distance
  (boat_speed : ℝ)
  (along_stream_distance : ℝ)
  (h1 : boat_speed = 8)
  (h2 : along_stream_distance = 11)
  : (2 * boat_speed - along_stream_distance) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boat_against_stream_distance_l3747_374769


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l3747_374759

theorem integer_roots_of_cubic (x : ℤ) :
  x^3 - 4*x^2 - 11*x + 24 = 0 ↔ x = -4 ∨ x = 3 ∨ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l3747_374759


namespace NUMINAMATH_CALUDE_mixture_ratio_after_mixing_l3747_374757

/-- Represents a mixture of two liquids -/
structure Mixture where
  total : ℚ
  ratio_alpha : ℕ
  ratio_beta : ℕ

/-- Calculates the amount of alpha in a mixture -/
def alpha_amount (m : Mixture) : ℚ :=
  m.total * (m.ratio_alpha : ℚ) / ((m.ratio_alpha + m.ratio_beta) : ℚ)

/-- Calculates the amount of beta in a mixture -/
def beta_amount (m : Mixture) : ℚ :=
  m.total * (m.ratio_beta : ℚ) / ((m.ratio_alpha + m.ratio_beta) : ℚ)

theorem mixture_ratio_after_mixing (m1 m2 : Mixture)
  (h1 : m1.total = 6 ∧ m1.ratio_alpha = 7 ∧ m1.ratio_beta = 2)
  (h2 : m2.total = 9 ∧ m2.ratio_alpha = 4 ∧ m2.ratio_beta = 7) :
  (alpha_amount m1 + alpha_amount m2) / (beta_amount m1 + beta_amount m2) = 262 / 233 := by
  sorry

#eval 262 / 233

end NUMINAMATH_CALUDE_mixture_ratio_after_mixing_l3747_374757


namespace NUMINAMATH_CALUDE_square_sum_and_product_l3747_374799

theorem square_sum_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 4) 
  (h2 : (x + y)^2 = 64) : 
  x^2 + y^2 = 34 ∧ x * y = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_and_product_l3747_374799


namespace NUMINAMATH_CALUDE_min_sum_squares_l3747_374723

def S : Finset Int := {-8, -6, -4, -1, 3, 5, 7, 14}

theorem min_sum_squares (p q r s t u v w : Int) 
  (hp : p ∈ S) (hq : q ∈ S) (hr : r ∈ S) (hs : s ∈ S)
  (ht : t ∈ S) (hu : u ∈ S) (hv : v ∈ S) (hw : w ∈ S)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
               q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
               r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
               s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
               t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
               u ≠ v ∧ u ≠ w ∧
               v ≠ w) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 98 ∧ 
  ∃ (p' q' r' s' t' u' v' w' : Int),
    p' ∈ S ∧ q' ∈ S ∧ r' ∈ S ∧ s' ∈ S ∧ t' ∈ S ∧ u' ∈ S ∧ v' ∈ S ∧ w' ∈ S ∧
    p' ≠ q' ∧ p' ≠ r' ∧ p' ≠ s' ∧ p' ≠ t' ∧ p' ≠ u' ∧ p' ≠ v' ∧ p' ≠ w' ∧
    q' ≠ r' ∧ q' ≠ s' ∧ q' ≠ t' ∧ q' ≠ u' ∧ q' ≠ v' ∧ q' ≠ w' ∧
    r' ≠ s' ∧ r' ≠ t' ∧ r' ≠ u' ∧ r' ≠ v' ∧ r' ≠ w' ∧
    s' ≠ t' ∧ s' ≠ u' ∧ s' ≠ v' ∧ s' ≠ w' ∧
    t' ≠ u' ∧ t' ≠ v' ∧ t' ≠ w' ∧
    u' ≠ v' ∧ u' ≠ w' ∧
    v' ≠ w' ∧
    (p' + q' + r' + s')^2 + (t' + u' + v' + w')^2 = 98 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3747_374723


namespace NUMINAMATH_CALUDE_strict_manager_proposal_l3747_374744

/-- Represents the total monthly salary before changes -/
def initial_total_salary : ℕ := 10000

/-- Represents the total monthly salary after the kind manager's proposal -/
def kind_manager_total_salary : ℕ := 24000

/-- Represents the salary threshold -/
def salary_threshold : ℕ := 500

/-- Represents the number of employees -/
def total_employees : ℕ := 14

theorem strict_manager_proposal (x y : ℕ) 
  (h1 : x + y = total_employees)
  (h2 : 500 * x + y * salary_threshold ≤ initial_total_salary)
  (h3 : 3 * 500 * x + (initial_total_salary - 500 * x) + 1000 * y = kind_manager_total_salary) :
  500 * (x + y) = 7000 := by
  sorry

end NUMINAMATH_CALUDE_strict_manager_proposal_l3747_374744


namespace NUMINAMATH_CALUDE_coffee_decaf_percentage_l3747_374728

theorem coffee_decaf_percentage 
  (initial_stock : ℝ) 
  (initial_decaf_percent : ℝ) 
  (additional_purchase : ℝ) 
  (final_decaf_percent : ℝ) :
  initial_stock = 400 →
  initial_decaf_percent = 20 →
  additional_purchase = 100 →
  final_decaf_percent = 26 →
  let total_stock := initial_stock + additional_purchase
  let initial_decaf := initial_stock * (initial_decaf_percent / 100)
  let final_decaf := total_stock * (final_decaf_percent / 100)
  let additional_decaf := final_decaf - initial_decaf
  (additional_decaf / additional_purchase) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_coffee_decaf_percentage_l3747_374728


namespace NUMINAMATH_CALUDE_product_complex_polar_form_l3747_374789

/-- The product of two complex numbers in polar form results in another complex number in polar form -/
theorem product_complex_polar_form 
  (z₁ : ℂ) (z₂ : ℂ) (r₁ θ₁ r₂ θ₂ : ℝ) :
  z₁ = r₁ * Complex.exp (θ₁ * Complex.I) →
  z₂ = r₂ * Complex.exp (θ₂ * Complex.I) →
  r₁ = 4 →
  r₂ = 5 →
  θ₁ = 45 * π / 180 →
  θ₂ = 72 * π / 180 →
  ∃ (r θ : ℝ), 
    z₁ * z₂ = r * Complex.exp (θ * Complex.I) ∧
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * π ∧
    r = 20 ∧
    θ = 297 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_product_complex_polar_form_l3747_374789


namespace NUMINAMATH_CALUDE_ferry_crossings_parity_ferry_crossings_opposite_ferry_after_99_crossings_l3747_374795

/-- Represents the two banks of the river --/
inductive Bank : Type
| Left : Bank
| Right : Bank

/-- Returns the opposite bank --/
def opposite_bank (b : Bank) : Bank :=
  match b with
  | Bank.Left => Bank.Right
  | Bank.Right => Bank.Left

/-- Represents the state of the ferry after a number of crossings --/
def ferry_position (start : Bank) (crossings : Nat) : Bank :=
  if crossings % 2 = 0 then start else opposite_bank start

theorem ferry_crossings_parity (start : Bank) (crossings : Nat) :
  ferry_position start crossings = start ↔ crossings % 2 = 0 :=
sorry

theorem ferry_crossings_opposite (start : Bank) (crossings : Nat) :
  ferry_position start crossings = opposite_bank start ↔ crossings % 2 = 1 :=
sorry

theorem ferry_after_99_crossings (start : Bank) :
  ferry_position start 99 = opposite_bank start :=
sorry

end NUMINAMATH_CALUDE_ferry_crossings_parity_ferry_crossings_opposite_ferry_after_99_crossings_l3747_374795


namespace NUMINAMATH_CALUDE_gcd_12569_36975_l3747_374736

theorem gcd_12569_36975 : Nat.gcd 12569 36975 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12569_36975_l3747_374736


namespace NUMINAMATH_CALUDE_cube_sum_value_l3747_374715

theorem cube_sum_value (a b R S : ℝ) : 
  a + b = R → 
  a^2 + b^2 = 12 → 
  a^3 + b^3 = S → 
  S = 32 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_value_l3747_374715


namespace NUMINAMATH_CALUDE_volume_of_S_l3747_374747

-- Define the solid S' in the first octant
def S' : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
                   x + 2*y ≤ 1 ∧ 2*x + z ≤ 1 ∧ y + 2*z ≤ 1}

-- State the theorem about the volume of S'
theorem volume_of_S' : MeasureTheory.volume S' = 1/48 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_S_l3747_374747


namespace NUMINAMATH_CALUDE_probability_12_draws_10_red_l3747_374749

/-- The number of yellow balls in the bag -/
def yellow_balls : ℕ := 5

/-- The number of red balls in the bag -/
def red_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := yellow_balls + red_balls

/-- The probability of drawing a red ball -/
def p_red : ℚ := red_balls / total_balls

/-- The probability of drawing a yellow ball -/
def p_yellow : ℚ := yellow_balls / total_balls

/-- The number of red balls needed to stop the process -/
def red_balls_needed : ℕ := 10

/-- The number of draws when the process stops -/
def total_draws : ℕ := 12

/-- The probability of drawing exactly 12 balls to get 10 red balls -/
theorem probability_12_draws_10_red (ξ : ℕ → ℚ) : 
  ξ total_draws = (Nat.choose (total_draws - 1) (red_balls_needed - 1)) * 
                  (p_red ^ red_balls_needed) * 
                  (p_yellow ^ (total_draws - red_balls_needed)) := by
  sorry

end NUMINAMATH_CALUDE_probability_12_draws_10_red_l3747_374749


namespace NUMINAMATH_CALUDE_six_people_arrangement_l3747_374777

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of positions where person A can stand (not at ends) -/
def validPositionsForA (n : ℕ) : ℕ := n - 2

/-- The number of ways to arrange the remaining people after placing A -/
def remainingArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The total number of valid arrangements -/
def validArrangements (n : ℕ) : ℕ :=
  (validPositionsForA n) * (remainingArrangements n)

theorem six_people_arrangement :
  validArrangements 6 = 480 := by
  sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l3747_374777


namespace NUMINAMATH_CALUDE_fundraising_amount_proof_l3747_374796

/-- Calculates the amount each person needs to raise in a fundraising event -/
def amount_per_person (total_goal : ℕ) (initial_donation : ℕ) (num_participants : ℕ) : ℕ :=
  (total_goal - initial_donation) / num_participants

/-- Proves that given the specified conditions, each person needs to raise $225 -/
theorem fundraising_amount_proof (total_goal : ℕ) (initial_donation : ℕ) (num_participants : ℕ) 
  (h1 : total_goal = 2000)
  (h2 : initial_donation = 200)
  (h3 : num_participants = 8) :
  amount_per_person total_goal initial_donation num_participants = 225 := by
  sorry

#eval amount_per_person 2000 200 8

end NUMINAMATH_CALUDE_fundraising_amount_proof_l3747_374796


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3747_374794

theorem complex_magnitude_problem (z : ℂ) (h : (z + 1) / (z - 2) = 1 - 3*I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3747_374794


namespace NUMINAMATH_CALUDE_positive_interval_l3747_374750

theorem positive_interval (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_interval_l3747_374750


namespace NUMINAMATH_CALUDE_coronavirus_cases_in_new_york_l3747_374753

theorem coronavirus_cases_in_new_york :
  ∀ (new_york california texas : ℕ),
    california = new_york / 2 →
    california = texas + 400 →
    new_york + california + texas = 3600 →
    new_york = 800 := by
  sorry

end NUMINAMATH_CALUDE_coronavirus_cases_in_new_york_l3747_374753


namespace NUMINAMATH_CALUDE_inequality_proof_l3747_374719

theorem inequality_proof (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 6)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧ 
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3747_374719


namespace NUMINAMATH_CALUDE_boxes_with_neither_l3747_374767

/-- Represents the set of boxes in Christine's storage room. -/
def Boxes : Type := Unit

/-- The total number of boxes. -/
def total_boxes : ℕ := 15

/-- The number of boxes containing markers. -/
def boxes_with_markers : ℕ := 8

/-- The number of boxes containing sharpies. -/
def boxes_with_sharpies : ℕ := 5

/-- The number of boxes containing both markers and sharpies. -/
def boxes_with_both : ℕ := 4

/-- Theorem stating the number of boxes containing neither markers nor sharpies. -/
theorem boxes_with_neither (b : Boxes) :
  total_boxes - (boxes_with_markers + boxes_with_sharpies - boxes_with_both) = 6 := by
  sorry


end NUMINAMATH_CALUDE_boxes_with_neither_l3747_374767


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l3747_374782

theorem fraction_subtraction_simplification :
  (8 : ℚ) / 29 - (5 : ℚ) / 87 = (19 : ℚ) / 87 ∧ 
  (∀ n d : ℤ, n ≠ 0 → (19 : ℚ) / 87 = (n : ℚ) / d → (abs n = 19 ∧ abs d = 87)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l3747_374782


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3747_374701

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

-- Define the asymptote of the hyperbola
def asymptote (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b / a) * x ∨ y = -(b / a) * x

-- Define the axis of the parabola
def parabola_axis (p : ℝ) (x : ℝ) : Prop :=
  x = -p / 2

-- Theorem statement
theorem hyperbola_equation (a b p : ℝ) :
  a > 0 ∧ b > 0 ∧ p > 0 ∧
  (∃ x₀ y₀, asymptote a b x₀ y₀ ∧ parabola_axis p x₀ ∧ x₀ = -2 ∧ y₀ = -4) ∧
  (∃ x₁ y₁ x₂ y₂, hyperbola a b x₁ y₁ ∧ x₁ = -a ∧ y₁ = 0 ∧
                  parabola p x₂ y₂ ∧ x₂ = p ∧ y₂ = 0 ∧
                  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 16) →
  a = 2 ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3747_374701


namespace NUMINAMATH_CALUDE_start_time_is_6am_l3747_374781

/-- Represents the hiking scenario with two hikers --/
structure HikingScenario where
  meetTime : ℝ       -- Time when hikers meet (in hours after midnight)
  rychlyEndTime : ℝ  -- Time when Mr. Rychlý finishes (in hours after midnight)
  loudaEndTime : ℝ   -- Time when Mr. Louda finishes (in hours after midnight)

/-- Calculates the start time of the hike given a HikingScenario --/
def calculateStartTime (scenario : HikingScenario) : ℝ :=
  scenario.meetTime - (scenario.rychlyEndTime - scenario.meetTime)

/-- Theorem stating that the start time is 6 AM (6 hours after midnight) --/
theorem start_time_is_6am (scenario : HikingScenario) 
  (h1 : scenario.meetTime = 10)
  (h2 : scenario.rychlyEndTime = 12)
  (h3 : scenario.loudaEndTime = 18) :
  calculateStartTime scenario = 6 := by
  sorry

#eval calculateStartTime { meetTime := 10, rychlyEndTime := 12, loudaEndTime := 18 }

end NUMINAMATH_CALUDE_start_time_is_6am_l3747_374781


namespace NUMINAMATH_CALUDE_james_printing_problem_l3747_374741

/-- Calculates the minimum number of sheets required for printing books -/
def sheets_required (num_books : ℕ) (pages_per_book : ℕ) (sides_per_sheet : ℕ) (pages_per_side : ℕ) : ℕ :=
  let total_pages := num_books * pages_per_book
  let pages_per_sheet := sides_per_sheet * pages_per_side
  (total_pages + pages_per_sheet - 1) / pages_per_sheet

theorem james_printing_problem :
  sheets_required 5 800 3 6 = 223 := by
  sorry

end NUMINAMATH_CALUDE_james_printing_problem_l3747_374741


namespace NUMINAMATH_CALUDE_f_inequality_l3747_374786

/-- The function f(x) = x^2 - 2x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + c

/-- Theorem stating that f(0) < f(4) < f(-4) for any real c -/
theorem f_inequality (c : ℝ) : f c 0 < f c 4 ∧ f c 4 < f c (-4) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l3747_374786


namespace NUMINAMATH_CALUDE_squared_binomial_subtraction_difference_of_squares_l3747_374709

-- Problem 1
theorem squared_binomial_subtraction (a b : ℝ) :
  a^2 * b - (-2 * a * b^2)^2 = a^2 * b - 4 * a^2 * b^4 := by sorry

-- Problem 2
theorem difference_of_squares (x y : ℝ) :
  (3 * x - 2 * y) * (3 * x + 2 * y) = 9 * x^2 - 4 * y^2 := by sorry

end NUMINAMATH_CALUDE_squared_binomial_subtraction_difference_of_squares_l3747_374709


namespace NUMINAMATH_CALUDE_curve_properties_l3747_374779

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the parametric curve C -/
def C (a : ℝ) (t : ℝ) : Point where
  x := 1 + 2 * t
  y := a * t^2

/-- The point M lies on the curve C -/
def M : Point := ⟨5, 4⟩

theorem curve_properties (a : ℝ) :
  (∃ t, C a t = M) →
  (a = 1 ∧ ∀ x y, (C 1 ((x - 1) / 2)).y = y ↔ 4 * y = (x - 1)^2) := by
  sorry


end NUMINAMATH_CALUDE_curve_properties_l3747_374779


namespace NUMINAMATH_CALUDE_opposite_numbers_subtraction_not_always_smaller_l3747_374737

-- Statement 1
theorem opposite_numbers (a b : ℝ) : a + b = 0 → a = -b := by sorry

-- Statement 2
theorem subtraction_not_always_smaller : ∃ x y : ℚ, x - y > y := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_subtraction_not_always_smaller_l3747_374737


namespace NUMINAMATH_CALUDE_race_distance_l3747_374774

theorem race_distance (d : ℝ) (a b c : ℝ) : 
  (d > 0) →
  (d / a = (d - 30) / b) →
  (d / b = (d - 15) / c) →
  (d / a = (d - 40) / c) →
  d = 90 := by
sorry

end NUMINAMATH_CALUDE_race_distance_l3747_374774


namespace NUMINAMATH_CALUDE_full_price_revenue_l3747_374791

/-- Represents the concert ticket sales problem. -/
structure ConcertTickets where
  total_tickets : ℕ
  total_revenue : ℕ
  full_price : ℕ
  discounted_price : ℕ
  full_price_tickets : ℕ
  discounted_tickets : ℕ

/-- The revenue generated from full-price tickets is $4500. -/
theorem full_price_revenue (ct : ConcertTickets)
  (h1 : ct.total_tickets = 200)
  (h2 : ct.total_revenue = 4500)
  (h3 : ct.discounted_price = ct.full_price / 3)
  (h4 : ct.total_tickets = ct.full_price_tickets + ct.discounted_tickets)
  (h5 : ct.total_revenue = ct.full_price * ct.full_price_tickets + ct.discounted_price * ct.discounted_tickets) :
  ct.full_price * ct.full_price_tickets = 4500 := by
  sorry

end NUMINAMATH_CALUDE_full_price_revenue_l3747_374791


namespace NUMINAMATH_CALUDE_zero_in_interval_l3747_374790

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x - 9

-- State the theorem
theorem zero_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3747_374790


namespace NUMINAMATH_CALUDE_victor_lost_lives_l3747_374732

theorem victor_lost_lives (current_lives : ℕ) (difference : ℕ) (lost_lives : ℕ) : 
  current_lives = 2 → difference = 12 → lost_lives - current_lives = difference → lost_lives = 14 := by
  sorry

end NUMINAMATH_CALUDE_victor_lost_lives_l3747_374732


namespace NUMINAMATH_CALUDE_function_properties_quadratic_inequality_solution_set_maximum_value_of_fraction_l3747_374751

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

theorem function_properties :
  ∀ a b : ℝ,
  (∀ x ∈ Set.Ioo (-3) 2, f a b x > 0) ∧
  (∀ x ∈ Set.Iic (-3) ∪ Set.Ici 2, f a b x < 0) →
  ∀ x, f a b x = -3 * x^2 - 3 * x + 15 :=
sorry

theorem quadratic_inequality_solution_set :
  ∀ a b c : ℝ,
  (∀ x : ℝ, a * x^2 + b * x + c ≤ 0) →
  c ≤ -25/12 :=
sorry

theorem maximum_value_of_fraction :
  ∀ x : ℝ,
  x > -1 →
  (f (-3) 5 x - 21) / (x + 1) ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_function_properties_quadratic_inequality_solution_set_maximum_value_of_fraction_l3747_374751


namespace NUMINAMATH_CALUDE_profit_ratio_from_investment_l3747_374703

/-- The profit ratio of two partners given their investment ratio and investment durations -/
theorem profit_ratio_from_investment 
  (p_investment q_investment : ℕ) 
  (p_duration q_duration : ℚ) 
  (h_investment_ratio : p_investment * 5 = q_investment * 7)
  (h_p_duration : p_duration = 5)
  (h_q_duration : q_duration = 11) :
  p_investment * p_duration * 11 = q_investment * q_duration * 7 :=
by sorry

end NUMINAMATH_CALUDE_profit_ratio_from_investment_l3747_374703


namespace NUMINAMATH_CALUDE_average_weight_b_c_l3747_374739

/-- Given three weights a, b, and c, prove that the average weight of b and c is 50 kg
    under the specified conditions. -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 60 →  -- The average weight of a, b, and c is 60 kg
  (a + b) / 2 = 70 →      -- The average weight of a and b is 70 kg
  b = 60 →                -- The weight of b is 60 kg
  (b + c) / 2 = 50 :=     -- The average weight of b and c is 50 kg
by sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l3747_374739


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l3747_374716

def f (x a : ℝ) : ℝ := (x + 1) * (x + a)

theorem even_function_implies_a_eq_neg_one (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l3747_374716


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_constant_term_binomial_expansion_proof_l3747_374788

/-- The constant term in the binomial expansion of (2x - 1/√x)^6 is 60 -/
theorem constant_term_binomial_expansion : ℕ :=
  let n : ℕ := 6
  let a : ℝ → ℝ := λ x ↦ 2 * x
  let b : ℝ → ℝ := λ x ↦ -1 / Real.sqrt x
  let expansion : ℝ → ℝ := λ x ↦ (a x + b x) ^ n
  let constant_term : ℕ := 60
  constant_term

/-- Proof of the theorem -/
theorem constant_term_binomial_expansion_proof : 
  constant_term_binomial_expansion = 60 := by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_constant_term_binomial_expansion_proof_l3747_374788


namespace NUMINAMATH_CALUDE_square_sum_equals_one_l3747_374780

theorem square_sum_equals_one (a b : ℝ) 
  (h : a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) = 1) : 
  a^2 + b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_one_l3747_374780


namespace NUMINAMATH_CALUDE_sheets_count_l3747_374752

/-- The number of sheets in a set of writing materials -/
def S : ℕ := sorry

/-- The number of envelopes in a set of writing materials -/
def E : ℕ := sorry

/-- John's equation: sheets minus envelopes equals 80 -/
axiom john_equation : S - E = 80

/-- Mary's equation: sheets equals 4 times envelopes -/
axiom mary_equation : S = 4 * E

/-- Theorem: The number of sheets in each set is 320 -/
theorem sheets_count : S = 320 := by sorry

end NUMINAMATH_CALUDE_sheets_count_l3747_374752


namespace NUMINAMATH_CALUDE_probability_three_heads_out_of_five_probability_three_heads_proof_l3747_374770

/-- The probability of three specific coins out of five coming up heads when all five are flipped simultaneously -/
theorem probability_three_heads_out_of_five : ℚ :=
  1 / 8

/-- The total number of possible outcomes when flipping five coins -/
def total_outcomes : ℕ := 2^5

/-- The number of successful outcomes where three specific coins are heads -/
def successful_outcomes : ℕ := 2^2

theorem probability_three_heads_proof :
  (successful_outcomes : ℚ) / total_outcomes = probability_three_heads_out_of_five :=
sorry

end NUMINAMATH_CALUDE_probability_three_heads_out_of_five_probability_three_heads_proof_l3747_374770


namespace NUMINAMATH_CALUDE_sequence_a1_value_l3747_374707

theorem sequence_a1_value (p q : ℝ) (a : ℕ → ℝ) 
  (hp : p > 0) (hq : q > 0)
  (ha_pos : ∀ n, a n > 0)
  (ha_0 : a 0 = 1)
  (ha_rec : ∀ n, a (n + 2) = p * a n - q * a (n + 1)) :
  a 1 = (-q + Real.sqrt (q^2 + 4*p)) / 2 :=
sorry

end NUMINAMATH_CALUDE_sequence_a1_value_l3747_374707


namespace NUMINAMATH_CALUDE_expression_evaluation_l3747_374758

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  (x^3 + 1) / x * (y^3 + 1) / y + (x^3 - 1) / y * (y^3 - 1) / x = 2 * x^2 * y^2 + 2 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3747_374758


namespace NUMINAMATH_CALUDE_sum_of_roots_f_2y_eq_10_l3747_374798

/-- The function f as defined in the problem -/
def f (x : ℝ) : ℝ := (3*x)^2 + 3*x + 1

/-- The theorem stating the sum of roots of f(2y) = 10 -/
theorem sum_of_roots_f_2y_eq_10 :
  ∃ y₁ y₂ : ℝ, f (2*y₁) = 10 ∧ f (2*y₂) = 10 ∧ y₁ + y₂ = -0.17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_f_2y_eq_10_l3747_374798


namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l3747_374713

/-- The quadratic function y = (k-1)x^2 + 2x - 1 intersects the x-axis if and only if k ≥ 0 and k ≠ 1 -/
theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 2 * x - 1 = 0) ↔ (k ≥ 0 ∧ k ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l3747_374713


namespace NUMINAMATH_CALUDE_absolute_value_problem_l3747_374718

theorem absolute_value_problem (m n : ℤ) 
  (hm : |m| = 4) (hn : |n| = 3) : 
  ((m * n > 0 → |m - n| = 1) ∧ 
   (m * n < 0 → |m + n| = 1)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l3747_374718


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3747_374765

/-- A 30-60-90 triangle with shortest side length 2 -/
structure Triangle30_60_90 where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  is_30_60_90 : True  -- Placeholder for the triangle's angle properties
  de_length : dist D E = 2

/-- A circle tangent to coordinate axes and parts of the triangle -/
structure TangentCircle where
  O : ℝ × ℝ  -- Center of the circle
  r : ℝ      -- Radius of the circle
  triangle : Triangle30_60_90
  tangent_to_axes : True  -- Placeholder for tangency to coordinate axes
  tangent_to_leg : True   -- Placeholder for tangency to one leg of the triangle
  tangent_to_hypotenuse : True  -- Placeholder for tangency to hypotenuse

/-- The main theorem stating the radius of the tangent circle -/
theorem tangent_circle_radius (c : TangentCircle) : c.r = (5 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3747_374765


namespace NUMINAMATH_CALUDE_shopping_money_l3747_374762

theorem shopping_money (initial_amount : ℝ) (remaining_amount : ℝ) : 
  remaining_amount = 140 →
  remaining_amount = initial_amount * (1 - 0.3) →
  initial_amount = 200 := by
sorry

end NUMINAMATH_CALUDE_shopping_money_l3747_374762


namespace NUMINAMATH_CALUDE_minimize_f_l3747_374721

/-- The function f(x) = x^2 + 8x + 3 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 3

/-- Theorem: The value of x that minimizes f(x) = x^2 + 8x + 3 is -4 -/
theorem minimize_f :
  ∃ (x_min : ℝ), x_min = -4 ∧ ∀ (x : ℝ), f x ≥ f x_min :=
by sorry

end NUMINAMATH_CALUDE_minimize_f_l3747_374721


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l3747_374720

/-- Represents a parabola y² = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Represents a point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

theorem parabola_focus_distance (C : Parabola) (A : PointOnParabola C)
  (h_focus_dist : Real.sqrt ((A.x - C.p / 2)^2 + A.y^2) = 12)
  (h_y_axis_dist : A.x = 9) :
  C.p = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l3747_374720


namespace NUMINAMATH_CALUDE_difference_of_squares_l3747_374763

theorem difference_of_squares (x : ℝ) : (x + 3) * (x - 3) = x^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3747_374763


namespace NUMINAMATH_CALUDE_percentage_calculation_l3747_374792

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 6000 → 
  P / 100 * (30 / 100) * (50 / 100) * N = 90 → 
  P = 10 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3747_374792


namespace NUMINAMATH_CALUDE_square_cut_impossible_l3747_374776

/-- Proves that a square with perimeter 40 cannot be cut into two identical rectangles with perimeter 20 each -/
theorem square_cut_impossible (square_perimeter : ℝ) (rect_perimeter : ℝ) : 
  square_perimeter = 40 → rect_perimeter = 20 → 
  ¬ ∃ (square_side rect_length rect_width : ℝ),
    (square_side * 4 = square_perimeter) ∧ 
    (rect_length + rect_width = square_side) ∧
    (2 * (rect_length + rect_width) = rect_perimeter) :=
by
  sorry

#check square_cut_impossible

end NUMINAMATH_CALUDE_square_cut_impossible_l3747_374776


namespace NUMINAMATH_CALUDE_square_perimeter_when_area_equals_diagonal_l3747_374708

theorem square_perimeter_when_area_equals_diagonal : 
  ∀ s : ℝ, s > 0 → 
  s^2 = s * Real.sqrt 2 → 
  4 * s = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_when_area_equals_diagonal_l3747_374708


namespace NUMINAMATH_CALUDE_min_c_value_l3747_374775

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

theorem min_c_value (a b c d e : ℕ) : 
  (a + 1 = b) → 
  (b + 1 = c) → 
  (c + 1 = d) → 
  (d + 1 = e) → 
  (is_perfect_square (b + c + d)) →
  (is_perfect_cube (a + b + c + d + e)) →
  (c ≥ 675 ∧ ∀ x < c, ¬(is_perfect_square (x - 1 + x + x + 1) ∧ 
                        is_perfect_cube (x - 2 + x - 1 + x + x + 1 + x + 2))) :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l3747_374775


namespace NUMINAMATH_CALUDE_michaels_matchsticks_l3747_374700

/-- The number of matchsticks Michael had originally -/
def original_matchsticks : ℕ := 1700

/-- The number of houses Michael created -/
def houses : ℕ := 30

/-- The number of towers Michael created -/
def towers : ℕ := 20

/-- The number of bridges Michael created -/
def bridges : ℕ := 10

/-- The number of matchsticks used for each house -/
def matchsticks_per_house : ℕ := 10

/-- The number of matchsticks used for each tower -/
def matchsticks_per_tower : ℕ := 15

/-- The number of matchsticks used for each bridge -/
def matchsticks_per_bridge : ℕ := 25

/-- Theorem stating that Michael's original pile of matchsticks was 1700 -/
theorem michaels_matchsticks :
  original_matchsticks = 2 * (houses * matchsticks_per_house +
                              towers * matchsticks_per_tower +
                              bridges * matchsticks_per_bridge) :=
by sorry

end NUMINAMATH_CALUDE_michaels_matchsticks_l3747_374700


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3747_374702

theorem arithmetic_sequence_ratio (a d : ℝ) : 
  (a + d) + (a + 3*d) = 6*a ∧ 
  a + 2*d = 10 →
  a / (a + 3*d) = 1/4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3747_374702


namespace NUMINAMATH_CALUDE_dvds_per_season_l3747_374714

theorem dvds_per_season (total_dvds : ℕ) (num_seasons : ℕ) 
  (h1 : total_dvds = 40) (h2 : num_seasons = 5) : 
  total_dvds / num_seasons = 8 := by
  sorry

end NUMINAMATH_CALUDE_dvds_per_season_l3747_374714


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l3747_374773

/-- Given that x is inversely proportional to y, this theorem proves that
    if x₁/x₂ = 3/5, then y₁/y₂ = 5/3, where y₁ and y₂ are the corresponding
    y values for x₁ and x₂. -/
theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) :
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y, x * y = k) →  -- x is inversely proportional to y
  x₁ / x₂ = 3 / 5 →
  y₁ / y₂ = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l3747_374773


namespace NUMINAMATH_CALUDE_kristin_green_beans_count_l3747_374725

/-- Represents the number of vegetables a person has -/
structure VegetableCount where
  carrots : ℕ
  cucumbers : ℕ
  bellPeppers : ℕ
  greenBeans : ℕ

/-- The problem statement -/
theorem kristin_green_beans_count 
  (jaylen : VegetableCount)
  (kristin : VegetableCount)
  (h1 : jaylen.carrots = 5)
  (h2 : jaylen.cucumbers = 2)
  (h3 : jaylen.bellPeppers = 2 * kristin.bellPeppers)
  (h4 : jaylen.greenBeans = kristin.greenBeans / 2 - 3)
  (h5 : jaylen.carrots + jaylen.cucumbers + jaylen.bellPeppers + jaylen.greenBeans = 18)
  (h6 : kristin.bellPeppers = 2) :
  kristin.greenBeans = 20 := by
  sorry

end NUMINAMATH_CALUDE_kristin_green_beans_count_l3747_374725


namespace NUMINAMATH_CALUDE_min_value_constraint_l3747_374755

theorem min_value_constraint (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + 2*y = 1) :
  2*x + 3*y^2 ≥ 0.75 := by
  sorry

end NUMINAMATH_CALUDE_min_value_constraint_l3747_374755


namespace NUMINAMATH_CALUDE_convex_polygon_symmetry_l3747_374733

/-- A polygon is convex if all its interior angles are less than or equal to 180 degrees. -/
def ConvexPolygon (P : Set (ℝ × ℝ)) : Prop := sorry

/-- A polygon has a center of symmetry if there exists a point such that every point of the polygon has a corresponding point that is equidistant from the center but in the opposite direction. -/
def HasCenterOfSymmetry (P : Set (ℝ × ℝ)) : Prop := sorry

/-- A polygon can be divided into smaller polygons if there exists a partition of the polygon into a finite number of non-overlapping smaller polygons. -/
def CanBeDivided (P : Set (ℝ × ℝ)) (subPolygons : Finset (Set (ℝ × ℝ))) : Prop := sorry

theorem convex_polygon_symmetry 
  (P : Set (ℝ × ℝ)) 
  (subPolygons : Finset (Set (ℝ × ℝ))) 
  (h1 : ConvexPolygon P) 
  (h2 : CanBeDivided P subPolygons) 
  (h3 : ∀ Q ∈ subPolygons, HasCenterOfSymmetry Q) : 
  HasCenterOfSymmetry P := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_symmetry_l3747_374733


namespace NUMINAMATH_CALUDE_somu_father_age_ratio_l3747_374754

/-- Represents the ages of Somu and his father -/
structure Ages where
  somu : ℕ
  father : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.somu = 14 ∧
  ages.somu - 7 = (ages.father - 7) / 5

/-- The theorem to prove -/
theorem somu_father_age_ratio (ages : Ages) :
  problem_conditions ages →
  (ages.somu : ℚ) / ages.father = 1 / 3 := by
  sorry

#check somu_father_age_ratio

end NUMINAMATH_CALUDE_somu_father_age_ratio_l3747_374754

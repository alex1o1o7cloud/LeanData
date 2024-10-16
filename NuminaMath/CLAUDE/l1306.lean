import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_s_l1306_130677

theorem min_value_of_s (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 3 * x^2 + 2 * y^2 + z^2 = 1) :
  (1 + z) / (x * y * z) ≥ 8 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_s_l1306_130677


namespace NUMINAMATH_CALUDE_friends_earrings_count_l1306_130619

def total_earrings (bella_earrings monica_earrings rachel_earrings olivia_earrings : ℕ) : ℕ :=
  bella_earrings + monica_earrings + rachel_earrings + olivia_earrings

theorem friends_earrings_count :
  ∀ (bella_earrings monica_earrings rachel_earrings olivia_earrings : ℕ),
    bella_earrings = 10 →
    bella_earrings = monica_earrings / 4 →
    monica_earrings = 2 * rachel_earrings →
    olivia_earrings = bella_earrings + monica_earrings + rachel_earrings + 5 →
    total_earrings bella_earrings monica_earrings rachel_earrings olivia_earrings = 145 :=
by
  sorry

#check friends_earrings_count

end NUMINAMATH_CALUDE_friends_earrings_count_l1306_130619


namespace NUMINAMATH_CALUDE_water_pouring_proof_l1306_130651

/-- Represents the fraction of water remaining after n pourings -/
def remainingWater (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

theorem water_pouring_proof :
  remainingWater 28 = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_water_pouring_proof_l1306_130651


namespace NUMINAMATH_CALUDE_abs_is_even_and_increasing_l1306_130613

def f (x : ℝ) := |x|

theorem abs_is_even_and_increasing :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_abs_is_even_and_increasing_l1306_130613


namespace NUMINAMATH_CALUDE_point_c_coordinates_l1306_130669

/-- Given two points A and B in ℝ², if vector BC is half of vector BA, 
    then the coordinates of point C are (0, 3/2) -/
theorem point_c_coordinates 
  (A B : ℝ × ℝ)
  (h_A : A = (1, 1))
  (h_B : B = (-1, 2))
  (h_BC : ∃ (C : ℝ × ℝ), C - B = (1/2) • (A - B)) :
  ∃ (C : ℝ × ℝ), C = (0, 3/2) := by
sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l1306_130669


namespace NUMINAMATH_CALUDE_ken_cycling_distance_l1306_130614

/-- Ken's cycling speed in miles per hour when it's raining -/
def rain_speed : ℝ := 30 * 3

/-- Ken's cycling speed in miles per hour when it's snowing -/
def snow_speed : ℝ := 10 * 3

/-- Number of rainy days in a week -/
def rainy_days : ℕ := 3

/-- Number of snowy days in a week -/
def snowy_days : ℕ := 4

/-- Hours Ken cycles per day -/
def hours_per_day : ℝ := 1

theorem ken_cycling_distance :
  rain_speed * rainy_days * hours_per_day + snow_speed * snowy_days * hours_per_day = 390 := by
  sorry

end NUMINAMATH_CALUDE_ken_cycling_distance_l1306_130614


namespace NUMINAMATH_CALUDE_min_value_of_E_l1306_130629

theorem min_value_of_E (x : ℝ) :
  let f (E : ℝ) := |x - 4| + |E| + |x - 5|
  (∃ (E : ℝ), f E = 10 ∧ ∀ (E' : ℝ), f E' ≥ 10) →
  (∃ (E_min : ℝ), |E_min| = 9 ∧ ∀ (E : ℝ), |E| ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_E_l1306_130629


namespace NUMINAMATH_CALUDE_log_27_3_l1306_130690

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_27_3_l1306_130690


namespace NUMINAMATH_CALUDE_root_product_l1306_130605

theorem root_product (x₁ x₂ : ℝ) (h₁ : x₁ * Real.log x₁ = 2006) (h₂ : x₂ * Real.exp x₂ = 2006) : 
  x₁ * x₂ = 2006 := by
sorry

end NUMINAMATH_CALUDE_root_product_l1306_130605


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1306_130661

theorem real_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  (2 * i / (1 + i)).re = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l1306_130661


namespace NUMINAMATH_CALUDE_power_inequality_l1306_130601

theorem power_inequality (a b : ℝ) (ha : a > 0) (ha1 : a ≠ 1) (hab : a^b > 1) : a * b > b := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l1306_130601


namespace NUMINAMATH_CALUDE_range_of_m_l1306_130680

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x ∈ Set.Icc (-1) m ∧ m > -1 ∧ |x| - 1 > 0) → 
  m ∈ Set.Ioo (-1) 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1306_130680


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1306_130604

theorem arithmetic_computation : 2 + 5 * 3 - 4 + 8 * 2 / 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1306_130604


namespace NUMINAMATH_CALUDE_pony_discount_rate_l1306_130607

-- Define the regular prices
def fox_price : ℝ := 15
def pony_price : ℝ := 18

-- Define the number of jeans purchased
def fox_quantity : ℕ := 3
def pony_quantity : ℕ := 2

-- Define the total savings
def total_savings : ℝ := 9

-- Define the sum of discount rates
def total_discount_rate : ℝ := 22

-- Theorem statement
theorem pony_discount_rate :
  ∃ (fox_discount pony_discount : ℝ),
    fox_discount + pony_discount = total_discount_rate ∧
    fox_quantity * fox_price * (fox_discount / 100) +
    pony_quantity * pony_price * (pony_discount / 100) = total_savings ∧
    pony_discount = 10 := by
  sorry

end NUMINAMATH_CALUDE_pony_discount_rate_l1306_130607


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l1306_130691

theorem smallest_four_digit_multiple_of_18 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 18 = 0 → n ≥ 1008 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l1306_130691


namespace NUMINAMATH_CALUDE_min_omega_for_shifted_sine_l1306_130668

/-- The minimum value of ω for a sine function with a specific shift -/
theorem min_omega_for_shifted_sine (ω : ℝ) (h_pos : ω > 0) :
  (∀ x, 3 * Real.sin (ω * x + π / 6) - 2 = 3 * Real.sin (ω * (x - 2 * π / 3) + π / 6) - 2) →
  ω ≥ 3 ∧ ∃ n : ℕ, ω = 3 * n := by
  sorry

end NUMINAMATH_CALUDE_min_omega_for_shifted_sine_l1306_130668


namespace NUMINAMATH_CALUDE_trapezoid_inner_quadrilateral_area_l1306_130640

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- A trapezoid is a quadrilateral with two parallel sides -/
structure Trapezoid extends Quadrilateral :=
  (parallel : (A.y - B.y) / (A.x - B.x) = (D.y - C.y) / (D.x - C.x))

/-- Calculate the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Check if a point lies on a line segment -/
def onSegment (P Q R : Point) : Prop := sorry

/-- Find the intersection point of two line segments -/
def intersectionPoint (P Q R S : Point) : Point := sorry

/-- Theorem: Area of inner quadrilateral is at most 1/4 of trapezoid area -/
theorem trapezoid_inner_quadrilateral_area 
  (ABCD : Trapezoid) 
  (E : Point) 
  (F : Point) 
  (H : Point) 
  (G : Point)
  (hE : onSegment ABCD.A ABCD.B E)
  (hF : onSegment ABCD.C ABCD.D F)
  (hH : H = intersectionPoint ABCD.C E ABCD.B F)
  (hG : G = intersectionPoint E ABCD.D ABCD.A F) :
  area ⟨E, H, F, G⟩ ≤ (1/4 : ℝ) * area ABCD.toQuadrilateral := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_inner_quadrilateral_area_l1306_130640


namespace NUMINAMATH_CALUDE_triangle_area_l1306_130650

/-- The area of a triangle with base 9 cm and height 12 cm is 54 cm². -/
theorem triangle_area : 
  ∀ (base height area : ℝ), 
  base = 9 → height = 12 → area = (1/2) * base * height → 
  area = 54 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1306_130650


namespace NUMINAMATH_CALUDE_patrick_savings_ratio_l1306_130675

theorem patrick_savings_ratio :
  ∀ (bicycle_cost initial_savings current_savings lent_amount : ℕ),
    bicycle_cost = 150 →
    lent_amount = 50 →
    current_savings = 25 →
    initial_savings = current_savings + lent_amount →
    (initial_savings : ℚ) / bicycle_cost = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_patrick_savings_ratio_l1306_130675


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l1306_130612

theorem floor_equation_solutions : 
  (∃ (S : Finset ℕ), S.card = 9 ∧ 
    (∀ x : ℕ, x ∈ S ↔ ⌊(x : ℚ) / 5⌋ = ⌊(x : ℚ) / 7⌋)) :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l1306_130612


namespace NUMINAMATH_CALUDE_nina_travel_distance_l1306_130610

theorem nina_travel_distance (x : ℕ) : 
  (12 * x + 12 * (2 * x) = 14400) → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_nina_travel_distance_l1306_130610


namespace NUMINAMATH_CALUDE_train_speed_problem_l1306_130606

/-- Proves that given the conditions of the train problem, the speeds of the regular and high-speed trains are 100 km/h and 250 km/h respectively. -/
theorem train_speed_problem (regular_speed : ℝ) (bullet_speed : ℝ) (high_speed : ℝ) (express_speed : ℝ)
  (h1 : bullet_speed = 2 * regular_speed)
  (h2 : high_speed = bullet_speed * 1.25)
  (h3 : (high_speed + regular_speed) / 2 = express_speed + 15)
  (h4 : (bullet_speed + regular_speed) / 2 = express_speed - 10) :
  regular_speed = 100 ∧ high_speed = 250 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1306_130606


namespace NUMINAMATH_CALUDE_circle_circumference_after_scaling_l1306_130689

theorem circle_circumference_after_scaling (a b : ℝ) (h1 : a = 7) (h2 : b = 24) : 
  let d := Real.sqrt (a^2 + b^2)
  let new_d := 1.5 * d
  let new_circumference := π * new_d
  new_circumference = 37.5 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_after_scaling_l1306_130689


namespace NUMINAMATH_CALUDE_existence_of_equal_point_l1306_130674

theorem existence_of_equal_point
  (f g : ℝ → ℝ)
  (hf : Continuous f)
  (hg : Continuous g)
  (hg_diff : Differentiable ℝ g)
  (h_condition : (f 0 - deriv g 0) * (deriv g 1 - f 1) > 0) :
  ∃ c ∈ (Set.Ioo 0 1), f c = deriv g c :=
sorry

end NUMINAMATH_CALUDE_existence_of_equal_point_l1306_130674


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l1306_130666

/-- Represents the charge function for a caterer -/
structure Caterer where
  base_fee : ℕ
  per_person_fee : ℕ

/-- Calculates the total charge for a given number of people -/
def total_charge (c : Caterer) (people : ℕ) : ℕ :=
  c.base_fee + c.per_person_fee * people

/-- The first caterer's pricing structure -/
def caterer1 : Caterer :=
  { base_fee := 100, per_person_fee := 15 }

/-- The second caterer's pricing structure -/
def caterer2 : Caterer :=
  { base_fee := 200, per_person_fee := 12 }

/-- Theorem stating that 34 is the least number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper_at_34 :
  (∀ n : ℕ, n < 34 → total_charge caterer1 n ≤ total_charge caterer2 n) ∧
  (total_charge caterer2 34 < total_charge caterer1 34) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l1306_130666


namespace NUMINAMATH_CALUDE_probability_at_least_one_boy_one_girl_l1306_130631

theorem probability_at_least_one_boy_one_girl :
  let p_boy : ℝ := 1 / 2
  let p_girl : ℝ := 1 - p_boy
  let num_children : ℕ := 4
  let p_all_boys : ℝ := p_boy ^ num_children
  let p_all_girls : ℝ := p_girl ^ num_children
  p_all_boys + p_all_girls = 1 / 8 →
  1 - (p_all_boys + p_all_girls) = 7 / 8 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_boy_one_girl_l1306_130631


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1306_130699

theorem inscribed_circle_radius 
  (R : ℝ) 
  (r : ℝ) 
  (h1 : R = 18) 
  (h2 : r = 9) 
  (h3 : r = R / 2) : 
  ∃ x : ℝ, x = 8 ∧ 
    (R - x)^2 - x^2 = (r + x)^2 - x^2 ∧ 
    x > 0 ∧ 
    x < R ∧ 
    x < r := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1306_130699


namespace NUMINAMATH_CALUDE_expression_evaluation_l1306_130637

theorem expression_evaluation :
  let x : ℚ := -1/3
  (2*x - 1)^2 - (3*x + 1)*(3*x - 1) + 5*x*(x - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1306_130637


namespace NUMINAMATH_CALUDE_quiz_competition_order_l1306_130617

theorem quiz_competition_order (A B C : ℕ) 
  (h1 : A + B = 2 * C)
  (h2 : 3 * A > 3 * B + 3 * C + 10)
  (h3 : 3 * B = 3 * C + 5)
  (h4 : A > 0 ∧ B > 0 ∧ C > 0) :
  C > A ∧ A > B := by
  sorry

end NUMINAMATH_CALUDE_quiz_competition_order_l1306_130617


namespace NUMINAMATH_CALUDE_analogous_property_is_about_surfaces_l1306_130625

/-- Represents a geometric property in plane geometry -/
structure PlaneProperty where
  description : String

/-- Represents a geometric property in solid geometry -/
structure SolidProperty where
  description : String

/-- Represents the analogy between plane and solid geometry properties -/
def analogy (plane : PlaneProperty) : SolidProperty :=
  sorry

/-- The plane geometry property about equilateral triangles -/
def triangle_property : PlaneProperty :=
  { description := "The sum of distances from any point inside an equilateral triangle to its three sides is constant" }

/-- Theorem stating that the analogous property in solid geometry is about surfaces -/
theorem analogous_property_is_about_surfaces :
  ∃ (surface_prop : SolidProperty),
    (analogy triangle_property).description = "A property about surfaces" :=
  sorry

end NUMINAMATH_CALUDE_analogous_property_is_about_surfaces_l1306_130625


namespace NUMINAMATH_CALUDE_eve_distance_difference_l1306_130667

def running_distances : List ℚ := [3/4, 2/3, 19/20, 3/4, 7/8]
def walking_distances : List ℚ := [1/2, 13/20, 5/6, 3/5, 4/5, 3/4]

theorem eve_distance_difference :
  (running_distances.sum - walking_distances.sum : ℚ) = -1416/10000 := by sorry

end NUMINAMATH_CALUDE_eve_distance_difference_l1306_130667


namespace NUMINAMATH_CALUDE_third_term_of_specific_arithmetic_sequence_l1306_130622

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- State the theorem
theorem third_term_of_specific_arithmetic_sequence 
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 2) :
  a 3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_third_term_of_specific_arithmetic_sequence_l1306_130622


namespace NUMINAMATH_CALUDE_total_animals_in_community_l1306_130623

theorem total_animals_in_community (total_families : ℕ) 
  (families_with_two_dogs : ℕ) (families_with_one_dog : ℕ) 
  (h1 : total_families = 50)
  (h2 : families_with_two_dogs = 15)
  (h3 : families_with_one_dog = 20) :
  (families_with_two_dogs * 2 + families_with_one_dog * 1 + 
   (total_families - families_with_two_dogs - families_with_one_dog) * 2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_in_community_l1306_130623


namespace NUMINAMATH_CALUDE_cos_phase_shift_l1306_130678

/-- The phase shift of y = cos(2x + π/2) is -π/4 --/
theorem cos_phase_shift : 
  let f := fun x => Real.cos (2 * x + π / 2)
  let phase_shift := fun (B C : ℝ) => -C / B
  phase_shift 2 (π / 2) = -π / 4 := by
sorry

end NUMINAMATH_CALUDE_cos_phase_shift_l1306_130678


namespace NUMINAMATH_CALUDE_tan_floor_eq_two_cos_sq_iff_pi_quarter_plus_two_pi_k_l1306_130663

/-- The floor function, which returns the greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Theorem stating that the equation [tan x] = 2 cos^2 x is satisfied if and only if x = π/4 + 2kπ, where k is an integer -/
theorem tan_floor_eq_two_cos_sq_iff_pi_quarter_plus_two_pi_k (x : ℝ) :
  floor (Real.tan x) = (2 : ℝ) * (Real.cos x)^2 ↔ ∃ k : ℤ, x = π/4 + 2*k*π :=
sorry

end NUMINAMATH_CALUDE_tan_floor_eq_two_cos_sq_iff_pi_quarter_plus_two_pi_k_l1306_130663


namespace NUMINAMATH_CALUDE_lemonade_stand_problem_l1306_130603

/-- Represents the lemonade stand problem --/
theorem lemonade_stand_problem 
  (glasses_per_gallon : ℕ)
  (gallons_made : ℕ)
  (price_per_glass : ℚ)
  (glasses_drunk : ℕ)
  (glasses_unsold : ℕ)
  (net_profit : ℚ)
  (h1 : glasses_per_gallon = 16)
  (h2 : gallons_made = 2)
  (h3 : price_per_glass = 1)
  (h4 : glasses_drunk = 5)
  (h5 : glasses_unsold = 6)
  (h6 : net_profit = 14) :
  (gallons_made * glasses_per_gallon - glasses_drunk - glasses_unsold) * price_per_glass - net_profit = gallons_made * (7/2 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_lemonade_stand_problem_l1306_130603


namespace NUMINAMATH_CALUDE_real_number_line_bijection_l1306_130670

-- Define the number line as a type
def NumberLine : Type := ℝ

-- Define the bijection between real numbers and points on the number line
def realToPoint : ℝ → NumberLine := id

-- Statement: There is a one-to-one correspondence between real numbers and points on the number line
theorem real_number_line_bijection : Function.Bijective realToPoint := by
  sorry

end NUMINAMATH_CALUDE_real_number_line_bijection_l1306_130670


namespace NUMINAMATH_CALUDE_lucky_draw_probabilities_l1306_130694

def probability_wang_wins : ℝ := 0.4
def probability_zhang_wins : ℝ := 0.2

theorem lucky_draw_probabilities :
  let p_both_win := probability_wang_wins * probability_zhang_wins
  let p_only_one_wins := probability_wang_wins * (1 - probability_zhang_wins) + (1 - probability_wang_wins) * probability_zhang_wins
  let p_at_most_one_wins := 1 - p_both_win
  (p_both_win = 0.08) ∧
  (p_only_one_wins = 0.44) ∧
  (p_at_most_one_wins = 0.92) := by
  sorry

end NUMINAMATH_CALUDE_lucky_draw_probabilities_l1306_130694


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1306_130615

-- Define the repeating decimals
def repeating_2 : ℚ := 2 / 9
def repeating_03 : ℚ := 3 / 99
def repeating_0004 : ℚ := 4 / 9999
def repeating_00005 : ℚ := 5 / 99999

-- State the theorem
theorem sum_of_repeating_decimals :
  repeating_2 + repeating_03 + repeating_0004 + repeating_00005 = 56534 / 99999 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1306_130615


namespace NUMINAMATH_CALUDE_solution_satisfies_congruences_solution_is_three_digit_solution_is_smallest_l1306_130683

/-- The smallest three-digit number satisfying the given congruences -/
def smallest_solution : ℕ := 230

/-- First congruence condition -/
def cong1 (x : ℕ) : Prop := 5 * x ≡ 15 [ZMOD 10]

/-- Second congruence condition -/
def cong2 (x : ℕ) : Prop := 3 * x + 4 ≡ 7 [ZMOD 8]

/-- Third congruence condition -/
def cong3 (x : ℕ) : Prop := -3 * x + 2 ≡ x [ZMOD 17]

/-- The solution satisfies all congruences -/
theorem solution_satisfies_congruences :
  cong1 smallest_solution ∧ 
  cong2 smallest_solution ∧ 
  cong3 smallest_solution :=
sorry

/-- The solution is a three-digit number -/
theorem solution_is_three_digit :
  100 ≤ smallest_solution ∧ smallest_solution < 1000 :=
sorry

/-- The solution is the smallest such number -/
theorem solution_is_smallest (n : ℕ) :
  (100 ≤ n ∧ n < smallest_solution) →
  ¬(cong1 n ∧ cong2 n ∧ cong3 n) :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_congruences_solution_is_three_digit_solution_is_smallest_l1306_130683


namespace NUMINAMATH_CALUDE_expression_evaluation_l1306_130624

theorem expression_evaluation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5 - 1/6) = 37 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1306_130624


namespace NUMINAMATH_CALUDE_square_greater_than_negative_double_l1306_130672

theorem square_greater_than_negative_double {a : ℝ} (h : a < -2) : a^2 > -2*a := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_negative_double_l1306_130672


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_1_l1306_130602

theorem min_value_of_3a_plus_1 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 2) :
  ∃ (min_val : ℝ), min_val = -5/4 ∧ ∀ (x : ℝ), 8 * x^2 + 6 * x + 5 = 2 → 3 * x + 1 ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_1_l1306_130602


namespace NUMINAMATH_CALUDE_max_value_of_z_l1306_130635

theorem max_value_of_z (x y : ℝ) (h1 : y ≥ x) (h2 : x + y ≤ 1) (h3 : y ≥ -1) :
  ∃ (z_max : ℝ), z_max = 1/2 ∧ ∀ z, z = 2*x - y → z ≤ z_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_z_l1306_130635


namespace NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l1306_130655

def certain_number : ℕ := 1152

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_x_for_perfect_cube :
  ∃! x : ℕ, x > 0 ∧ is_perfect_cube (certain_number * x) ∧
    ∀ y : ℕ, y > 0 ∧ y < x → ¬is_perfect_cube (certain_number * y) ∧
    certain_number * x = 12 * certain_number :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l1306_130655


namespace NUMINAMATH_CALUDE_total_peanuts_l1306_130608

/-- The number of peanuts initially in the box -/
def initial_peanuts : ℕ := 10

/-- The number of peanuts Mary adds to the box -/
def added_peanuts : ℕ := 8

/-- Theorem stating the total number of peanuts in the box -/
theorem total_peanuts : initial_peanuts + added_peanuts = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_peanuts_l1306_130608


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1306_130646

theorem equilateral_triangle_perimeter (area : ℝ) (p : ℝ) : 
  area = 50 * Real.sqrt 12 → p = 60 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1306_130646


namespace NUMINAMATH_CALUDE_ny_mets_fans_l1306_130684

/-- The number of NY Mets fans in a town with specific fan ratios --/
theorem ny_mets_fans (total : ℕ) (y m r d : ℚ) : 
  total = 780 →
  y / m = 3 / 2 →
  m / r = 4 / 5 →
  r / d = 7 / (3/2) →
  y + m + r + d = total →
  ⌊m⌋ = 178 := by
  sorry

end NUMINAMATH_CALUDE_ny_mets_fans_l1306_130684


namespace NUMINAMATH_CALUDE_mission_duration_percentage_l1306_130658

theorem mission_duration_percentage (P : ℝ) : 
  (5 + P / 100 * 5 + 3 = 11) → P = 60 := by
sorry

end NUMINAMATH_CALUDE_mission_duration_percentage_l1306_130658


namespace NUMINAMATH_CALUDE_tangent_line_equations_l1306_130698

/-- The equations of the lines passing through point (1,1) and tangent to the curve y = x³ + 1 -/
theorem tangent_line_equations : 
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b → (x = 1 ∧ y = 1)) ∧ 
    (∃ x₀ : ℝ, 
      (x₀^3 + 1 = m * x₀ + b) ∧ 
      (3 * x₀^2 = m)) ∧
    ((m = 0 ∧ b = 1) ∨ (m = 27/4 ∧ b = -23/4)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l1306_130698


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1306_130639

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1306_130639


namespace NUMINAMATH_CALUDE_petes_number_l1306_130676

theorem petes_number (x : ℝ) : 3 * (2 * x + 12) = 90 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l1306_130676


namespace NUMINAMATH_CALUDE_daily_apple_harvest_l1306_130659

/-- The number of sections in the apple orchard -/
def orchard_sections : ℕ := 8

/-- The number of sacks of apples harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks of apples harvested daily -/
def total_sacks : ℕ := orchard_sections * sacks_per_section

theorem daily_apple_harvest :
  total_sacks = 360 :=
by sorry

end NUMINAMATH_CALUDE_daily_apple_harvest_l1306_130659


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l1306_130679

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) : ∃ d : ℕ, d > 0 ∧ d ∣ n ∧ ∀ k : ℕ, k > 0 → k ∣ n → k ≤ d := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l1306_130679


namespace NUMINAMATH_CALUDE_combination_98_96_l1306_130685

theorem combination_98_96 : Nat.choose 98 96 = 4753 := by
  sorry

end NUMINAMATH_CALUDE_combination_98_96_l1306_130685


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1306_130618

/-- Given a quadratic function f(x) = 2x^2 + bx + c with solution set (0, 2) for f(x) < 0,
    if f(x) + t ≥ 2 holds for all real x, then t ≥ 4 -/
theorem quadratic_inequality_range (b c t : ℝ) : 
  (∀ x, x ∈ Set.Ioo 0 2 ↔ 2*x^2 + b*x + c < 0) →
  (∀ x, 2*x^2 + b*x + c + t ≥ 2) →
  t ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_range_l1306_130618


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1306_130656

theorem absolute_value_inequality_solution_set : 
  {x : ℝ | |x - 2| ≤ 1} = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1306_130656


namespace NUMINAMATH_CALUDE_log_8641_between_consecutive_integers_l1306_130641

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_8641_between_consecutive_integers : 
  ∃ (c d : ℤ), c + 1 = d ∧ 
  (log10 1000 : ℝ) = 3 ∧
  (log10 10000 : ℝ) = 4 ∧
  1000 < 8641 ∧ 8641 < 10000 ∧
  Monotone log10 ∧
  (c : ℝ) < log10 8641 ∧ log10 8641 < (d : ℝ) ∧
  c + d = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_8641_between_consecutive_integers_l1306_130641


namespace NUMINAMATH_CALUDE_twentieth_term_of_combined_sequence_l1306_130609

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

def geometric_sequence (g₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := g₁ * r ^ (n - 1)

def combined_sequence (a₁ g₁ d r : ℝ) (n : ℕ) : ℝ :=
  arithmetic_sequence a₁ d n + geometric_sequence g₁ r n

theorem twentieth_term_of_combined_sequence :
  combined_sequence 3 2 4 2 20 = 1048655 := by sorry

end NUMINAMATH_CALUDE_twentieth_term_of_combined_sequence_l1306_130609


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1306_130647

/-- Given vectors a and b in ℝ², prove that they are perpendicular if and only if x = -3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) : 
  a = (-1, 3) → b = (-3, x) → (a.1 * b.1 + a.2 * b.2 = 0 ↔ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1306_130647


namespace NUMINAMATH_CALUDE_vector_sum_scalar_multiple_l1306_130662

/-- Given vectors a and b in ℝ², prove that a + 2b equals the expected result. -/
theorem vector_sum_scalar_multiple (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (-2, 1)) :
  a + 2 • b = (-3, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_scalar_multiple_l1306_130662


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l1306_130616

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l1306_130616


namespace NUMINAMATH_CALUDE_train_speed_l1306_130620

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 1500) (h2 : time = 50) :
  length / time = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1306_130620


namespace NUMINAMATH_CALUDE_fourth_power_sum_l1306_130652

theorem fourth_power_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (square_sum_condition : a^2 + b^2 + c^2 = 3)
  (cube_sum_condition : a^3 + b^3 + c^3 = 4) :
  a^4 + b^4 + c^4 = 7.833 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l1306_130652


namespace NUMINAMATH_CALUDE_arccos_cos_ten_l1306_130653

open Real

-- Define the problem statement
theorem arccos_cos_ten :
  let x := 10
  let y := arccos (cos x)
  0 ≤ y ∧ y ≤ π →
  y = x - 2 * π :=
by sorry

end NUMINAMATH_CALUDE_arccos_cos_ten_l1306_130653


namespace NUMINAMATH_CALUDE_thousandth_spirit_enters_on_fourth_floor_l1306_130644

/-- Represents the number of house spirits that enter the elevator on each floor during a complete up-and-down trip -/
def spirits_per_cycle (num_floors : ℕ) : ℕ := 2 * (num_floors - 1) + 2

/-- Calculates the floor on which the nth house spirit enters the elevator -/
def floor_of_nth_spirit (n : ℕ) (num_floors : ℕ) : ℕ :=
  let complete_cycles := (n - 1) / spirits_per_cycle num_floors
  let remaining_spirits := (n - 1) % spirits_per_cycle num_floors
  if remaining_spirits < num_floors then
    remaining_spirits + 1
  else
    2 * num_floors - remaining_spirits - 1

theorem thousandth_spirit_enters_on_fourth_floor :
  floor_of_nth_spirit 1000 7 = 4 := by sorry

end NUMINAMATH_CALUDE_thousandth_spirit_enters_on_fourth_floor_l1306_130644


namespace NUMINAMATH_CALUDE_pet_store_cages_l1306_130688

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : 
  initial_puppies = 56 →
  sold_puppies = 24 →
  puppies_per_cage = 4 →
  (initial_puppies - sold_puppies) / puppies_per_cage = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l1306_130688


namespace NUMINAMATH_CALUDE_sugar_used_in_two_minutes_l1306_130638

/-- Calculates the total sugar used in chocolate production over a given time period. -/
def sugarUsed (sugarPerBar : ℝ) (barsPerMinute : ℕ) (minutes : ℕ) : ℝ :=
  sugarPerBar * (barsPerMinute : ℝ) * (minutes : ℝ)

/-- Theorem stating that given the specified production parameters, 
    the total sugar used in two minutes is 108 grams. -/
theorem sugar_used_in_two_minutes :
  sugarUsed 1.5 36 2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_sugar_used_in_two_minutes_l1306_130638


namespace NUMINAMATH_CALUDE_circle_line_intersection_range_l1306_130697

/-- Given a circle C and a line l, if they have a common point, 
    then the range of values for a is [-1/2, 1/2) -/
theorem circle_line_intersection_range (a : ℝ) : 
  let C := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*a*x + 2*a*y + 2*a^2 + 2*a - 1 = 0}
  let l := {(x, y) : ℝ × ℝ | x - y - 1 = 0}
  (∃ p, p ∈ C ∩ l) → a ∈ Set.Icc (-1/2) (1/2) := by
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_range_l1306_130697


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1306_130660

/-- Proves that the ratio of Saras's age to the combined age of Kul and Ani is 1:2 -/
theorem age_ratio_proof (kul_age saras_age ani_age : ℕ) 
  (h1 : kul_age = 22)
  (h2 : saras_age = 33)
  (h3 : ani_age = 44) : 
  (saras_age : ℚ) / (kul_age + ani_age : ℚ) = 1 / 2 := by
  sorry

#check age_ratio_proof

end NUMINAMATH_CALUDE_age_ratio_proof_l1306_130660


namespace NUMINAMATH_CALUDE_data_center_connections_l1306_130671

theorem data_center_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_data_center_connections_l1306_130671


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1306_130696

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 * a 1 - 10 * a 1 + 16 = 0) →
  (a 19 * a 19 - 10 * a 19 + 16 = 0) →
  a 8 * a 10 * a 12 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1306_130696


namespace NUMINAMATH_CALUDE_solution_set_for_a_zero_range_of_a_for_solution_exists_l1306_130643

-- Define the functions f and g
def f (x : ℝ) : ℝ := abs (x + 1)
def g (a : ℝ) (x : ℝ) : ℝ := 2 * abs x + a

-- Theorem for part (I)
theorem solution_set_for_a_zero :
  {x : ℝ | f x ≥ g 0 x} = Set.Icc (-1/3) 1 := by sorry

-- Theorem for part (II)
theorem range_of_a_for_solution_exists :
  {a : ℝ | ∃ x, f x ≥ g a x} = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_zero_range_of_a_for_solution_exists_l1306_130643


namespace NUMINAMATH_CALUDE_winner_votes_not_unique_l1306_130634

/-- Represents an election result --/
structure ElectionResult where
  totalVotes : ℕ
  winnerVotes : ℕ
  secondPlaceVotes : ℕ

/-- Conditions of the election --/
def electionConditions (result : ElectionResult) : Prop :=
  (result.winnerVotes : ℚ) / result.totalVotes = 58 / 100 ∧
  result.winnerVotes - result.secondPlaceVotes = 1200

/-- Theorem stating that the number of votes for the winning candidate cannot be uniquely determined --/
theorem winner_votes_not_unique :
  ∃ (result1 result2 : ElectionResult),
    result1 ≠ result2 ∧
    electionConditions result1 ∧
    electionConditions result2 :=
sorry

end NUMINAMATH_CALUDE_winner_votes_not_unique_l1306_130634


namespace NUMINAMATH_CALUDE_avg_percent_grades_5_6_midville_easton_l1306_130664

/-- Represents a school with its total number of students and percentages for each grade --/
structure School where
  total_students : ℕ
  grade_k_percent : ℚ
  grade_1_percent : ℚ
  grade_2_percent : ℚ
  grade_3_percent : ℚ
  grade_4_percent : ℚ
  grade_5_percent : ℚ
  grade_6_percent : ℚ

def midville : School := {
  total_students := 150,
  grade_k_percent := 18/100,
  grade_1_percent := 14/100,
  grade_2_percent := 15/100,
  grade_3_percent := 12/100,
  grade_4_percent := 16/100,
  grade_5_percent := 12/100,
  grade_6_percent := 13/100
}

def easton : School := {
  total_students := 250,
  grade_k_percent := 10/100,
  grade_1_percent := 14/100,
  grade_2_percent := 17/100,
  grade_3_percent := 18/100,
  grade_4_percent := 13/100,
  grade_5_percent := 15/100,
  grade_6_percent := 13/100
}

/-- Calculates the average percentage of students in grades 5 and 6 for two schools combined --/
def avg_percent_grades_5_6 (s1 s2 : School) : ℚ :=
  let total_students := s1.total_students + s2.total_students
  let students_5_6 := s1.total_students * (s1.grade_5_percent + s1.grade_6_percent) +
                      s2.total_students * (s2.grade_5_percent + s2.grade_6_percent)
  students_5_6 / total_students

theorem avg_percent_grades_5_6_midville_easton :
  avg_percent_grades_5_6 midville easton = 2725/10000 := by
  sorry

end NUMINAMATH_CALUDE_avg_percent_grades_5_6_midville_easton_l1306_130664


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1306_130626

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the condition c = 2a cos B
def condition (t : Triangle) : Prop :=
  t.c = 2 * t.a * Real.cos t.angleB

-- State the theorem
theorem isosceles_triangle (t : Triangle) (h : condition t) : t.a = t.b := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1306_130626


namespace NUMINAMATH_CALUDE_only_baseball_count_l1306_130686

/-- Represents the number of people in different categories in a class --/
structure ClassSports where
  total : ℕ
  both : ℕ
  onlyFootball : ℕ
  neither : ℕ

/-- Theorem stating the number of people who only like baseball --/
theorem only_baseball_count (c : ClassSports) 
  (h1 : c.total = 16)
  (h2 : c.both = 5)
  (h3 : c.onlyFootball = 3)
  (h4 : c.neither = 6) :
  c.total - (c.both + c.onlyFootball + c.neither) = 2 :=
sorry

end NUMINAMATH_CALUDE_only_baseball_count_l1306_130686


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1306_130682

theorem line_segment_endpoint (y : ℝ) : 
  y > 0 → 
  Real.sqrt ((7 - 2)^2 + (y - 4)^2) = 6 → 
  y = 4 + Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1306_130682


namespace NUMINAMATH_CALUDE_sum_of_digits_of_7_pow_25_l1306_130692

/-- The sum of the tens digit and the ones digit of 7^25 -/
def sum_of_digits : ℕ :=
  let n : ℕ := 7^25
  (n / 10 % 10) + (n % 10)

/-- Theorem stating that the sum of the tens digit and the ones digit of 7^25 is 7 -/
theorem sum_of_digits_of_7_pow_25 : sum_of_digits = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_7_pow_25_l1306_130692


namespace NUMINAMATH_CALUDE_tan_30_degrees_l1306_130649

theorem tan_30_degrees : Real.tan (30 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_degrees_l1306_130649


namespace NUMINAMATH_CALUDE_problem_statement_l1306_130665

theorem problem_statement : (1 / (64^(1/3))^9) * 8^6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1306_130665


namespace NUMINAMATH_CALUDE_nova_monthly_donation_l1306_130627

/-- Nova's monthly donation to charity -/
def monthly_donation : ℕ := 1707

/-- Nova's total annual donation to charity -/
def annual_donation : ℕ := 20484

/-- Number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: Nova's monthly donation is $1,707 -/
theorem nova_monthly_donation :
  monthly_donation = annual_donation / months_in_year :=
by sorry

end NUMINAMATH_CALUDE_nova_monthly_donation_l1306_130627


namespace NUMINAMATH_CALUDE_tangent_problems_l1306_130628

theorem tangent_problems (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + Real.pi/4) = -3) ∧
  (Real.sin (2*α) / (Real.sin α ^ 2 + Real.sin α * Real.cos α) = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_tangent_problems_l1306_130628


namespace NUMINAMATH_CALUDE_easter_egg_hunt_l1306_130695

theorem easter_egg_hunt (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_person : ℕ) 
  (shondas_kids : ℕ) (friends : ℕ) (shonda : ℕ) :
  baskets = 15 →
  eggs_per_basket = 12 →
  eggs_per_person = 9 →
  shondas_kids = 2 →
  friends = 10 →
  shonda = 1 →
  (baskets * eggs_per_basket) / eggs_per_person - (shondas_kids + friends + shonda) = 7 :=
by sorry

end NUMINAMATH_CALUDE_easter_egg_hunt_l1306_130695


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l1306_130673

/-- A parabola with equation y = x^2 - 8x + m has its vertex on the x-axis if and only if m = 16 -/
theorem parabola_vertex_on_x_axis (m : ℝ) :
  (∃ x, x^2 - 8*x + m = 0 ∧ ∀ y, y^2 - 8*y + m ≥ x^2 - 8*x + m) ↔ m = 16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l1306_130673


namespace NUMINAMATH_CALUDE_compound_interest_problem_l1306_130642

-- Define the compound interest function
def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n

-- State the theorem
theorem compound_interest_problem :
  ∃ (P r : ℝ), 
    compound_interest P r 2 = 8800 ∧
    compound_interest P r 3 = 9261 ∧
    abs (P - 7945.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l1306_130642


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1306_130654

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 5 ∧ x₂ = -1 ∧ 
  x₁^2 - 4*x₁ - 5 = 0 ∧ 
  x₂^2 - 4*x₂ - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1306_130654


namespace NUMINAMATH_CALUDE_cylinder_cross_section_area_l1306_130636

/-- The area of the cross-section of a cylinder intersected by a plane -/
theorem cylinder_cross_section_area 
  (r : ℝ) -- radius of the cylinder base
  (α : ℝ) -- angle between the intersecting plane and the base plane
  (h₁ : r > 0) -- radius is positive
  (h₂ : 0 < α ∧ α < π / 2) -- angle is between 0 and π/2 (exclusive)
  : ∃ (A : ℝ), A = π * r^2 / Real.cos α :=
sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_area_l1306_130636


namespace NUMINAMATH_CALUDE_two_co_presidents_probability_l1306_130687

def club_sizes : List Nat := [6, 8, 9, 10]
def co_presidents_per_club : Nat := 2

def probability_two_co_presidents (sizes : List Nat) (co_pres : Nat) : ℚ :=
  let probabilities := sizes.map (λ n =>
    (Nat.choose (n - co_pres) (co_pres)) / (Nat.choose n 4))
  (1 / 4 : ℚ) * (probabilities.sum)

theorem two_co_presidents_probability :
  probability_two_co_presidents club_sizes co_presidents_per_club = 2286/10000 := by
  sorry

end NUMINAMATH_CALUDE_two_co_presidents_probability_l1306_130687


namespace NUMINAMATH_CALUDE_solution_set_equivalence_range_of_a_l1306_130645

-- Define the function f
def f (a b x : ℝ) := x^2 - a*x + b

-- Part 1
theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 - a*x + 1 < 0 ↔ 1/3 < x ∧ x < 1/2) :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x, f a (2*a - 3) x ≥ 0) →
  2 ≤ a ∧ a ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_range_of_a_l1306_130645


namespace NUMINAMATH_CALUDE_interval_length_for_inequality_l1306_130657

theorem interval_length_for_inequality : ∃ (a b : ℚ),
  (∀ x : ℝ, |5 * x^2 - 2/5| ≤ |x - 8| ↔ a ≤ x ∧ x ≤ b) ∧
  b - a = 13/5 :=
by sorry

end NUMINAMATH_CALUDE_interval_length_for_inequality_l1306_130657


namespace NUMINAMATH_CALUDE_range_of_m_range_of_x_l1306_130648

-- Define propositions p and q
def p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x + 1 ∧ x + 1 < 1 + m ∧ m > 0

-- Theorem 1
theorem range_of_m (m : ℝ) :
  (∀ x, ¬(p x) → ¬(q x m)) →
  0 < m ∧ m ≤ 1 :=
sorry

-- Theorem 2
theorem range_of_x (x : ℝ) :
  (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) →
  (-5 ≤ x ∧ x < -1) ∨ x = 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_x_l1306_130648


namespace NUMINAMATH_CALUDE_hundredths_place_of_seven_twentyfifths_l1306_130693

theorem hundredths_place_of_seven_twentyfifths : ∃ (n : ℕ), (7 : ℚ) / 25 = (n + 28) / 100 ∧ n % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_hundredths_place_of_seven_twentyfifths_l1306_130693


namespace NUMINAMATH_CALUDE_statement_1_incorrect_statement_4_incorrect_l1306_130681

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Statement 1
theorem statement_1_incorrect 
  (α : Plane) (l m n : Line) : 
  ¬(∀ (α : Plane) (l m n : Line), 
    contains α m → contains α n → perpendicular l m → perpendicular l n → 
    perpendicularToPlane l α) := 
by sorry

-- Statement 4
theorem statement_4_incorrect 
  (α : Plane) (l m n : Line) : 
  ¬(∀ (α : Plane) (l m n : Line), 
    contains α m → perpendicularToPlane n α → perpendicular l n → 
    parallel l m) := 
by sorry

end NUMINAMATH_CALUDE_statement_1_incorrect_statement_4_incorrect_l1306_130681


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l1306_130633

theorem product_zero_implies_factor_zero (a b : ℝ) : a * b = 0 → a = 0 ∨ b = 0 := by
  contrapose!
  intro h
  sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l1306_130633


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_4a_l1306_130611

theorem factorization_a_squared_minus_4a (a : ℝ) : a^2 - 4*a = a*(a - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_4a_l1306_130611


namespace NUMINAMATH_CALUDE_min_value_expression_l1306_130621

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x^2 / (x + 2)) + (y^2 / (y + 1)) ≥ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1306_130621


namespace NUMINAMATH_CALUDE_count_valid_sequences_l1306_130630

/-- Represents a die throw result -/
inductive DieThrow
  | even (n : Nat)
  | odd (n : Nat)

/-- Represents a point in 2D space -/
structure Point where
  x : Nat
  y : Nat

/-- Defines how a point moves based on a die throw -/
def move (p : Point) (t : DieThrow) : Point :=
  match t with
  | DieThrow.even n => Point.mk (p.x + n) p.y
  | DieThrow.odd n => Point.mk p.x (p.y + n)

/-- Defines a valid sequence of die throws -/
def validSequence (seq : List DieThrow) : Prop :=
  let finalPoint := seq.foldl move (Point.mk 0 0)
  finalPoint.x = 4 ∧ finalPoint.y = 4

/-- The main theorem to prove -/
theorem count_valid_sequences : 
  (∃ (seqs : List (List DieThrow)), 
    (∀ seq ∈ seqs, validSequence seq) ∧ 
    (∀ seq, validSequence seq → seq ∈ seqs) ∧
    seqs.length = 38) := by
  sorry

end NUMINAMATH_CALUDE_count_valid_sequences_l1306_130630


namespace NUMINAMATH_CALUDE_sqrt_seven_minus_fraction_inequality_l1306_130600

theorem sqrt_seven_minus_fraction_inequality (m n : ℕ) (h : Real.sqrt 7 - (m : ℝ) / n > 0) :
  Real.sqrt 7 - (m : ℝ) / n > 1 / (m * n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_minus_fraction_inequality_l1306_130600


namespace NUMINAMATH_CALUDE_truth_telling_probability_l1306_130632

/-- The probability of two independent events occurring simultaneously -/
def simultaneous_probability (p_a p_b : ℝ) : ℝ := p_a * p_b

/-- Proof that given A speaks the truth 55% of the times and B speaks the truth 60% of the times, 
    the probability that they both tell the truth simultaneously is 0.33 -/
theorem truth_telling_probability : 
  let p_a : ℝ := 0.55
  let p_b : ℝ := 0.60
  simultaneous_probability p_a p_b = 0.33 := by
sorry

end NUMINAMATH_CALUDE_truth_telling_probability_l1306_130632

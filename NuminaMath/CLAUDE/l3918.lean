import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_l3918_391821

def machine_step (N : ℕ) : ℕ :=
  if N % 2 = 1 then 5 * N + 3
  else if N % 3 = 0 then N / 3
  else N + 1

def machine_process (N : ℕ) : ℕ :=
  (machine_step ∘ machine_step ∘ machine_step ∘ machine_step ∘ machine_step) N

theorem unique_solution :
  ∀ N : ℕ, N > 0 → (machine_process N = 1 ↔ N = 6) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3918_391821


namespace NUMINAMATH_CALUDE_cone_lateral_area_l3918_391876

/-- The lateral area of a cone with base radius 3 and height 4 is 15π -/
theorem cone_lateral_area : 
  let r : ℝ := 3
  let h : ℝ := 4
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let S : ℝ := π * r * l
  S = 15 * π := by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l3918_391876


namespace NUMINAMATH_CALUDE_paint_intensity_problem_l3918_391831

theorem paint_intensity_problem (original_intensity new_intensity replacement_fraction : ℝ) 
  (h1 : original_intensity = 0.1)
  (h2 : new_intensity = 0.15)
  (h3 : replacement_fraction = 0.5) :
  let added_intensity := (new_intensity - (1 - replacement_fraction) * original_intensity) / replacement_fraction
  added_intensity = 0.2 := by
sorry

end NUMINAMATH_CALUDE_paint_intensity_problem_l3918_391831


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3918_391807

/-- Given that 2a - b = -3, prove that 4a - 2b = -6 --/
theorem quadratic_expression_value (a b : ℝ) (h : 2 * a - b = -3) :
  4 * a - 2 * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3918_391807


namespace NUMINAMATH_CALUDE_broken_line_length_lower_bound_l3918_391873

/-- A broken line in a square -/
structure BrokenLine where
  -- The square containing the broken line
  square : Set (ℝ × ℝ)
  -- The broken line itself
  line : Set (ℝ × ℝ)
  -- The square has side length 50
  square_side : ∀ (x y : ℝ), (x, y) ∈ square → 0 ≤ x ∧ x ≤ 50 ∧ 0 ≤ y ∧ y ≤ 50
  -- The broken line is contained within the square
  line_in_square : line ⊆ square
  -- For any point in the square, there's a point on the line within distance 1
  close_point : ∀ (p : ℝ × ℝ), p ∈ square → ∃ (q : ℝ × ℝ), q ∈ line ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ 1

/-- The length of a broken line -/
noncomputable def length (bl : BrokenLine) : ℝ := sorry

/-- Theorem: The length of the broken line is greater than 1248 -/
theorem broken_line_length_lower_bound (bl : BrokenLine) : length bl > 1248 := by
  sorry

end NUMINAMATH_CALUDE_broken_line_length_lower_bound_l3918_391873


namespace NUMINAMATH_CALUDE_no_positive_solutions_iff_p_in_range_l3918_391836

/-- The set A of real solutions to the quadratic equation x^2 + (p + 2)x + 1 = 0 -/
def A (p : ℝ) : Set ℝ :=
  {x : ℝ | x^2 + (p + 2)*x + 1 = 0}

/-- The theorem stating the equivalence between A having no positive real solutions
    and p belonging to the specified range -/
theorem no_positive_solutions_iff_p_in_range (p : ℝ) :
  (A p ∩ Set.Ici 0 = ∅) ↔ p ∈ Set.Ioo (-4) 0 ∪ Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_solutions_iff_p_in_range_l3918_391836


namespace NUMINAMATH_CALUDE_consumption_decrease_l3918_391818

/-- Represents a country with its production capabilities -/
structure Country where
  zucchini : ℕ
  cauliflower : ℕ

/-- Calculates the total consumption of each crop under free trade -/
def freeTradeTotalConsumption (a b : Country) : ℕ := by
  sorry

/-- Calculates the total consumption of each crop under autarky -/
def autarkyTotalConsumption (a b : Country) : ℕ := by
  sorry

/-- Theorem stating that consumption decreases by 4 tons when countries merge and trade is banned -/
theorem consumption_decrease (a b : Country) 
  (h1 : a.zucchini = 20 ∧ a.cauliflower = 16)
  (h2 : b.zucchini = 36 ∧ b.cauliflower = 24) :
  freeTradeTotalConsumption a b - autarkyTotalConsumption a b = 4 := by
  sorry

end NUMINAMATH_CALUDE_consumption_decrease_l3918_391818


namespace NUMINAMATH_CALUDE_min_both_beethoven_vivaldi_l3918_391884

/-- The minimum number of people who like both Beethoven and Vivaldi in a group of 120 people,
    where 95 like Beethoven and 80 like Vivaldi. -/
theorem min_both_beethoven_vivaldi (total : ℕ) (beethoven : ℕ) (vivaldi : ℕ)
    (h_total : total = 120)
    (h_beethoven : beethoven = 95)
    (h_vivaldi : vivaldi = 80) :
    beethoven + vivaldi - total ≥ 55 := by
  sorry

end NUMINAMATH_CALUDE_min_both_beethoven_vivaldi_l3918_391884


namespace NUMINAMATH_CALUDE_translate_line_upward_5_units_l3918_391859

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- Translates a linear function vertically by a given amount -/
def translateVertically (f : LinearFunction) (amount : ℝ) : LinearFunction :=
  { slope := f.slope, yIntercept := f.yIntercept + amount }

theorem translate_line_upward_5_units :
  let original : LinearFunction := { slope := 2, yIntercept := -4 }
  let translated := translateVertically original 5
  translated = { slope := 2, yIntercept := 1 } := by
  sorry

end NUMINAMATH_CALUDE_translate_line_upward_5_units_l3918_391859


namespace NUMINAMATH_CALUDE_shipping_cost_for_five_pounds_l3918_391893

/-- Calculates the shipping cost based on weight and rates -/
def shipping_cost (flat_fee : ℝ) (per_pound_rate : ℝ) (weight : ℝ) : ℝ :=
  flat_fee + per_pound_rate * weight

/-- Proves that the shipping cost for a 5-pound package is $9.00 -/
theorem shipping_cost_for_five_pounds :
  shipping_cost 5 0.8 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_for_five_pounds_l3918_391893


namespace NUMINAMATH_CALUDE_total_fireworks_is_1188_l3918_391815

/-- Calculates the total number of fireworks used in the New Year's Eve display -/
def total_fireworks : ℕ :=
  let fireworks_per_number : ℕ := 6
  let fireworks_per_regular_letter : ℕ := 5
  let fireworks_for_H : ℕ := 8
  let fireworks_for_E : ℕ := 7
  let fireworks_for_L : ℕ := 6
  let fireworks_for_O : ℕ := 9
  let fireworks_for_square : ℕ := 4
  let fireworks_for_triangle : ℕ := 3
  let fireworks_for_circle : ℕ := 12
  let additional_boxes : ℕ := 100
  let fireworks_per_box : ℕ := 10

  let years_fireworks := fireworks_per_number * 4 * 3
  let happy_new_year_fireworks := fireworks_per_regular_letter * 11 + fireworks_per_number
  let geometric_shapes_fireworks := fireworks_for_square + fireworks_for_triangle + fireworks_for_circle
  let hello_fireworks := fireworks_for_H + fireworks_for_E + fireworks_for_L * 2 + fireworks_for_O
  let additional_fireworks := additional_boxes * fireworks_per_box

  years_fireworks + happy_new_year_fireworks + geometric_shapes_fireworks + hello_fireworks + additional_fireworks

theorem total_fireworks_is_1188 : total_fireworks = 1188 := by
  sorry

end NUMINAMATH_CALUDE_total_fireworks_is_1188_l3918_391815


namespace NUMINAMATH_CALUDE_saras_sister_notebooks_l3918_391855

/-- Calculates the final number of notebooks given initial, ordered, and lost quantities. -/
def final_notebooks (initial ordered lost : ℕ) : ℕ :=
  initial + ordered - lost

/-- Theorem stating that Sara's sister's final number of notebooks is 8. -/
theorem saras_sister_notebooks : final_notebooks 4 6 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_saras_sister_notebooks_l3918_391855


namespace NUMINAMATH_CALUDE_man_brother_age_difference_l3918_391890

/-- Represents the age difference between a man and his brother -/
def ageDifference (manAge brotherAge : ℕ) : ℕ := manAge - brotherAge

/-- The problem statement -/
theorem man_brother_age_difference :
  ∀ (manAge brotherAge : ℕ),
    brotherAge = 10 →
    manAge > brotherAge →
    manAge + 2 = 2 * (brotherAge + 2) →
    ageDifference manAge brotherAge = 12 := by
  sorry

end NUMINAMATH_CALUDE_man_brother_age_difference_l3918_391890


namespace NUMINAMATH_CALUDE_perfect_square_addition_l3918_391879

theorem perfect_square_addition : ∃ x : ℤ,
  (∃ a : ℤ, 100 + x = a^2) ∧
  (∃ b : ℤ, 164 + x = b^2) ∧
  x = 125 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_addition_l3918_391879


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3918_391878

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 15*x + 6 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = 15 ∧ r₁ * r₂ = 6 ∧ r₁^2 + r₂^2 = 213 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3918_391878


namespace NUMINAMATH_CALUDE_product_sum_reciprocals_l3918_391829

theorem product_sum_reciprocals : (3 * 5 * 7) * (1/3 + 1/5 + 1/7) = 71 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_reciprocals_l3918_391829


namespace NUMINAMATH_CALUDE_nested_average_equality_l3918_391891

def avg_pair (a b : ℚ) : ℚ := (a + b) / 2

def avg_quad (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem nested_average_equality : 
  avg_quad 
    (avg_quad (avg_pair 2 4) (avg_pair 1 3) (avg_pair 0 2) (avg_pair 1 1))
    (avg_pair 3 3)
    (avg_pair 2 2)
    (avg_pair 4 0) = 35 / 16 := by
  sorry

end NUMINAMATH_CALUDE_nested_average_equality_l3918_391891


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3918_391874

/-- For an arithmetic sequence with general term a_n = 3n - 4, 
    the difference between the first term and the common difference is -4. -/
theorem arithmetic_sequence_property : 
  ∀ (a : ℕ → ℤ), 
  (∀ n, a n = 3*n - 4) → 
  (a 1 - (a 2 - a 1) = -4) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3918_391874


namespace NUMINAMATH_CALUDE_marshmallow_challenge_l3918_391837

/-- The Marshmallow Challenge Theorem -/
theorem marshmallow_challenge (haley michael brandon : ℕ) 
  (haley_holds : haley = 8)
  (michael_holds : michael = 3 * haley)
  (brandon_holds : brandon = michael / 2) :
  haley + michael + brandon = 44 := by
  sorry

end NUMINAMATH_CALUDE_marshmallow_challenge_l3918_391837


namespace NUMINAMATH_CALUDE_equation_solution_l3918_391867

theorem equation_solution :
  let x : ℝ := (173 * 240) / 120
  ∃ ε > 0, ε < 0.005 ∧ |x - 345.33| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3918_391867


namespace NUMINAMATH_CALUDE_delegation_selection_ways_l3918_391814

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of men in the brigade -/
def num_men : ℕ := 10

/-- The number of women in the brigade -/
def num_women : ℕ := 8

/-- The number of men to be selected for the delegation -/
def men_in_delegation : ℕ := 3

/-- The number of women to be selected for the delegation -/
def women_in_delegation : ℕ := 2

/-- The theorem stating the number of ways to select the delegation -/
theorem delegation_selection_ways :
  (choose num_men men_in_delegation) * (choose num_women women_in_delegation) = 3360 := by
  sorry

end NUMINAMATH_CALUDE_delegation_selection_ways_l3918_391814


namespace NUMINAMATH_CALUDE_jump_rope_record_rate_l3918_391896

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The record number of consecutive ropes jumped -/
def record_jumps : ℕ := 54000

/-- The time limit in hours -/
def time_limit : ℕ := 5

/-- The required rate of jumps per second -/
def required_rate : ℚ := 3

theorem jump_rope_record_rate :
  (record_jumps : ℚ) / ((time_limit * seconds_per_hour) : ℚ) = required_rate :=
sorry

end NUMINAMATH_CALUDE_jump_rope_record_rate_l3918_391896


namespace NUMINAMATH_CALUDE_ladder_angle_elevation_l3918_391849

def ladder_foot_distance : ℝ := 4.6
def ladder_length : ℝ := 9.2

theorem ladder_angle_elevation :
  let cos_angle := ladder_foot_distance / ladder_length
  let angle := Real.arccos cos_angle
  ∃ ε > 0, abs (angle - Real.pi / 3) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ladder_angle_elevation_l3918_391849


namespace NUMINAMATH_CALUDE_smallest_n_for_seating_arrangement_l3918_391827

theorem smallest_n_for_seating_arrangement (k : ℕ) : 
  (2 ≤ k) → 
  (∃ n : ℕ, 
    k < n ∧ 
    (2 * (n - 1).factorial * (n - k + 2) = n * (n - 1).factorial) ∧
    (∀ m : ℕ, m < n → 
      (2 ≤ m ∧ k < m) → 
      (2 * (m - 1).factorial * (m - k + 2) ≠ m * (m - 1).factorial))) → 
  (∃ n : ℕ, 
    k < n ∧ 
    (2 * (n - 1).factorial * (n - k + 2) = n * (n - 1).factorial) ∧
    n = 12) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_seating_arrangement_l3918_391827


namespace NUMINAMATH_CALUDE_aziz_parents_years_in_america_before_birth_l3918_391826

theorem aziz_parents_years_in_america_before_birth :
  let current_year : ℕ := 2021
  let aziz_age : ℕ := 36
  let parents_move_year : ℕ := 1982
  let aziz_birth_year : ℕ := current_year - aziz_age
  let years_before_birth : ℕ := aziz_birth_year - parents_move_year
  years_before_birth = 3 :=
by sorry

end NUMINAMATH_CALUDE_aziz_parents_years_in_america_before_birth_l3918_391826


namespace NUMINAMATH_CALUDE_vertical_distance_to_charlie_l3918_391822

/-- The vertical distance between the midpoint of the line segment connecting
    (8, -15) and (2, 10), and the point (5, 3) is 5.5 units. -/
theorem vertical_distance_to_charlie : 
  let annie : ℝ × ℝ := (8, -15)
  let barbara : ℝ × ℝ := (2, 10)
  let charlie : ℝ × ℝ := (5, 3)
  let midpoint : ℝ × ℝ := ((annie.1 + barbara.1) / 2, (annie.2 + barbara.2) / 2)
  charlie.2 - midpoint.2 = 5.5 := by sorry

end NUMINAMATH_CALUDE_vertical_distance_to_charlie_l3918_391822


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3918_391846

theorem intersection_of_sets :
  let P : Set ℕ := {1, 3, 5}
  let Q : Set ℕ := {x | 2 ≤ x ∧ x ≤ 5}
  P ∩ Q = {3, 5} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3918_391846


namespace NUMINAMATH_CALUDE_triangle_formation_l3918_391861

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given sticks of length 6 and 12, proves which of the given lengths can form a triangle -/
theorem triangle_formation (l : ℝ) : 
  (l = 5 ∨ l = 6 ∨ l = 11 ∨ l = 20) → 
  (can_form_triangle 6 12 l ↔ l = 11) :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l3918_391861


namespace NUMINAMATH_CALUDE_average_speed_on_time_l3918_391887

/-- The average speed needed to reach the destination on time given the conditions -/
theorem average_speed_on_time (total_distance : ℝ) (late_speed : ℝ) (late_time : ℝ) :
  total_distance = 70 →
  late_speed = 35 →
  late_time = 0.25 →
  (total_distance / late_speed) - late_time = total_distance / 40 :=
by sorry

end NUMINAMATH_CALUDE_average_speed_on_time_l3918_391887


namespace NUMINAMATH_CALUDE_average_b_c_is_70_l3918_391899

/-- Given two numbers a and b with an average of 50, and a third number c such that c - a = 40,
    prove that the average of b and c is 70. -/
theorem average_b_c_is_70 (a b c : ℝ) 
    (h1 : (a + b) / 2 = 50)
    (h2 : c - a = 40) : 
  (b + c) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_b_c_is_70_l3918_391899


namespace NUMINAMATH_CALUDE_stamp_exhibition_l3918_391840

theorem stamp_exhibition (people : ℕ) (total_stamps : ℕ) : 
  (3 * people + 24 = total_stamps) →
  (4 * people = total_stamps + 26) →
  total_stamps = 174 := by
sorry

end NUMINAMATH_CALUDE_stamp_exhibition_l3918_391840


namespace NUMINAMATH_CALUDE_employee_pay_calculation_l3918_391843

def total_pay : ℝ := 570
def x_pay_ratio : ℝ := 1.2

theorem employee_pay_calculation (x y : ℝ) 
  (h1 : x + y = total_pay) 
  (h2 : x = x_pay_ratio * y) : 
  y = 259.09 := by sorry

end NUMINAMATH_CALUDE_employee_pay_calculation_l3918_391843


namespace NUMINAMATH_CALUDE_first_equation_is_midpoint_second_equation_is_midpoint_iff_l3918_391800

/-- Definition of a midpoint equation -/
def is_midpoint_equation (a b : ℚ) : Prop :=
  a ≠ 0 ∧ ((-b) / a = (a + b) / 2)

/-- First part of the problem -/
theorem first_equation_is_midpoint : is_midpoint_equation 4 (-8/3) := by
  sorry

/-- Second part of the problem -/
theorem second_equation_is_midpoint_iff (m : ℚ) : 
  is_midpoint_equation 5 (m - 1) ↔ m = -18/7 := by
  sorry

end NUMINAMATH_CALUDE_first_equation_is_midpoint_second_equation_is_midpoint_iff_l3918_391800


namespace NUMINAMATH_CALUDE_pharmacy_purchase_cost_bob_pharmacy_purchase_cost_l3918_391872

/-- Calculates the total cost of a pharmacy purchase including sales tax -/
theorem pharmacy_purchase_cost (nose_spray_cost : ℚ) (nose_spray_count : ℕ) 
  (nose_spray_discount : ℚ) (cough_syrup_cost : ℚ) (cough_syrup_count : ℕ) 
  (cough_syrup_discount : ℚ) (ibuprofen_cost : ℚ) (ibuprofen_count : ℕ) 
  (sales_tax_rate : ℚ) : ℚ :=
  let nose_spray_total := (nose_spray_cost * ↑(nose_spray_count / 2)) * (1 - nose_spray_discount)
  let cough_syrup_total := (cough_syrup_cost * ↑cough_syrup_count) * (1 - cough_syrup_discount)
  let ibuprofen_total := ibuprofen_cost * ↑ibuprofen_count
  let subtotal := nose_spray_total + cough_syrup_total + ibuprofen_total
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  total_with_tax

/-- The total cost of Bob's pharmacy purchase, rounded to the nearest cent, is $56.38 -/
theorem bob_pharmacy_purchase_cost : 
  ⌊pharmacy_purchase_cost 3 10 (1/5) 7 4 (1/10) 5 3 (2/25) * 100⌋ / 100 = 56381 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_pharmacy_purchase_cost_bob_pharmacy_purchase_cost_l3918_391872


namespace NUMINAMATH_CALUDE_darren_tshirts_l3918_391809

/-- The number of packs of white t-shirts Darren bought -/
def white_packs : ℕ := 5

/-- The number of packs of blue t-shirts Darren bought -/
def blue_packs : ℕ := 3

/-- The number of t-shirts in each pack of white t-shirts -/
def white_per_pack : ℕ := 6

/-- The number of t-shirts in each pack of blue t-shirts -/
def blue_per_pack : ℕ := 9

/-- The total number of t-shirts Darren bought -/
def total_tshirts : ℕ := white_packs * white_per_pack + blue_packs * blue_per_pack

theorem darren_tshirts : total_tshirts = 57 := by
  sorry

end NUMINAMATH_CALUDE_darren_tshirts_l3918_391809


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l3918_391869

theorem quadratic_root_implies_coefficient (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 6 = 0 ∧ x = 2) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l3918_391869


namespace NUMINAMATH_CALUDE_units_digit_of_4569_pow_804_l3918_391810

theorem units_digit_of_4569_pow_804 : (4569^804) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_4569_pow_804_l3918_391810


namespace NUMINAMATH_CALUDE_percentage_equivalence_l3918_391892

theorem percentage_equivalence (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equivalence_l3918_391892


namespace NUMINAMATH_CALUDE_solution_value_l3918_391844

theorem solution_value (m : ℝ) : (3 * m - 2 * 3 = 6) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3918_391844


namespace NUMINAMATH_CALUDE_degree_of_specific_monomial_l3918_391886

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (x_exp y_exp : ℕ) : ℕ := x_exp + y_exp

/-- The monomial -1/4 * π * x^2 * y^3 has degree 5 -/
theorem degree_of_specific_monomial :
  degree_of_monomial 2 3 = 5 := by sorry

end NUMINAMATH_CALUDE_degree_of_specific_monomial_l3918_391886


namespace NUMINAMATH_CALUDE_alligator_journey_time_l3918_391832

/-- The additional time taken for the return journey of alligators -/
def additional_time (initial_time : ℕ) (total_alligators : ℕ) (total_time : ℕ) : ℕ :=
  (total_time - initial_time) / total_alligators - initial_time

/-- Theorem stating that the additional time for the return journey is 2 hours -/
theorem alligator_journey_time : additional_time 4 7 46 = 2 := by
  sorry

end NUMINAMATH_CALUDE_alligator_journey_time_l3918_391832


namespace NUMINAMATH_CALUDE_correct_operation_l3918_391862

theorem correct_operation (a b : ℝ) : -a^2*b + 2*a^2*b = a^2*b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3918_391862


namespace NUMINAMATH_CALUDE_box_depth_l3918_391845

theorem box_depth (length width : ℕ) (num_cubes : ℕ) (depth : ℕ) : 
  length = 35 → 
  width = 20 → 
  num_cubes = 56 →
  (∃ (cube_edge : ℕ), 
    cube_edge ∣ length ∧ 
    cube_edge ∣ width ∧ 
    cube_edge ∣ depth ∧
    cube_edge ^ 3 * num_cubes = length * width * depth) →
  depth = 10 := by
  sorry


end NUMINAMATH_CALUDE_box_depth_l3918_391845


namespace NUMINAMATH_CALUDE_train_speed_increase_l3918_391841

theorem train_speed_increase (distance : ℝ) (speed_increase : ℝ) (time_reduction : ℝ) (speed_limit : ℝ)
  (h1 : distance = 1600)
  (h2 : speed_increase = 20)
  (h3 : time_reduction = 4)
  (h4 : speed_limit = 140) :
  ∃ (original_speed : ℝ),
    original_speed > 0 ∧
    distance / original_speed = distance / (original_speed + speed_increase) + time_reduction ∧
    original_speed + speed_increase < speed_limit :=
by sorry

#check train_speed_increase

end NUMINAMATH_CALUDE_train_speed_increase_l3918_391841


namespace NUMINAMATH_CALUDE_congruence_solution_l3918_391813

theorem congruence_solution (n : ℤ) : (13 * n) % 47 = 8 ↔ n % 47 = 4 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l3918_391813


namespace NUMINAMATH_CALUDE_townspeople_win_probability_l3918_391850

/-- The probability that the townspeople win in a game with 2 townspeople and 1 goon -/
theorem townspeople_win_probability :
  let total_participants : ℕ := 2 + 1
  let num_goons : ℕ := 1
  let townspeople_win_condition := (num_goons / total_participants : ℚ)
  townspeople_win_condition = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_townspeople_win_probability_l3918_391850


namespace NUMINAMATH_CALUDE_lucca_ball_count_l3918_391804

theorem lucca_ball_count :
  ∀ (lucca_balls : ℕ) (lucca_basketballs : ℕ) (lucien_basketballs : ℕ),
    lucca_basketballs = lucca_balls / 10 →
    lucien_basketballs = 40 →
    lucca_basketballs + lucien_basketballs = 50 →
    lucca_balls = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_lucca_ball_count_l3918_391804


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l3918_391852

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (69/29, 43/29)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 5*x - 6*y = 3

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 8*x + 2*y = 22

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem intersection_point_unique :
  ∀ x y : ℚ, line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_intersection_point_unique_l3918_391852


namespace NUMINAMATH_CALUDE_tan_sqrt_three_iff_periodic_l3918_391863

theorem tan_sqrt_three_iff_periodic (x : ℝ) : 
  Real.tan x = Real.sqrt 3 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sqrt_three_iff_periodic_l3918_391863


namespace NUMINAMATH_CALUDE_smallest_k_for_zero_difference_l3918_391833

def u (n : ℕ) := n^4 + n^2 + n

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
  | f => fun n => f (n + 1) - f n

def iteratedΔ : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
  | 0 => id
  | k + 1 => Δ ∘ iteratedΔ k

theorem smallest_k_for_zero_difference :
  ∃ k, k = 5 ∧ 
    (∀ n, iteratedΔ k u n = 0) ∧
    (∀ j < k, ∃ n, iteratedΔ j u n ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_zero_difference_l3918_391833


namespace NUMINAMATH_CALUDE_divisibility_of_repeated_eight_l3918_391883

theorem divisibility_of_repeated_eight : ∃ k : ℕ, 8 * (10^1974 - 1) / 9 = 13 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_repeated_eight_l3918_391883


namespace NUMINAMATH_CALUDE_lilacs_sold_l3918_391870

/-- Represents the number of lilacs sold -/
def lilacs : ℕ := sorry

/-- Represents the number of roses sold -/
def roses : ℕ := 3 * lilacs

/-- Represents the number of gardenias sold -/
def gardenias : ℕ := lilacs / 2

/-- The total number of flowers sold -/
def total_flowers : ℕ := 45

/-- Theorem stating that the number of lilacs sold is 10 -/
theorem lilacs_sold : lilacs = 10 := by
  sorry

end NUMINAMATH_CALUDE_lilacs_sold_l3918_391870


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_l3918_391854

theorem quadratic_equation_sum (x p q : ℝ) : 
  (5 * x^2 - 30 * x - 45 = 0) → 
  ((x + p)^2 = q) → 
  (p + q = 15) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_l3918_391854


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3918_391847

theorem square_sum_geq_product_sum (x y z : ℝ) : x^2 + y^2 + z^2 ≥ x*y + y*z + z*x := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l3918_391847


namespace NUMINAMATH_CALUDE_max_value_sin_cos_sum_l3918_391808

theorem max_value_sin_cos_sum (a b : ℝ) :
  ∃ (M : ℝ), M = Real.sqrt (a^2 + b^2) ∧
  (∀ t : ℝ, 0 < t ∧ t < 2 * Real.pi → a * Real.sin t + b * Real.cos t ≤ M) ∧
  (∃ t : ℝ, 0 < t ∧ t < 2 * Real.pi ∧ a * Real.sin t + b * Real.cos t = M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_sum_l3918_391808


namespace NUMINAMATH_CALUDE_equation_solutions_l3918_391894

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -Real.sqrt 2 ∧ x₁^2 + Real.sqrt 2 * x₁ - Real.sqrt 6 = Real.sqrt 3 * x₁) ∧
  (x₂ = Real.sqrt 3 ∧ x₂^2 + Real.sqrt 2 * x₂ - Real.sqrt 6 = Real.sqrt 3 * x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3918_391894


namespace NUMINAMATH_CALUDE_tim_balloon_count_l3918_391824

/-- Given that Dan has 58.0 violet balloons and 10.0 times more violet balloons than Tim,
    prove that Tim has 5.8 violet balloons. -/
theorem tim_balloon_count : 
  ∀ (dan_balloons tim_balloons : ℝ),
    dan_balloons = 58.0 →
    dan_balloons = 10.0 * tim_balloons →
    tim_balloons = 5.8 := by
  sorry

end NUMINAMATH_CALUDE_tim_balloon_count_l3918_391824


namespace NUMINAMATH_CALUDE_turtle_count_l3918_391812

/-- Represents the number of turtles in the lake -/
def total_turtles : ℕ := 100

/-- Percentage of female turtles -/
def female_percentage : ℚ := 60 / 100

/-- Percentage of male turtles with stripes -/
def male_striped_percentage : ℚ := 25 / 100

/-- Number of baby male turtles with stripes -/
def baby_striped_males : ℕ := 4

/-- Percentage of adult male turtles with stripes -/
def adult_striped_percentage : ℚ := 60 / 100

theorem turtle_count :
  total_turtles = 100 :=
sorry

end NUMINAMATH_CALUDE_turtle_count_l3918_391812


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_q_unique_l3918_391860

/-- The cubic polynomial q(x) that satisfies the given conditions -/
def q (x : ℝ) : ℝ := -4 * x^3 + 24 * x^2 - 44 * x + 24

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  q 1 = 0 ∧ q 2 = 0 ∧ q 3 = 0 ∧ q 4 = -24 := by
  sorry

/-- Theorem stating that q(x) is the unique cubic polynomial satisfying the conditions -/
theorem q_unique (p : ℝ → ℝ) (h_cubic : ∃ a b c d, ∀ x, p x = a * x^3 + b * x^2 + c * x + d) 
  (h_cond : p 1 = 0 ∧ p 2 = 0 ∧ p 3 = 0 ∧ p 4 = -24) :
  ∀ x, p x = q x := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_q_unique_l3918_391860


namespace NUMINAMATH_CALUDE_max_sum_abc_l3918_391816

theorem max_sum_abc (a b c : ℤ) 
  (h1 : a + b = 2006) 
  (h2 : c - a = 2005) 
  (h3 : a < b) : 
  ∃ (m : ℤ), m = 5013 ∧ a + b + c ≤ m ∧ ∃ (a' b' c' : ℤ), a' + b' = 2006 ∧ c' - a' = 2005 ∧ a' < b' ∧ a' + b' + c' = m :=
sorry

end NUMINAMATH_CALUDE_max_sum_abc_l3918_391816


namespace NUMINAMATH_CALUDE_spurs_basketballs_l3918_391877

/-- The total number of basketballs for a team -/
def total_basketballs (num_players : ℕ) (balls_per_player : ℕ) : ℕ :=
  num_players * balls_per_player

/-- Theorem: A team of 22 players, each with 11 basketballs, has 242 basketballs in total -/
theorem spurs_basketballs : total_basketballs 22 11 = 242 := by
  sorry

end NUMINAMATH_CALUDE_spurs_basketballs_l3918_391877


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l3918_391835

theorem cube_edge_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^3 / b^3 = 64 → a / b = 4 := by
sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l3918_391835


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l3918_391823

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 143 → x^2 + y^2 ≥ 145 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l3918_391823


namespace NUMINAMATH_CALUDE_problem_statement_l3918_391834

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = x + y) :
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = a + b → 1/a + 1/b ≥ 2) ∧
  (1/x + 1/y = 2 ↔ x = 1 ∧ y = 1) ∧
  ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = a + b ∧ (a + 1) * (b + 1) = 5 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3918_391834


namespace NUMINAMATH_CALUDE_vector_parallel_implies_m_l3918_391830

def vector_a (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (-3, 1)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_parallel_implies_m (m : ℝ) :
  parallel ((2 * vector_a m).1 + vector_b.1, (2 * vector_a m).2 + vector_b.2) vector_b →
  m = -1/3 := by
sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_m_l3918_391830


namespace NUMINAMATH_CALUDE_percentage_increase_l3918_391868

theorem percentage_increase (N : ℝ) (P : ℝ) : 
  N = 80 →
  N + (P / 100) * N - (N - (25 / 100) * N) = 30 →
  P = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3918_391868


namespace NUMINAMATH_CALUDE_circle_equation_a_range_l3918_391838

/-- A circle in the xy-plane can be represented by the equation x^2 + y^2 - 2x + 2y + a = 0,
    where a is a real number. This theorem states that the range of a for which this equation
    represents a circle is (-∞, 2). -/
theorem circle_equation_a_range :
  ∀ a : ℝ, (∃ x y : ℝ, x^2 + y^2 - 2*x + 2*y + a = 0 ∧ 
    ∀ x' y' : ℝ, x'^2 + y'^2 - 2*x' + 2*y' + a = 0 → (x' - x)^2 + (y' - y)^2 = Constant)
  ↔ a < 2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_a_range_l3918_391838


namespace NUMINAMATH_CALUDE_sum_of_integers_l3918_391825

theorem sum_of_integers : (-25) + 34 + 156 + (-65) = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3918_391825


namespace NUMINAMATH_CALUDE_cloth_cost_price_calculation_l3918_391828

/-- The cost price of one metre of cloth, given the selling details --/
def cost_price_per_metre (cloth_length : ℕ) (selling_price : ℚ) (profit_per_metre : ℚ) : ℚ :=
  (selling_price - cloth_length * profit_per_metre) / cloth_length

theorem cloth_cost_price_calculation :
  let cloth_length : ℕ := 92
  let selling_price : ℚ := 9890
  let profit_per_metre : ℚ := 24
  cost_price_per_metre cloth_length selling_price profit_per_metre = 83.5 := by
sorry


end NUMINAMATH_CALUDE_cloth_cost_price_calculation_l3918_391828


namespace NUMINAMATH_CALUDE_f_zero_gt_f_four_l3918_391875

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- State that f is differentiable on ℝ
variable (hf : Differentiable ℝ f)

-- Define the condition that f(x) = x² + 2f''(2)x - 3
variable (hf_eq : ∀ x, f x = x^2 + 2 * (deriv^[2] f 2) * x - 3)

-- Theorem to prove
theorem f_zero_gt_f_four : f 0 > f 4 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_gt_f_four_l3918_391875


namespace NUMINAMATH_CALUDE_factorization_equality_l3918_391858

theorem factorization_equality (a b : ℝ) : a * b^2 - a = a * (b + 1) * (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3918_391858


namespace NUMINAMATH_CALUDE_total_age_problem_l3918_391889

/-- Given three people a, b, and c, where a is two years older than b, 
    b is twice as old as c, and b is 10 years old, 
    prove that the total of their ages is 27 years. -/
theorem total_age_problem (a b c : ℕ) : 
  b = 10 → a = b + 2 → b = 2 * c → a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_total_age_problem_l3918_391889


namespace NUMINAMATH_CALUDE_robert_ate_ten_chocolates_l3918_391839

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- The number of additional chocolates Robert ate compared to Nickel -/
def robert_additional_chocolates : ℕ := 5

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := nickel_chocolates + robert_additional_chocolates

theorem robert_ate_ten_chocolates : robert_chocolates = 10 := by
  sorry

end NUMINAMATH_CALUDE_robert_ate_ten_chocolates_l3918_391839


namespace NUMINAMATH_CALUDE_x_not_equal_one_l3918_391811

theorem x_not_equal_one (x : ℝ) (h : (x - 1)^0 = 1) : x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_x_not_equal_one_l3918_391811


namespace NUMINAMATH_CALUDE_tangent_slope_acute_implies_a_equals_one_l3918_391856

/-- Given a curve C: y = x^3 - 2ax^2 + 2ax, if the slope of the tangent line
    at any point on the curve is acute, then a = 1, where a is an integer. -/
theorem tangent_slope_acute_implies_a_equals_one (a : ℤ) : 
  (∀ x : ℝ, 0 < 3*x^2 - 4*a*x + 2*a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_acute_implies_a_equals_one_l3918_391856


namespace NUMINAMATH_CALUDE_equation_and_inequality_solution_l3918_391819

theorem equation_and_inequality_solution :
  (∃ x : ℝ, 3 * (x - 2) - (1 - 2 * x) = 3 ∧ x = 2) ∧
  (∀ x : ℝ, 2 * x - 1 < 4 * x + 3 ↔ x > -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_and_inequality_solution_l3918_391819


namespace NUMINAMATH_CALUDE_cake_frosting_time_difference_l3918_391866

/-- The time difference for frosting 10 cakes between normal and sprained wrist conditions -/
theorem cake_frosting_time_difference 
  (normal_time : ℕ) 
  (sprained_time : ℕ) 
  (num_cakes : ℕ) 
  (h1 : normal_time = 5)
  (h2 : sprained_time = 8)
  (h3 : num_cakes = 10) : 
  (sprained_time * num_cakes) - (normal_time * num_cakes) = 30 := by
  sorry

#check cake_frosting_time_difference

end NUMINAMATH_CALUDE_cake_frosting_time_difference_l3918_391866


namespace NUMINAMATH_CALUDE_florist_bouquets_l3918_391820

/-- The number of flower colors --/
def num_colors : ℕ := 4

/-- The number of flowers in each bouquet --/
def flowers_per_bouquet : ℕ := 9

/-- The number of seeds planted for each color --/
def seeds_per_color : ℕ := 125

/-- The number of red flowers killed by fungus --/
def red_killed : ℕ := 45

/-- The number of yellow flowers killed by fungus --/
def yellow_killed : ℕ := 61

/-- The number of orange flowers killed by fungus --/
def orange_killed : ℕ := 30

/-- The number of purple flowers killed by fungus --/
def purple_killed : ℕ := 40

/-- Theorem: The florist can make 36 bouquets --/
theorem florist_bouquets :
  (num_colors * seeds_per_color - (red_killed + yellow_killed + orange_killed + purple_killed)) / flowers_per_bouquet = 36 :=
by sorry

end NUMINAMATH_CALUDE_florist_bouquets_l3918_391820


namespace NUMINAMATH_CALUDE_bales_stored_l3918_391802

theorem bales_stored (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 73)
  (h2 : final_bales = 96) :
  final_bales - initial_bales = 23 := by
sorry

end NUMINAMATH_CALUDE_bales_stored_l3918_391802


namespace NUMINAMATH_CALUDE_tournament_max_matches_l3918_391898

/-- Represents a round-robin tennis tournament -/
structure TennisTournament where
  players : ℕ
  original_days : ℕ
  rest_days : ℕ

/-- Calculates the maximum number of matches that can be completed in a tournament -/
def max_matches (t : TennisTournament) : ℕ :=
  min
    ((t.players * (t.players - 1)) / 2)
    ((t.players / 2) * (t.original_days - t.rest_days))

/-- Theorem: In a tournament with 10 players, 9 original days, and 1 rest day, 
    the maximum number of matches is 40 -/
theorem tournament_max_matches :
  let t : TennisTournament := ⟨10, 9, 1⟩
  max_matches t = 40 := by
  sorry


end NUMINAMATH_CALUDE_tournament_max_matches_l3918_391898


namespace NUMINAMATH_CALUDE_grid_routes_equal_binomial_coefficient_l3918_391805

def grid_width : ℕ := 10
def grid_height : ℕ := 5

def num_routes : ℕ := Nat.choose (grid_width + grid_height) grid_height

theorem grid_routes_equal_binomial_coefficient :
  num_routes = Nat.choose (grid_width + grid_height) grid_height :=
by sorry

end NUMINAMATH_CALUDE_grid_routes_equal_binomial_coefficient_l3918_391805


namespace NUMINAMATH_CALUDE_c_7_equals_448_l3918_391885

/-- Sequence definition -/
def a (n : ℕ) : ℕ := n

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ := a n * b n

/-- Theorem stating that c_7 equals 448 -/
theorem c_7_equals_448 : c 7 = 448 := by
  sorry

end NUMINAMATH_CALUDE_c_7_equals_448_l3918_391885


namespace NUMINAMATH_CALUDE_double_y_plus_8_not_less_than_negative_3_l3918_391853

theorem double_y_plus_8_not_less_than_negative_3 :
  ∀ y : ℝ, (2 * y + 8 ≥ -3) ↔ (∃ z : ℝ, z = 2 * y ∧ z + 8 ≥ -3) :=
by sorry

end NUMINAMATH_CALUDE_double_y_plus_8_not_less_than_negative_3_l3918_391853


namespace NUMINAMATH_CALUDE_cody_age_l3918_391880

theorem cody_age (grandmother_age : ℕ) (age_ratio : ℕ) (cody_age : ℕ) : 
  grandmother_age = 84 →
  grandmother_age = age_ratio * cody_age →
  age_ratio = 6 →
  cody_age = 14 := by
sorry

end NUMINAMATH_CALUDE_cody_age_l3918_391880


namespace NUMINAMATH_CALUDE_age_ratio_equation_exists_l3918_391842

/-- Represents the ages of three people in terms of a common multiplier -/
structure AgeRatio :=
  (x : ℝ)  -- Common multiplier
  (y : ℝ)  -- Number of years ago

/-- The equation relating the ages and the sum from y years ago -/
def ageEquation (r : AgeRatio) : Prop :=
  20 * r.x - 3 * r.y = 76

theorem age_ratio_equation_exists :
  ∃ r : AgeRatio, ageEquation r :=
sorry

end NUMINAMATH_CALUDE_age_ratio_equation_exists_l3918_391842


namespace NUMINAMATH_CALUDE_function_equality_l3918_391803

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- Theorem statement
theorem function_equality (x : ℝ) : 
  (2 * f x - 10 = f (x - 2)) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l3918_391803


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l3918_391848

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- State the theorem
theorem sum_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a5 : a 5 = 3) 
  (h_a6 : a 6 = -2) : 
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3) :=
sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l3918_391848


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3918_391817

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific arithmetic sequence from the problem
def specific_sequence (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ a 3 = 7 ∧ a 7 = 3

-- Theorem statement
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) (h : specific_sequence a) : 
  ∀ n : ℕ, a n = -n + 10 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3918_391817


namespace NUMINAMATH_CALUDE_sum_of_ages_is_41_l3918_391864

/-- The sum of Henry and Jill's present ages -/
def sumOfAges (henryAge : ℕ) (jillAge : ℕ) : ℕ :=
  henryAge + jillAge

/-- Theorem stating that the sum of Henry and Jill's present ages is 41 -/
theorem sum_of_ages_is_41 (henryAge : ℕ) (jillAge : ℕ) 
  (h1 : henryAge = 25) 
  (h2 : jillAge = 16) : 
  sumOfAges henryAge jillAge = 41 := by
  sorry

#check sum_of_ages_is_41

end NUMINAMATH_CALUDE_sum_of_ages_is_41_l3918_391864


namespace NUMINAMATH_CALUDE_cleaning_time_proof_l3918_391888

def grove_width : ℕ := 4
def grove_length : ℕ := 5
def cleaning_time_per_tree : ℕ := 6
def minutes_per_hour : ℕ := 60

theorem cleaning_time_proof :
  let total_trees := grove_width * grove_length
  let total_cleaning_time := total_trees * cleaning_time_per_tree
  let cleaning_time_hours := total_cleaning_time / minutes_per_hour
  let actual_cleaning_time := cleaning_time_hours / 2
  actual_cleaning_time = 1 := by sorry

end NUMINAMATH_CALUDE_cleaning_time_proof_l3918_391888


namespace NUMINAMATH_CALUDE_cubic_difference_l3918_391897

theorem cubic_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : 
  a^3 - b^3 = 108 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l3918_391897


namespace NUMINAMATH_CALUDE_choir_average_age_l3918_391895

theorem choir_average_age (num_females : ℕ) (num_males : ℕ) 
  (avg_age_females : ℚ) (avg_age_males : ℚ) :
  num_females = 12 →
  num_males = 18 →
  avg_age_females = 28 →
  avg_age_males = 38 →
  let total_people := num_females + num_males
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  total_age / total_people = 34 := by
  sorry

end NUMINAMATH_CALUDE_choir_average_age_l3918_391895


namespace NUMINAMATH_CALUDE_correct_difference_is_1552_l3918_391882

/-- Calculates the correct difference given the erroneous calculation and mistakes made --/
def correct_difference (erroneous_difference : ℕ) 
  (units_mistake : ℕ) (tens_mistake : ℕ) (hundreds_mistake : ℕ) : ℕ :=
  erroneous_difference - hundreds_mistake + tens_mistake - units_mistake

/-- Proves that the correct difference is 1552 given the specific mistakes in the problem --/
theorem correct_difference_is_1552 : 
  correct_difference 1994 2 60 500 = 1552 := by sorry

end NUMINAMATH_CALUDE_correct_difference_is_1552_l3918_391882


namespace NUMINAMATH_CALUDE_race_probability_l3918_391851

theorem race_probability (total_cars : ℕ) (prob_x prob_y prob_z : ℚ) 
  (h_total : total_cars = 10)
  (h_x : prob_x = 1/7)
  (h_y : prob_y = 1/3)
  (h_z : prob_z = 1/5)
  (h_no_tie : ∀ a b : ℕ, a ≠ b → a ≤ total_cars → b ≤ total_cars → 
    (prob_x + prob_y + prob_z ≤ 1)) :
  prob_x + prob_y + prob_z = 71/105 := by
sorry

end NUMINAMATH_CALUDE_race_probability_l3918_391851


namespace NUMINAMATH_CALUDE_prob_no_adjacent_birch_is_2_55_l3918_391865

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of trees -/
def total_trees : ℕ := 15

/-- The number of birch trees -/
def birch_trees : ℕ := 6

/-- The number of non-birch trees -/
def non_birch_trees : ℕ := total_trees - birch_trees

/-- The probability of no two birch trees being adjacent when arranged randomly -/
def prob_no_adjacent_birch : ℚ := 
  choose (non_birch_trees + 1) birch_trees / choose total_trees birch_trees

theorem prob_no_adjacent_birch_is_2_55 : 
  prob_no_adjacent_birch = 2 / 55 := by sorry

end NUMINAMATH_CALUDE_prob_no_adjacent_birch_is_2_55_l3918_391865


namespace NUMINAMATH_CALUDE_racers_meeting_time_l3918_391857

/-- The time in seconds for the Racing Magic to complete one lap -/
def racing_magic_lap_time : ℕ := 60

/-- The number of laps the Charging Bull completes in one hour -/
def charging_bull_laps_per_hour : ℕ := 40

/-- The time in seconds for the Charging Bull to complete one lap -/
def charging_bull_lap_time : ℕ := 3600 / charging_bull_laps_per_hour

/-- The least common multiple of the two lap times -/
def lcm_lap_times : ℕ := Nat.lcm racing_magic_lap_time charging_bull_lap_time

/-- The time in minutes for the racers to meet at the starting point for the second time -/
def meeting_time_minutes : ℕ := lcm_lap_times / 60

theorem racers_meeting_time :
  meeting_time_minutes = 3 := by sorry

end NUMINAMATH_CALUDE_racers_meeting_time_l3918_391857


namespace NUMINAMATH_CALUDE_all_paths_end_at_z_l3918_391806

-- Define the graph structure
structure DirectedGraph (V : Type) where
  edge : V → V → Prop
  distinct_edge : ∀ a b : V, edge a b → a ≠ b
  at_most_one : ∀ a b : V, Unique (edge a b)

-- Define the property mentioned in the problem
def has_common_target {V : Type} (G : DirectedGraph V) (x u v w : V) : Prop :=
  x ≠ u ∧ x ≠ v ∧ u ≠ v ∧ G.edge x u ∧ G.edge x v → ∃ w, G.edge u w ∧ G.edge v w

-- Define a path in the graph
def is_path {V : Type} (G : DirectedGraph V) : List V → Prop
  | [] => True
  | [_] => True
  | (a::b::rest) => G.edge a b ∧ is_path G (b::rest)

-- Define the length of a path
def path_length {V : Type} : List V → Nat
  | [] => 0
  | [_] => 0
  | (_::rest) => 1 + path_length rest

-- The main theorem
theorem all_paths_end_at_z {V : Type} (G : DirectedGraph V) (x z : V) (n : Nat) :
  (∀ a b c w : V, has_common_target G a b c w) →
  (∃ path : List V, is_path G path ∧ path.head? = some x ∧ path.getLast? = some z ∧ path_length path = n) →
  (∀ v : V, ¬G.edge z v) →
  (∀ path : List V, is_path G path ∧ path.head? = some x → path_length path = n ∧ path.getLast? = some z) :=
by sorry

end NUMINAMATH_CALUDE_all_paths_end_at_z_l3918_391806


namespace NUMINAMATH_CALUDE_jordan_run_time_l3918_391801

/-- Given that Jordan runs 4 miles in 1/3 of the time Steve takes to run 6 miles,
    and Steve takes 36 minutes to run 6 miles, prove that Jordan would take 30 minutes to run 10 miles. -/
theorem jordan_run_time (jordan_distance : ℝ) (steve_distance : ℝ) (steve_time : ℝ) 
  (h1 : jordan_distance = 4)
  (h2 : steve_distance = 6)
  (h3 : steve_time = 36)
  (h4 : jordan_distance * steve_time = steve_distance * (steve_time / 3)) :
  (10 : ℝ) * (steve_time / jordan_distance) = 30 := by
  sorry

end NUMINAMATH_CALUDE_jordan_run_time_l3918_391801


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3918_391871

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2 - Complex.I) :
  z.im = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3918_391871


namespace NUMINAMATH_CALUDE_test_score_ratio_l3918_391881

theorem test_score_ratio (total_questions : ℕ) (score : ℕ) (correct_answers : ℕ)
  (h1 : total_questions = 100)
  (h2 : score = 79)
  (h3 : correct_answers = 93)
  (h4 : correct_answers ≤ total_questions) :
  (total_questions - correct_answers) / correct_answers = 7 / 93 := by
sorry

end NUMINAMATH_CALUDE_test_score_ratio_l3918_391881

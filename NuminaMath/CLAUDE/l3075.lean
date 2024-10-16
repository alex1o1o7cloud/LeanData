import Mathlib

namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l3075_307538

-- Define the trapezoid and its properties
structure Trapezoid :=
  (AB CD : ℝ)
  (area_ratio : ℝ)
  (sum_parallel_sides : ℝ)
  (h_positive : AB > 0)
  (h_area_ratio : area_ratio = 5 / 3)
  (h_sum : AB + CD = sum_parallel_sides)

-- Theorem statement
theorem trapezoid_segment_length (t : Trapezoid) (h : t.sum_parallel_sides = 160) :
  t.AB = 100 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l3075_307538


namespace NUMINAMATH_CALUDE_sphere_radius_from_hole_l3075_307567

/-- Given a sphere intersecting a plane, if the resulting circular hole has a diameter of 30 cm
    and a depth of 10 cm, then the radius of the sphere is 16.25 cm. -/
theorem sphere_radius_from_hole (r : ℝ) (h : r > 0) :
  (∃ x : ℝ, x > 0 ∧ x^2 + 15^2 = (x + 10)^2 ∧ r^2 = x^2 + 15^2) →
  r = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_hole_l3075_307567


namespace NUMINAMATH_CALUDE_person_b_work_days_l3075_307560

/-- Given that person A can complete a work in 30 days, and together with person B
    they complete 2/9 of the work in 4 days, prove that person B can complete
    the work alone in 45 days. -/
theorem person_b_work_days (a_days : ℕ) (combined_work : ℚ) (combined_days : ℕ) :
  a_days = 30 →
  combined_work = 2 / 9 →
  combined_days = 4 →
  ∃ b_days : ℕ,
    b_days = 45 ∧
    combined_work = combined_days * (1 / a_days + 1 / b_days) :=
by sorry

end NUMINAMATH_CALUDE_person_b_work_days_l3075_307560


namespace NUMINAMATH_CALUDE_directional_vector_of_line_l3075_307577

/-- Given a line with equation 3x + 2y - 1 = 0, prove that (2, -3) is a directional vector --/
theorem directional_vector_of_line (x y : ℝ) :
  (3 * x + 2 * y - 1 = 0) → (2 * 3 + (-3) * 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_directional_vector_of_line_l3075_307577


namespace NUMINAMATH_CALUDE_bead_necklace_problem_l3075_307533

theorem bead_necklace_problem (total_beads : Nat) (num_necklaces : Nat) (h1 : total_beads = 31) (h2 : num_necklaces = 4) :
  total_beads % num_necklaces = 3 := by
  sorry

end NUMINAMATH_CALUDE_bead_necklace_problem_l3075_307533


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3075_307530

theorem polynomial_expansion (t : ℝ) :
  (3 * t^2 - 4 * t + 3) * (-2 * t^2 + 3 * t - 4) =
  -6 * t^4 + 17 * t^3 - 30 * t^2 + 25 * t - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3075_307530


namespace NUMINAMATH_CALUDE_product_of_roots_l3075_307588

theorem product_of_roots (x₁ x₂ : ℝ) 
  (h1 : x₁^2 - 2*x₁ = 2) 
  (h2 : x₂^2 - 2*x₂ = 2) 
  (h3 : x₁ ≠ x₂) : 
  x₁ * x₂ = -2 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l3075_307588


namespace NUMINAMATH_CALUDE_chris_savings_l3075_307522

theorem chris_savings (x : ℝ) 
  (grandmother : ℝ) (aunt_uncle : ℝ) (parents : ℝ) (total : ℝ)
  (h1 : grandmother = 25)
  (h2 : aunt_uncle = 20)
  (h3 : parents = 75)
  (h4 : total = 279)
  (h5 : x + grandmother + aunt_uncle + parents = total) :
  x = 159 := by
sorry

end NUMINAMATH_CALUDE_chris_savings_l3075_307522


namespace NUMINAMATH_CALUDE_diamond_three_four_l3075_307549

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := a^2 + 2*a*b

-- Define the ◇ operation
def diamond (a b : ℝ) : ℝ := 4*a + 6*b - (oplus a b)

-- Theorem statement
theorem diamond_three_four : diamond 3 4 = 3 := by sorry

end NUMINAMATH_CALUDE_diamond_three_four_l3075_307549


namespace NUMINAMATH_CALUDE_combined_weight_is_9500_l3075_307566

def regular_dinosaur_weight : ℕ := 800
def number_of_regular_dinosaurs : ℕ := 5
def barney_extra_weight : ℕ := 1500

def combined_weight : ℕ :=
  (regular_dinosaur_weight * number_of_regular_dinosaurs) + 
  (regular_dinosaur_weight * number_of_regular_dinosaurs + barney_extra_weight)

theorem combined_weight_is_9500 : combined_weight = 9500 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_is_9500_l3075_307566


namespace NUMINAMATH_CALUDE_root_sum_power_five_l3075_307513

theorem root_sum_power_five (ζ₁ ζ₂ ζ₃ : ℂ) : 
  (ζ₁^3 - ζ₁^2 - 2*ζ₁ - 2 = 0) →
  (ζ₂^3 - ζ₂^2 - 2*ζ₂ - 2 = 0) →
  (ζ₃^3 - ζ₃^2 - 2*ζ₃ - 2 = 0) →
  (ζ₁ + ζ₂ + ζ₃ = 1) →
  (ζ₁^2 + ζ₂^2 + ζ₃^2 = 5) →
  (ζ₁^3 + ζ₂^3 + ζ₃^3 = 11) →
  (ζ₁^5 + ζ₂^5 + ζ₃^5 = 55) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_power_five_l3075_307513


namespace NUMINAMATH_CALUDE_fifth_boy_payment_is_35_l3075_307509

/-- The total cost of the video game system -/
def total_cost : ℚ := 120

/-- The amount paid by the fourth boy -/
def fourth_boy_payment : ℚ := 20

/-- The payment fractions for the first three boys -/
def first_boy_fraction : ℚ := 1/3
def second_boy_fraction : ℚ := 1/4
def third_boy_fraction : ℚ := 1/5

/-- The amounts paid by each boy -/
noncomputable def first_boy_payment (second third fourth fifth : ℚ) : ℚ :=
  first_boy_fraction * (second + third + fourth + fifth)

noncomputable def second_boy_payment (first third fourth fifth : ℚ) : ℚ :=
  second_boy_fraction * (first + third + fourth + fifth)

noncomputable def third_boy_payment (first second fourth fifth : ℚ) : ℚ :=
  third_boy_fraction * (first + second + fourth + fifth)

/-- The theorem stating that the fifth boy paid $35 -/
theorem fifth_boy_payment_is_35 :
  ∃ (first second third fifth : ℚ),
    first = first_boy_payment second third fourth_boy_payment fifth ∧
    second = second_boy_payment first third fourth_boy_payment fifth ∧
    third = third_boy_payment first second fourth_boy_payment fifth ∧
    first + second + third + fourth_boy_payment + fifth = total_cost ∧
    fifth = 35 := by
  sorry

end NUMINAMATH_CALUDE_fifth_boy_payment_is_35_l3075_307509


namespace NUMINAMATH_CALUDE_perfectville_run_difference_l3075_307539

theorem perfectville_run_difference (street_width : ℕ) (block_side : ℕ) : 
  street_width = 30 → block_side = 500 → 
  4 * (block_side + 2 * street_width) - 4 * block_side = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_perfectville_run_difference_l3075_307539


namespace NUMINAMATH_CALUDE_tan_difference_inequality_l3075_307563

theorem tan_difference_inequality (x y n : ℝ) (hn : n > 0) (h : Real.tan x = n * Real.tan y) :
  Real.tan (x - y) ^ 2 ≤ (n - 1) ^ 2 / (4 * n) := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_inequality_l3075_307563


namespace NUMINAMATH_CALUDE_inverse_36_mod_53_l3075_307581

theorem inverse_36_mod_53 (h : (17⁻¹ : ZMod 53) = 26) : (36⁻¹ : ZMod 53) = 27 := by
  sorry

end NUMINAMATH_CALUDE_inverse_36_mod_53_l3075_307581


namespace NUMINAMATH_CALUDE_card_game_cost_l3075_307529

theorem card_game_cost (rare_count : ℕ) (uncommon_count : ℕ) (common_count : ℕ)
                       (rare_cost : ℚ) (uncommon_cost : ℚ) (total_cost : ℚ) :
  rare_count = 19 →
  uncommon_count = 11 →
  common_count = 30 →
  rare_cost = 1 →
  uncommon_cost = (1/2) →
  total_cost = 32 →
  (total_cost - (rare_count * rare_cost + uncommon_count * uncommon_cost)) / common_count = (1/4) := by
sorry

end NUMINAMATH_CALUDE_card_game_cost_l3075_307529


namespace NUMINAMATH_CALUDE_point_in_inequality_region_implies_B_range_l3075_307521

/-- Given a point A (1, 2) inside the plane region corresponding to the linear inequality 2x - By + 3 ≥ 0, 
    prove that the range of the real number B is B ≤ 2.5. -/
theorem point_in_inequality_region_implies_B_range (B : ℝ) : 
  (2 * 1 - B * 2 + 3 ≥ 0) → B ≤ 2.5 := by
  sorry

end NUMINAMATH_CALUDE_point_in_inequality_region_implies_B_range_l3075_307521


namespace NUMINAMATH_CALUDE_expression_simplification_l3075_307597

theorem expression_simplification (x : ℝ) : 
  ((3 * x + 6) - 5 * x) / 3 = -(2/3) * x + 2 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3075_307597


namespace NUMINAMATH_CALUDE_largest_two_digit_divisible_by_six_ending_in_four_l3075_307599

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_four (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_six_ending_in_four :
  ∃ (max : ℕ), 
    is_two_digit max ∧ 
    max % 6 = 0 ∧ 
    ends_in_four max ∧
    ∀ (n : ℕ), is_two_digit n → n % 6 = 0 → ends_in_four n → n ≤ max :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_two_digit_divisible_by_six_ending_in_four_l3075_307599


namespace NUMINAMATH_CALUDE_hall_width_proof_l3075_307512

/-- Given a rectangular hall with specified dimensions and cost constraints, 
    prove that the width of the hall is 17 meters. -/
theorem hall_width_proof (length height : ℝ) (cost_per_sqm total_cost : ℝ) :
  length = 20 →
  height = 5 →
  cost_per_sqm = 60 →
  total_cost = 57000 →
  ∃ w : ℝ, (2 * length * w + 2 * length * height + 2 * w * height) * cost_per_sqm = total_cost ∧ w = 17 :=
by sorry

end NUMINAMATH_CALUDE_hall_width_proof_l3075_307512


namespace NUMINAMATH_CALUDE_skyscraper_anniversary_l3075_307535

theorem skyscraper_anniversary (years_since_built : ℕ) (years_to_anniversary : ℕ) (years_before_anniversary : ℕ) : 
  years_since_built = 100 →
  years_to_anniversary = 200 →
  years_before_anniversary = 5 →
  years_to_anniversary - years_before_anniversary - years_since_built = 95 :=
by sorry

end NUMINAMATH_CALUDE_skyscraper_anniversary_l3075_307535


namespace NUMINAMATH_CALUDE_tims_doctor_visit_cost_l3075_307543

theorem tims_doctor_visit_cost (tim_total_payment : ℝ) (cat_visit_cost : ℝ) (cat_insurance_coverage : ℝ) (tim_insurance_coverage_percent : ℝ) : 
  tim_total_payment = 135 →
  cat_visit_cost = 120 →
  cat_insurance_coverage = 60 →
  tim_insurance_coverage_percent = 75 →
  ∃ (doctor_visit_cost : ℝ),
    doctor_visit_cost = 300 ∧
    tim_total_payment = (1 - tim_insurance_coverage_percent / 100) * doctor_visit_cost + (cat_visit_cost - cat_insurance_coverage) :=
by sorry

end NUMINAMATH_CALUDE_tims_doctor_visit_cost_l3075_307543


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l3075_307594

/-- The length of the path traversed by vertex C of an equilateral triangle rotating inside a square -/
theorem triangle_rotation_path_length :
  ∀ (triangle_side square_side : ℝ),
  triangle_side = 3 →
  square_side = 6 →
  ∃ (path_length : ℝ),
  path_length = 18 * Real.pi ∧
  path_length = 12 * (triangle_side * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l3075_307594


namespace NUMINAMATH_CALUDE_vector_BC_proof_l3075_307565

-- Define the points and vectors
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (2, 1)
def AC : ℝ × ℝ := (-3, -2)

-- State the theorem
theorem vector_BC_proof : 
  let BC : ℝ × ℝ := (AC.1 - (B.1 - A.1), AC.2 - (B.2 - A.2))
  BC = (-5, -2) := by sorry

end NUMINAMATH_CALUDE_vector_BC_proof_l3075_307565


namespace NUMINAMATH_CALUDE_complement_B_intersect_A_m_value_for_intersection_l3075_307593

-- Define set A
def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}

-- Define set B with parameter m
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem 1
theorem complement_B_intersect_A :
  (Set.compl (B 3) ∩ A) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem m_value_for_intersection :
  ∃ m : ℝ, (A ∩ B m) = {x | -1 < x ∧ x < 4} → m = 8 := by sorry

end NUMINAMATH_CALUDE_complement_B_intersect_A_m_value_for_intersection_l3075_307593


namespace NUMINAMATH_CALUDE_lesser_fraction_l3075_307510

theorem lesser_fraction (x y : ℚ) : 
  x + y = 8/9 → x * y = 1/8 → min x y = 7/40 := by
  sorry

end NUMINAMATH_CALUDE_lesser_fraction_l3075_307510


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3075_307558

theorem solution_set_equivalence (x y : ℝ) :
  (x^2 + 3*x*y + 2*y^2) * (x^2*y^2 - 1) = 0 ↔
  y = -x/2 ∨ y = -x ∨ y = -1/x ∨ y = 1/x :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3075_307558


namespace NUMINAMATH_CALUDE_city_distance_l3075_307547

def is_valid_distance (S : ℕ) : Prop :=
  ∀ x : ℕ, x ≤ S → (Nat.gcd x (S - x) = 1 ∨ Nat.gcd x (S - x) = 3 ∨ Nat.gcd x (S - x) = 13)

theorem city_distance : 
  (∃ S : ℕ, is_valid_distance S ∧ ∀ T : ℕ, T < S → ¬is_valid_distance T) ∧
  (∀ S : ℕ, (is_valid_distance S ∧ ∀ T : ℕ, T < S → ¬is_valid_distance T) → S = 39) :=
sorry

end NUMINAMATH_CALUDE_city_distance_l3075_307547


namespace NUMINAMATH_CALUDE_parabola_directrix_l3075_307524

/-- The equation of the directrix of a parabola with equation y = -4x² is y = 1/16 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = -4 * x^2) → (∃ k : ℝ, y = k ∧ k = 1/16) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3075_307524


namespace NUMINAMATH_CALUDE_david_presents_l3075_307561

/-- The total number of presents David received -/
def total_presents (christmas_presents : ℕ) (birthday_presents : ℕ) : ℕ :=
  christmas_presents + birthday_presents

/-- Theorem: Given the conditions, David received 90 presents in total -/
theorem david_presents : 
  ∀ (christmas_presents birthday_presents : ℕ),
  christmas_presents = 60 →
  christmas_presents = 2 * birthday_presents →
  total_presents christmas_presents birthday_presents = 90 := by
  sorry

end NUMINAMATH_CALUDE_david_presents_l3075_307561


namespace NUMINAMATH_CALUDE_base_10_144_equals_base_12_100_l3075_307514

def base_10_to_12 (n : ℕ) : List ℕ := sorry

theorem base_10_144_equals_base_12_100 :
  base_10_to_12 144 = [1, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_base_10_144_equals_base_12_100_l3075_307514


namespace NUMINAMATH_CALUDE_division_problem_l3075_307544

theorem division_problem (divisor quotient remainder number : ℕ) : 
  divisor = 12 → 
  quotient = 9 → 
  remainder = 1 → 
  number = divisor * quotient + remainder → 
  number = 109 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3075_307544


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3075_307516

def base_conversion (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + base * acc) 0

theorem smallest_dual_base_representation :
  ∃ (a b : Nat), a > 2 ∧ b > 2 ∧
  base_conversion [2, 1] a = 7 ∧
  base_conversion [1, 2] b = 7 ∧
  (∀ (x y : Nat), x > 2 → y > 2 →
    base_conversion [2, 1] x = base_conversion [1, 2] y →
    base_conversion [2, 1] x ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3075_307516


namespace NUMINAMATH_CALUDE_inverse_of_A_squared_l3075_307519

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = ![![3, -2], ![1, 1]]) : 
  (A^2)⁻¹ = ![![7, -8], ![4, -1]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_squared_l3075_307519


namespace NUMINAMATH_CALUDE_folded_rectangle_perimeter_l3075_307569

/-- Given a rectangle with length 20 cm and width 12 cm, when folded along its diagonal,
    the perimeter of the resulting shaded region is 64 cm. -/
theorem folded_rectangle_perimeter :
  ∀ (length width : ℝ),
    length = 20 →
    width = 12 →
    let perimeter := (length + width) * 2
    perimeter = 64 := by
  sorry

end NUMINAMATH_CALUDE_folded_rectangle_perimeter_l3075_307569


namespace NUMINAMATH_CALUDE_john_years_taking_pictures_l3075_307564

/-- Calculates the number of years John has been taking pictures given the following conditions:
  * John takes 10 pictures every day
  * Each memory card can store 50 images
  * Each memory card costs $60
  * John spent $13,140 on memory cards
-/
def years_taking_pictures (
  pictures_per_day : ℕ)
  (images_per_card : ℕ)
  (card_cost : ℕ)
  (total_spent : ℕ)
  : ℕ :=
  let cards_bought := total_spent / card_cost
  let total_images := cards_bought * images_per_card
  let days_taking_pictures := total_images / pictures_per_day
  days_taking_pictures / 365

theorem john_years_taking_pictures :
  years_taking_pictures 10 50 60 13140 = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_years_taking_pictures_l3075_307564


namespace NUMINAMATH_CALUDE_badge_exchange_l3075_307527

theorem badge_exchange (x : ℕ) : 
  (x + 5 - (6 * (x + 5)) / 25 + x / 5 = x - x / 5 + (6 * (x + 5)) / 25 - 1) → 
  (x = 45 ∧ x + 5 = 50) := by
  sorry

end NUMINAMATH_CALUDE_badge_exchange_l3075_307527


namespace NUMINAMATH_CALUDE_systematic_sampling_l3075_307505

theorem systematic_sampling 
  (total_students : Nat) 
  (num_segments : Nat) 
  (segment_size : Nat) 
  (sixteenth_segment_num : Nat) :
  total_students = 160 →
  num_segments = 20 →
  segment_size = 8 →
  sixteenth_segment_num = 125 →
  ∃ (first_segment_num : Nat),
    first_segment_num = 5 ∧
    sixteenth_segment_num = first_segment_num + segment_size * (16 - 1) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l3075_307505


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_min_value_of_expression_l3075_307583

/-- Given a > 0, b > 0, and the minimum value of |x+a| + |x-b| is 4, then a + b = 4 -/
theorem sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_min : ∀ x, |x + a| + |x - b| ≥ 4) : a + b = 4 := by sorry

/-- Given a + b = 4, the minimum value of (1/4)a² + (1/9)b² is 16/13 -/
theorem min_value_of_expression (a b : ℝ) (h : a + b = 4) :
  ∀ x y, x > 0 → y > 0 → x + y = 4 → (1/4) * a^2 + (1/9) * b^2 ≤ (1/4) * x^2 + (1/9) * y^2 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_min_value_of_expression_l3075_307583


namespace NUMINAMATH_CALUDE_multiply_704_12_by_3_10_l3075_307554

-- Define a function to convert from base 12 to base 10
def base12ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 12
def base10ToBase12 (n : ℕ) : ℕ := sorry

-- Define the given number in base 12
def given_number : ℕ := 704

-- Define the multiplier in base 10
def multiplier : ℕ := 3

-- Theorem statement
theorem multiply_704_12_by_3_10 : 
  base10ToBase12 (base12ToBase10 given_number * multiplier) = 1910 := by
  sorry

end NUMINAMATH_CALUDE_multiply_704_12_by_3_10_l3075_307554


namespace NUMINAMATH_CALUDE_smallest_solution_l3075_307592

-- Define the equation
def equation (t : ℝ) : Prop :=
  (16 * t^3 - 49 * t^2 + 35 * t - 6) / (4 * t - 3) + 7 * t = 8 * t - 2

-- Define the set of all t that satisfy the equation
def solution_set : Set ℝ := {t | equation t}

-- Theorem statement
theorem smallest_solution :
  ∃ (t_min : ℝ), t_min ∈ solution_set ∧ t_min = 3/4 ∧ ∀ (t : ℝ), t ∈ solution_set → t_min ≤ t :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_l3075_307592


namespace NUMINAMATH_CALUDE_analogical_reasoning_example_l3075_307526

/-- Represents different types of reasoning -/
inductive ReasoningType
  | Deductive
  | Inductive
  | Analogical
  | Other

/-- Determines the type of reasoning for a given statement -/
def determineReasoningType (statement : String) : ReasoningType :=
  match statement with
  | "Inferring the properties of a spatial quadrilateral from the properties of a plane triangle" => ReasoningType.Analogical
  | _ => ReasoningType.Other

/-- Theorem stating that the given statement is an example of analogical reasoning -/
theorem analogical_reasoning_example :
  determineReasoningType "Inferring the properties of a spatial quadrilateral from the properties of a plane triangle" = ReasoningType.Analogical := by
  sorry


end NUMINAMATH_CALUDE_analogical_reasoning_example_l3075_307526


namespace NUMINAMATH_CALUDE_garden_perimeter_l3075_307571

/-- The perimeter of a rectangular garden with width 8 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is equal to 64 meters. -/
theorem garden_perimeter : 
  ∀ (garden_length : ℝ),
  garden_length > 0 →
  8 * garden_length = 16 * 12 →
  2 * (garden_length + 8) = 64 := by
sorry

end NUMINAMATH_CALUDE_garden_perimeter_l3075_307571


namespace NUMINAMATH_CALUDE_parabola_equation_l3075_307532

-- Define a parabola passing through a point
def parabola_through_point (x y : ℝ) : Prop :=
  (y^2 = x) ∨ (x^2 = -8*y)

-- Theorem statement
theorem parabola_equation : parabola_through_point 4 (-2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3075_307532


namespace NUMINAMATH_CALUDE_exists_unobserved_planet_l3075_307587

/-- Represents a planet in the system -/
structure Planet where
  id : Nat

/-- Represents the system of planets -/
structure PlanetSystem where
  planets : Finset Planet
  distance : Planet → Planet → ℝ
  nearest_neighbor : Planet → Planet
  num_planets_odd : Odd (Finset.card planets)
  different_distances : ∀ p q r s : Planet, p ≠ q → r ≠ s → (p, q) ≠ (r, s) → distance p q ≠ distance r s
  nearest_is_nearest : ∀ p q : Planet, p ≠ q → distance p (nearest_neighbor p) ≤ distance p q

/-- The main theorem: In a system with an odd number of planets, where each planet has an astronomer
    observing the nearest planet and all inter-planet distances are unique, there exists at least
    one planet that is not being observed. -/
theorem exists_unobserved_planet (sys : PlanetSystem) :
  ∃ p : Planet, p ∈ sys.planets ∧ ∀ q : Planet, q ∈ sys.planets → sys.nearest_neighbor q ≠ p :=
sorry

end NUMINAMATH_CALUDE_exists_unobserved_planet_l3075_307587


namespace NUMINAMATH_CALUDE_intersection_points_line_slope_l3075_307568

theorem intersection_points_line_slope :
  ∀ (s : ℝ) (x y : ℝ),
    (2 * x - 3 * y = 4 * s + 6) →
    (2 * x + y = 3 * s + 1) →
    y = -2/13 * x - 14/13 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_line_slope_l3075_307568


namespace NUMINAMATH_CALUDE_largest_common_remainder_l3075_307542

theorem largest_common_remainder :
  ∃ (n : ℕ) (r : ℕ),
    2013 ≤ n ∧ n ≤ 2156 ∧
    n % 5 = r ∧ n % 11 = r ∧ n % 13 = r ∧
    ∀ (m : ℕ),
      (2013 ≤ m ∧ m ≤ 2156 ∧ 
       ∃ (s : ℕ), m % 5 = s ∧ m % 11 = s ∧ m % 13 = s) →
      s ≤ r ∧
    r = 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_remainder_l3075_307542


namespace NUMINAMATH_CALUDE_smallest_stairs_solution_l3075_307528

theorem smallest_stairs_solution (n : ℕ) : 
  (n > 20 ∧ n % 6 = 5 ∧ n % 7 = 4) → n ≥ 53 :=
by sorry

end NUMINAMATH_CALUDE_smallest_stairs_solution_l3075_307528


namespace NUMINAMATH_CALUDE_certain_number_is_three_l3075_307545

theorem certain_number_is_three (n : ℝ) (x : ℤ) (h1 : n^(2*x) = 3^(12-x)) (h2 : x = 4) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_three_l3075_307545


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l3075_307548

theorem system_of_equations_solutions :
  -- First system
  (∃ x y : ℝ, y = 3*x ∧ 7*x - 2*y = 2 → x = 2 ∧ y = 6) ∧
  -- Second system
  (∃ x y : ℝ, 2*x + 5*y = -4 ∧ 5*x + 2*y = 11 → x = 3 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l3075_307548


namespace NUMINAMATH_CALUDE_triangle_area_l3075_307557

theorem triangle_area (a b c : ℝ) (h1 : a = 18) (h2 : b = 80) (h3 : c = 82) :
  (1/2) * a * b = 720 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3075_307557


namespace NUMINAMATH_CALUDE_man_mass_on_boat_l3075_307517

/-- The mass of a man who causes a boat to sink by a certain depth in water. -/
def mass_of_man (boat_length boat_breadth sinking_depth : ℝ) : ℝ :=
  boat_length * boat_breadth * sinking_depth * 1000

theorem man_mass_on_boat :
  mass_of_man 3 2 0.018 = 108 := by
  sorry

end NUMINAMATH_CALUDE_man_mass_on_boat_l3075_307517


namespace NUMINAMATH_CALUDE_complement_of_intersection_l3075_307556

def U : Set Nat := {1, 2, 3, 4}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {2, 3, 4}

theorem complement_of_intersection (U M N : Set Nat) 
  (hU : U = {1, 2, 3, 4}) 
  (hM : M = {1, 2, 3}) 
  (hN : N = {2, 3, 4}) : 
  (U \ (M ∩ N)) = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l3075_307556


namespace NUMINAMATH_CALUDE_badminton_players_l3075_307598

theorem badminton_players (total : ℕ) (tennis : ℕ) (neither : ℕ) (both : ℕ) : 
  total = 30 → tennis = 19 → neither = 2 → both = 7 → 
  ∃ badminton : ℕ, badminton = 16 ∧ total = badminton + tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_badminton_players_l3075_307598


namespace NUMINAMATH_CALUDE_min_value_theorem_l3075_307534

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2) :
  x^2 / (2*y) + 4*y^2 / x ≥ 2 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2*y = 2 ∧ x^2 / (2*y) + 4*y^2 / x = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3075_307534


namespace NUMINAMATH_CALUDE_dave_animal_books_l3075_307584

/-- The number of books about animals Dave bought -/
def num_animal_books : ℕ := sorry

/-- The number of books about outer space Dave bought -/
def num_space_books : ℕ := 6

/-- The number of books about trains Dave bought -/
def num_train_books : ℕ := 3

/-- The cost of each book in dollars -/
def cost_per_book : ℕ := 6

/-- The total amount Dave spent on books in dollars -/
def total_spent : ℕ := 102

theorem dave_animal_books : 
  num_animal_books = 8 ∧
  num_animal_books * cost_per_book + num_space_books * cost_per_book + num_train_books * cost_per_book = total_spent :=
sorry

end NUMINAMATH_CALUDE_dave_animal_books_l3075_307584


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3075_307523

theorem complex_equation_solution (m : ℝ) : 
  (m - 1 : ℂ) + 2*m*Complex.I = 1 + 4*Complex.I → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3075_307523


namespace NUMINAMATH_CALUDE_min_production_volume_correct_l3075_307570

/-- The minimum production volume to avoid a loss -/
def min_production_volume : ℕ := 150

/-- The total cost function -/
def total_cost (x : ℕ) : ℝ := 3000 + 20 * x - 0.1 * x^2

/-- The selling price per unit -/
def selling_price : ℝ := 25

/-- Theorem stating the minimum production volume to avoid a loss -/
theorem min_production_volume_correct :
  ∀ x : ℕ, 0 < x → x < 240 →
  (selling_price * x ≥ total_cost x ↔ x ≥ min_production_volume) := by
  sorry

end NUMINAMATH_CALUDE_min_production_volume_correct_l3075_307570


namespace NUMINAMATH_CALUDE_train_crossing_time_l3075_307573

/-- Proves that a train with given length and speed takes a specific time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 160 ∧ 
  train_speed_kmh = 72 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 8 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3075_307573


namespace NUMINAMATH_CALUDE_multiples_of_3_or_5_not_6_up_to_200_l3075_307502

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n / m : ℕ)

def count_multiples_of_3_or_5_not_6 (upper_bound : ℕ) : ℕ :=
  count_multiples upper_bound 3 +
  count_multiples upper_bound 5 -
  count_multiples upper_bound 15 -
  count_multiples upper_bound 6

theorem multiples_of_3_or_5_not_6_up_to_200 :
  count_multiples_of_3_or_5_not_6 200 = 60 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_3_or_5_not_6_up_to_200_l3075_307502


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3075_307578

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the first inequality
def S := Set.Ioo 1 2

-- Define the second inequality
def g (a b c x : ℝ) := a - c * (x^2 - x - 1) - b * x

-- Define the solution set of the second inequality
def T := {x : ℝ | x ≤ -3/2 ∨ x ≥ 1}

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) (h : ∀ x, x ∈ S ↔ f a b c x > 0) :
  ∀ x, x ∈ T ↔ g a b c x ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3075_307578


namespace NUMINAMATH_CALUDE_unique_x_value_l3075_307503

theorem unique_x_value : ∃! x : ℤ, 
  1 < x ∧ x < 9 ∧ 
  2 < x ∧ x < 15 ∧ 
  -1 < x ∧ x < 7 ∧ 
  0 < x ∧ x < 4 ∧ 
  x + 1 < 5 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_x_value_l3075_307503


namespace NUMINAMATH_CALUDE_exists_arithmetic_not_m_sequence_l3075_307582

/-- Definition of "M sequence" -/
def is_m_sequence (b : ℕ → ℝ) (c : ℕ → ℝ) : Prop :=
  (∀ n, b n < b (n + 1)) ∧ 
  (∀ n, c n < c (n + 1)) ∧
  (∀ n, ∃ m, c n ≤ b m ∧ b m ≤ c (n + 1))

/-- Arithmetic sequence -/
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) - a n = d

/-- Partial sum sequence -/
def partial_sum (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => partial_sum a n + a (n + 1)

/-- Main theorem -/
theorem exists_arithmetic_not_m_sequence :
  ∃ a : ℕ → ℝ, is_arithmetic a ∧ ¬(is_m_sequence a (partial_sum a)) := by
  sorry

end NUMINAMATH_CALUDE_exists_arithmetic_not_m_sequence_l3075_307582


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l3075_307504

def euler_family_ages : List ℕ := [8, 8, 12, 10, 10, 16]

theorem euler_family_mean_age :
  (euler_family_ages.sum : ℚ) / euler_family_ages.length = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l3075_307504


namespace NUMINAMATH_CALUDE_exists_solution_l3075_307590

theorem exists_solution : ∃ (a b : ℝ), a * b = a^2 - a * b + b^2 ∧ a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_l3075_307590


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3075_307501

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x, -3 ≤ x^2 - 2*a*x + a ∧ x^2 - 2*a*x + a ≤ -2) → (a = 2 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3075_307501


namespace NUMINAMATH_CALUDE_floor_sum_example_l3075_307576

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l3075_307576


namespace NUMINAMATH_CALUDE_spade_calculation_l3075_307515

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_calculation : spade (spade 3 5) (spade 6 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l3075_307515


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3075_307536

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a x : V) :
  3 • (a + x) = x → x = -(3/2 : ℝ) • a := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3075_307536


namespace NUMINAMATH_CALUDE_problem_statement_l3075_307562

theorem problem_statement (x n : ℕ) : 
  x = 5^n - 1 →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2^(Nat.log2 x) * p * q * 11) →
  x = 3124 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3075_307562


namespace NUMINAMATH_CALUDE_games_not_working_l3075_307520

theorem games_not_working (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (good_games : ℕ) :
  games_from_friend = 2 →
  games_from_garage_sale = 2 →
  good_games = 2 →
  games_from_friend + games_from_garage_sale - good_games = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_games_not_working_l3075_307520


namespace NUMINAMATH_CALUDE_expression_value_l3075_307500

theorem expression_value (a b : ℝ) 
  (h1 : 10 * a^2 - 3 * b^2 + 5 * a * b = 0) 
  (h2 : 9 * a^2 - b^2 ≠ 0) : 
  (2 * a - b) / (3 * a - b) + (5 * b - a) / (3 * a + b) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3075_307500


namespace NUMINAMATH_CALUDE_paths_in_7x7_grid_l3075_307518

/-- The number of paths in a square grid from bottom left to top right -/
def num_paths (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- The theorem stating that the number of paths in a 7x7 grid is 3432 -/
theorem paths_in_7x7_grid : num_paths 7 = 3432 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_7x7_grid_l3075_307518


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l3075_307596

/-- 
Given a quadratic equation kx^2 - 2x - 1 = 0, this theorem states that
for the equation to have two distinct real roots, k must be greater than -1
and not equal to 0.
-/
theorem quadratic_distinct_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2*x - 1 = 0 ∧ k * y^2 - 2*y - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_condition_l3075_307596


namespace NUMINAMATH_CALUDE_ceiling_negative_fraction_squared_l3075_307589

theorem ceiling_negative_fraction_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_fraction_squared_l3075_307589


namespace NUMINAMATH_CALUDE_saving_time_proof_l3075_307550

def down_payment : ℝ := 108000
def monthly_savings : ℝ := 3000
def months_in_year : ℝ := 12

theorem saving_time_proof : 
  (down_payment / monthly_savings) / months_in_year = 3 := by
sorry

end NUMINAMATH_CALUDE_saving_time_proof_l3075_307550


namespace NUMINAMATH_CALUDE_common_divisors_9240_13860_l3075_307595

/-- The number of positive divisors that two natural numbers have in common -/
def common_divisors_count (a b : ℕ) : ℕ := (Nat.divisors (Nat.gcd a b)).card

/-- Theorem stating that 9240 and 13860 have 48 positive divisors in common -/
theorem common_divisors_9240_13860 :
  common_divisors_count 9240 13860 = 48 := by sorry

end NUMINAMATH_CALUDE_common_divisors_9240_13860_l3075_307595


namespace NUMINAMATH_CALUDE_simplify_expression_l3075_307541

theorem simplify_expression (x : ℝ) : (3 * x + 25) + (200 * x - 50) = 203 * x - 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3075_307541


namespace NUMINAMATH_CALUDE_field_trip_students_l3075_307555

theorem field_trip_students (num_vans : ℕ) (num_minibusses : ℕ) (students_per_van : ℕ) (students_per_minibus : ℕ)
  (h1 : num_vans = 6)
  (h2 : num_minibusses = 4)
  (h3 : students_per_van = 10)
  (h4 : students_per_minibus = 24) :
  num_vans * students_per_van + num_minibusses * students_per_minibus = 156 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_students_l3075_307555


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3075_307508

theorem chess_tournament_games (n : ℕ) (h : n = 10) : 
  n * (n - 1) = 90 → 2 * (n * (n - 1)) = 180 := by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l3075_307508


namespace NUMINAMATH_CALUDE_complex_product_simplification_l3075_307546

/-- Given real non-zero numbers a and b, and the imaginary unit i,
    prove that (ax+biy)(ax-biy) = a^2x^2 + b^2y^2 -/
theorem complex_product_simplification
  (a b x y : ℝ) (i : ℂ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hi : i^2 = -1)
  : (a*x + b*i*y) * (a*x - b*i*y) = a^2 * x^2 + b^2 * y^2 :=
by sorry

end NUMINAMATH_CALUDE_complex_product_simplification_l3075_307546


namespace NUMINAMATH_CALUDE_positive_intervals_l3075_307586

-- Define the expression
def f (x : ℝ) : ℝ := (x + 1) * (x - 1) * (x - 2)

-- State the theorem
theorem positive_intervals (x : ℝ) : f x > 0 ↔ x ∈ Set.Ioo (-1) 1 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_positive_intervals_l3075_307586


namespace NUMINAMATH_CALUDE_tennis_percentage_is_31_percent_l3075_307559

/-- The percentage of students who prefer tennis in both schools combined -/
def combined_tennis_percentage (north_total : ℕ) (south_total : ℕ) 
  (north_tennis_percent : ℚ) (south_tennis_percent : ℚ) : ℚ :=
  let north_tennis := (north_total : ℚ) * north_tennis_percent
  let south_tennis := (south_total : ℚ) * south_tennis_percent
  let total_tennis := north_tennis + south_tennis
  let total_students := (north_total + south_total : ℚ)
  total_tennis / total_students

/-- Theorem stating that the percentage of students who prefer tennis in both schools combined is 31% -/
theorem tennis_percentage_is_31_percent :
  combined_tennis_percentage 1800 2700 (25/100) (35/100) = 31/100 := by
  sorry

end NUMINAMATH_CALUDE_tennis_percentage_is_31_percent_l3075_307559


namespace NUMINAMATH_CALUDE_solution_check_l3075_307580

-- Define the equation
def equation (x y : ℚ) : Prop := x - 2 * y = 1

-- Define the sets of values
def setA : ℚ × ℚ := (0, -1/2)
def setB : ℚ × ℚ := (1, 1)
def setC : ℚ × ℚ := (1, 0)
def setD : ℚ × ℚ := (-1, -1)

-- Theorem stating that setB is not a solution while others are
theorem solution_check :
  ¬(equation setB.1 setB.2) ∧
  (equation setA.1 setA.2) ∧
  (equation setC.1 setC.2) ∧
  (equation setD.1 setD.2) :=
sorry

end NUMINAMATH_CALUDE_solution_check_l3075_307580


namespace NUMINAMATH_CALUDE_min_sum_of_product_l3075_307552

theorem min_sum_of_product (a b c : ℕ+) (h : a * b * c = 3432) :
  ∃ (x y z : ℕ+), x * y * z = 3432 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 56 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l3075_307552


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3075_307506

/-- Given a rectangle with length 8 and diagonal 17, its perimeter is 46 -/
theorem rectangle_perimeter (length width diagonal : ℝ) : 
  length = 8 → 
  diagonal = 17 → 
  length^2 + width^2 = diagonal^2 → 
  2 * (length + width) = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3075_307506


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3075_307553

theorem quadratic_one_solution (n : ℝ) : 
  (n > 0 ∧ ∃! x, 4 * x^2 + n * x + 1 = 0) ↔ n = 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3075_307553


namespace NUMINAMATH_CALUDE_birds_in_pet_shop_l3075_307511

/-- The number of birds in a pet shop -/
def number_of_birds (total animals : ℕ) (kittens hamsters : ℕ) : ℕ :=
  total - kittens - hamsters

/-- Theorem: There are 30 birds in the pet shop -/
theorem birds_in_pet_shop :
  let total := 77
  let kittens := 32
  let hamsters := 15
  number_of_birds total kittens hamsters = 30 := by
sorry

end NUMINAMATH_CALUDE_birds_in_pet_shop_l3075_307511


namespace NUMINAMATH_CALUDE_total_albums_l3075_307507

/-- The number of albums each person has -/
structure Albums where
  adele : ℕ
  bridget : ℕ
  katrina : ℕ
  miriam : ℕ

/-- The conditions of the problem -/
def album_conditions (a : Albums) : Prop :=
  a.adele = 30 ∧
  a.bridget = a.adele - 15 ∧
  a.katrina = 6 * a.bridget ∧
  a.miriam = 5 * a.katrina

/-- The theorem to prove -/
theorem total_albums (a : Albums) (h : album_conditions a) : 
  a.adele + a.bridget + a.katrina + a.miriam = 585 := by
  sorry

#check total_albums

end NUMINAMATH_CALUDE_total_albums_l3075_307507


namespace NUMINAMATH_CALUDE_externally_tangent_circle_radius_l3075_307585

/-- The radius of a circle externally tangent to three circles in a right triangle -/
theorem externally_tangent_circle_radius (A B C : ℝ × ℝ) (h_right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0) 
  (h_AB : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 3)
  (h_AC : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 6)
  (r_A : ℝ) (r_B : ℝ) (r_C : ℝ)
  (h_r_A : r_A = 1) (h_r_B : r_B = 2) (h_r_C : r_C = 3) :
  ∃ R : ℝ, R = (8 * Real.sqrt 11 - 19) / 7 ∧
    ∀ O : ℝ × ℝ, (Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2) = R + r_A) ∧
                 (Real.sqrt ((O.1 - B.1)^2 + (O.2 - B.2)^2) = R + r_B) ∧
                 (Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2) = R + r_C) :=
by sorry

end NUMINAMATH_CALUDE_externally_tangent_circle_radius_l3075_307585


namespace NUMINAMATH_CALUDE_triangle_properties_l3075_307540

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (BD : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / 2 = b / 3 ∧ b / 3 = c / 4 →
  BD = Real.sqrt 31 →
  BD * 2 = c →
  Real.tan C = -Real.sqrt 15 ∧
  (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3075_307540


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3075_307525

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Checks if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (given_line : Line) 
  (point : ℝ × ℝ) 
  (parallel_line : Line) :
  given_line.a = 1 ∧ 
  given_line.b = -2 ∧ 
  given_line.c = 0 ∧
  point = (1, 6) ∧
  parallel_line.a = 1 ∧
  parallel_line.b = -2 ∧
  parallel_line.c = 11 →
  parallel_line.contains point.1 point.2 ∧
  Line.parallel given_line parallel_line := by
  sorry

#check parallel_line_through_point

end NUMINAMATH_CALUDE_parallel_line_through_point_l3075_307525


namespace NUMINAMATH_CALUDE_mark_parking_tickets_l3075_307574

theorem mark_parking_tickets (total_tickets : ℕ) (sarah_speeding : ℕ)
  (h1 : total_tickets = 24)
  (h2 : sarah_speeding = 6) :
  ∃ (mark_parking sarah_parking : ℕ),
    mark_parking = 2 * sarah_parking ∧
    total_tickets = sarah_parking + mark_parking + 2 * sarah_speeding ∧
    mark_parking = 8 := by
  sorry

end NUMINAMATH_CALUDE_mark_parking_tickets_l3075_307574


namespace NUMINAMATH_CALUDE_f_neither_even_nor_odd_l3075_307531

-- Define the function f(x) = x^2 on the domain -1 < x ≤ 1
def f (x : ℝ) : ℝ := x^2

-- Define the domain of the function
def domain (x : ℝ) : Prop := -1 < x ∧ x ≤ 1

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, domain x → domain (-x) → f (-x) = f x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, domain x → domain (-x) → f (-x) = -f x

-- Theorem stating that f is neither even nor odd
theorem f_neither_even_nor_odd :
  ¬(is_even f) ∧ ¬(is_odd f) :=
sorry

end NUMINAMATH_CALUDE_f_neither_even_nor_odd_l3075_307531


namespace NUMINAMATH_CALUDE_solution_is_correct_l3075_307575

/-- The imaginary unit i such that i^2 = -1 -/
noncomputable def i : ℂ := Complex.I

/-- The equation to be solved -/
def equation (z : ℂ) : Prop := 2 * z + (5 - 3 * i) = 6 + 11 * i

/-- The theorem stating that 1/2 + 7i is the solution to the equation -/
theorem solution_is_correct : equation (1/2 + 7 * i) := by
  sorry

end NUMINAMATH_CALUDE_solution_is_correct_l3075_307575


namespace NUMINAMATH_CALUDE_tan_equality_periodic_l3075_307591

theorem tan_equality_periodic (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) → n = -30 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_periodic_l3075_307591


namespace NUMINAMATH_CALUDE_cycling_distance_is_four_point_five_l3075_307572

/-- Represents the cycling scenario with given conditions -/
structure CyclingScenario where
  speed : ℝ  -- Original speed in miles per hour
  time : ℝ   -- Original time taken in hours
  distance : ℝ -- Distance cycled in miles

/-- The conditions of the cycling problem -/
def cycling_conditions (scenario : CyclingScenario) : Prop :=
  -- Distance is speed multiplied by time
  scenario.distance = scenario.speed * scenario.time ∧
  -- Faster speed condition
  scenario.distance = (scenario.speed + 1/4) * (3/4 * scenario.time) ∧
  -- Slower speed condition
  scenario.distance = (scenario.speed - 1/4) * (scenario.time + 3)

/-- The theorem to be proved -/
theorem cycling_distance_is_four_point_five :
  ∀ (scenario : CyclingScenario), cycling_conditions scenario → scenario.distance = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_cycling_distance_is_four_point_five_l3075_307572


namespace NUMINAMATH_CALUDE_exists_projective_map_three_points_l3075_307579

-- Define the necessary structures
structure ProjectivePlane where
  Point : Type
  Line : Type
  incidence : Point → Line → Prop

-- Define a projective map
def ProjectiveMap (π : ProjectivePlane) := π.Point → π.Point

-- State the theorem
theorem exists_projective_map_three_points 
  (π : ProjectivePlane) 
  (l₀ l : π.Line) 
  (A₀ B₀ C₀ A B C : π.Point)
  (on_l₀ : π.incidence A₀ l₀ ∧ π.incidence B₀ l₀ ∧ π.incidence C₀ l₀)
  (on_l : π.incidence A l ∧ π.incidence B l ∧ π.incidence C l) :
  ∃ (f : ProjectiveMap π), 
    f A₀ = A ∧ f B₀ = B ∧ f C₀ = C := by
  sorry

end NUMINAMATH_CALUDE_exists_projective_map_three_points_l3075_307579


namespace NUMINAMATH_CALUDE_parabola_tangent_lines_l3075_307551

/-- A parabola defined by y^2 = 8x -/
def parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- The point P -/
def P : ℝ × ℝ := (2, 4)

/-- A line that has exactly one common point with the parabola and passes through P -/
def tangent_line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - P.2 = m * (p.1 - P.1)}

/-- The number of lines passing through P and having exactly one common point with the parabola -/
def num_tangent_lines : ℕ := 2

theorem parabola_tangent_lines :
  ∃ (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧
  (∀ m : ℝ, (tangent_line m ∩ parabola).Nonempty → m = m₁ ∨ m = m₂) ∧
  (tangent_line m₁ ∩ parabola).Nonempty ∧
  (tangent_line m₂ ∩ parabola).Nonempty :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_lines_l3075_307551


namespace NUMINAMATH_CALUDE_max_area_inscribed_equilateral_triangle_max_area_inscribed_equilateral_triangle_proof_l3075_307537

/-- The maximum area of an equilateral triangle inscribed in a 12 by 13 rectangle --/
theorem max_area_inscribed_equilateral_triangle : ℝ :=
  let rectangle_width : ℝ := 12
  let rectangle_height : ℝ := 13
  let max_area : ℝ := 312 * Real.sqrt 3 - 936
  max_area

/-- Proof that the maximum area of an equilateral triangle inscribed in a 12 by 13 rectangle is 312√3 - 936 --/
theorem max_area_inscribed_equilateral_triangle_proof :
  max_area_inscribed_equilateral_triangle = 312 * Real.sqrt 3 - 936 := by
  sorry

end NUMINAMATH_CALUDE_max_area_inscribed_equilateral_triangle_max_area_inscribed_equilateral_triangle_proof_l3075_307537

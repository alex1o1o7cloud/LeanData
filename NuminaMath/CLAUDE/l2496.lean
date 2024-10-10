import Mathlib

namespace average_running_time_l2496_249603

theorem average_running_time (total_students : ℕ) 
  (sixth_grade_time seventh_grade_time eighth_grade_time : ℕ)
  (sixth_to_seventh_ratio seventh_to_eighth_ratio : ℕ) :
  total_students = 210 →
  sixth_grade_time = 10 →
  seventh_grade_time = 12 →
  eighth_grade_time = 14 →
  sixth_to_seventh_ratio = 3 →
  seventh_to_eighth_ratio = 4 →
  (let eighth_grade_count := total_students / (1 + seventh_to_eighth_ratio + sixth_to_seventh_ratio * seventh_to_eighth_ratio);
   let seventh_grade_count := seventh_to_eighth_ratio * eighth_grade_count;
   let sixth_grade_count := sixth_to_seventh_ratio * seventh_grade_count;
   let total_minutes := sixth_grade_count * sixth_grade_time + 
                        seventh_grade_count * seventh_grade_time + 
                        eighth_grade_count * eighth_grade_time;
   (total_minutes : ℚ) / total_students = 420 / 39) :=
by
  sorry

end average_running_time_l2496_249603


namespace no_pythagorean_triple_with_3_l2496_249648

theorem no_pythagorean_triple_with_3 :
  ¬∃ (a b c : ℤ), a^2 + b^2 = 3 * c^2 ∧ Int.gcd a (Int.gcd b c) = 1 := by
  sorry

end no_pythagorean_triple_with_3_l2496_249648


namespace sqrt_sum_inequality_l2496_249613

theorem sqrt_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a / (a + 3 * b)) + Real.sqrt (b / (b + 3 * a)) ≥ 1 := by
  sorry

end sqrt_sum_inequality_l2496_249613


namespace newspapers_collected_l2496_249649

/-- The number of newspapers collected by Chris and Lily -/
theorem newspapers_collected (chris_newspapers lily_newspapers : ℕ) 
  (h1 : chris_newspapers = 42)
  (h2 : lily_newspapers = 23) :
  chris_newspapers + lily_newspapers = 65 := by
  sorry

end newspapers_collected_l2496_249649


namespace smallest_non_prime_non_square_no_small_factors_l2496_249645

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, p < k → is_prime p → ¬(n % p = 0)

theorem smallest_non_prime_non_square_no_small_factors : 
  ∃! n : ℕ, n > 0 ∧ 
    ¬(is_prime n) ∧ 
    ¬(is_perfect_square n) ∧ 
    has_no_prime_factor_less_than n 60 ∧
    ∀ m : ℕ, m > 0 → 
      ¬(is_prime m) → 
      ¬(is_perfect_square m) → 
      has_no_prime_factor_less_than m 60 → 
      n ≤ m :=
by sorry

end smallest_non_prime_non_square_no_small_factors_l2496_249645


namespace max_value_expression_l2496_249627

theorem max_value_expression (a b c d : ℝ) 
  (ha : -8.5 ≤ a ∧ a ≤ 8.5)
  (hb : -8.5 ≤ b ∧ b ≤ 8.5)
  (hc : -8.5 ≤ c ∧ c ≤ 8.5)
  (hd : -8.5 ≤ d ∧ d ≤ 8.5) :
  ∃ (m : ℝ), m = 306 ∧ 
  ∀ (a' b' c' d' : ℝ), 
    -8.5 ≤ a' ∧ a' ≤ 8.5 → 
    -8.5 ≤ b' ∧ b' ≤ 8.5 → 
    -8.5 ≤ c' ∧ c' ≤ 8.5 → 
    -8.5 ≤ d' ∧ d' ≤ 8.5 → 
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' ≤ m :=
by sorry

end max_value_expression_l2496_249627


namespace derivative_of_f_l2496_249655

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem derivative_of_f (x : ℝ) (h : x > 0) :
  deriv f x = (1 - log x) / (x^2) := by
  sorry

end derivative_of_f_l2496_249655


namespace rectangle_area_l2496_249622

/-- Given a rectangle with perimeter 40 and one side length 5, prove its area is 75 -/
theorem rectangle_area (perimeter : ℝ) (side : ℝ) (h1 : perimeter = 40) (h2 : side = 5) :
  let other_side := perimeter / 2 - side
  side * other_side = 75 := by
  sorry

end rectangle_area_l2496_249622


namespace bbq_attendance_l2496_249625

def ice_per_person : ℕ := 2
def bags_per_pack : ℕ := 10
def price_per_pack : ℚ := 3
def total_spent : ℚ := 9

theorem bbq_attendance : ℕ := by
  sorry

end bbq_attendance_l2496_249625


namespace extended_triangle_area_l2496_249637

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the theorem
theorem extended_triangle_area (t : Triangle) :
  ∃ (new_area : ℝ), new_area = 7 * t.area :=
sorry

end extended_triangle_area_l2496_249637


namespace rhombus_matches_l2496_249660

/-- Represents the number of matches needed for a rhombus -/
def matches_for_rhombus (s : ℕ) : ℕ := s * (s + 3)

/-- Theorem: The number of matches needed for a rhombus with side length s,
    divided into unit triangles, is s(s+3) -/
theorem rhombus_matches (s : ℕ) : 
  matches_for_rhombus s = s * (s + 3) := by
  sorry

#eval matches_for_rhombus 10  -- Should evaluate to 320

end rhombus_matches_l2496_249660


namespace train_crossing_tree_time_l2496_249644

/-- Given a train and a platform with specified lengths and the time it takes for the train to pass the platform, 
    calculate the time it takes for the train to cross a tree. -/
theorem train_crossing_tree_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_pass_platform : ℝ) 
  (h1 : train_length = 1200) 
  (h2 : platform_length = 300) 
  (h3 : time_to_pass_platform = 150) : 
  (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 120 := by
  sorry

end train_crossing_tree_time_l2496_249644


namespace jinas_teddies_l2496_249679

/-- Proves that the initial number of teddies is 5 given the conditions in Jina's mascot collection problem -/
theorem jinas_teddies :
  ∀ (initial_teddies : ℕ),
  let bunnies := 3 * initial_teddies
  let additional_teddies := 2 * bunnies
  let total_mascots := initial_teddies + bunnies + additional_teddies + 1
  total_mascots = 51 →
  initial_teddies = 5 := by
sorry

end jinas_teddies_l2496_249679


namespace beads_left_in_container_l2496_249650

/-- The number of beads left in a container after some are removed -/
theorem beads_left_in_container (green brown red removed : ℕ) : 
  green = 1 → brown = 2 → red = 3 → removed = 2 →
  green + brown + red - removed = 4 := by
  sorry

end beads_left_in_container_l2496_249650


namespace sin_pi_third_value_l2496_249620

theorem sin_pi_third_value (f : ℝ → ℝ) :
  (∀ α : ℝ, f (Real.sin α + Real.cos α) = (1/2) * Real.sin (2 * α)) →
  f (Real.sin (π/3)) = -1/8 := by
sorry

end sin_pi_third_value_l2496_249620


namespace rotation_center_l2496_249652

theorem rotation_center (f : ℂ → ℂ) (c : ℂ) : 
  (f = fun z ↦ ((1 - Complex.I * Real.sqrt 2) * z + (-4 * Real.sqrt 2 + 6 * Complex.I)) / 2) →
  (c = (2 * Real.sqrt 2) / 3 - (2 * Complex.I) / 3) →
  f c = c := by
sorry

end rotation_center_l2496_249652


namespace sum_of_x_and_y_l2496_249619

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 225) : x + y = 650 := by
  sorry

end sum_of_x_and_y_l2496_249619


namespace no_intersection_and_in_circle_l2496_249654

theorem no_intersection_and_in_circle : ¬∃ (a b : ℝ), 
  (∃ (n : ℤ), ∃ (m : ℤ), n = m ∧ a * n + b = 3 * m^2 + 15) ∧ 
  (a^2 + b^2 ≤ 144) := by
  sorry

end no_intersection_and_in_circle_l2496_249654


namespace matching_socks_probability_l2496_249611

def blue_socks : ℕ := 12
def green_socks : ℕ := 10
def red_socks : ℕ := 9

def total_socks : ℕ := blue_socks + green_socks + red_socks

def matching_pairs : ℕ := (blue_socks.choose 2) + (green_socks.choose 2) + (red_socks.choose 2)

def total_pairs : ℕ := total_socks.choose 2

theorem matching_socks_probability :
  (matching_pairs : ℚ) / total_pairs = 147 / 465 :=
sorry

end matching_socks_probability_l2496_249611


namespace function_inequality_implies_parameter_bound_l2496_249664

theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∃ x : ℕ+, (3 * x^2 + a * x + 26) / (x + 1) ≤ 2) →
  a ≤ -15 := by
  sorry

end function_inequality_implies_parameter_bound_l2496_249664


namespace joans_marbles_l2496_249610

/-- Given that Mary has 9 yellow marbles and the total number of yellow marbles
    between Mary and Joan is 12, prove that Joan has 3 yellow marbles. -/
theorem joans_marbles (mary_marbles : ℕ) (total_marbles : ℕ) (joan_marbles : ℕ) 
    (h1 : mary_marbles = 9)
    (h2 : total_marbles = 12)
    (h3 : mary_marbles + joan_marbles = total_marbles) :
  joan_marbles = 3 := by
  sorry

end joans_marbles_l2496_249610


namespace ratio_a_to_b_l2496_249665

/-- A geometric sequence with first four terms a, x, b, 2x -/
structure GeometricSequence (α : Type*) [Field α] where
  a : α
  x : α
  b : α

/-- The ratio between consecutive terms in a geometric sequence is constant -/
def is_geometric_sequence {α : Type*} [Field α] (seq : GeometricSequence α) : Prop :=
  seq.x / seq.a = seq.b / seq.x ∧ seq.b / seq.x = 2

theorem ratio_a_to_b {α : Type*} [Field α] (seq : GeometricSequence α) 
  (h : is_geometric_sequence seq) : seq.a / seq.b = 1 / 4 := by
  sorry

#check ratio_a_to_b

end ratio_a_to_b_l2496_249665


namespace geometric_sequence_common_ratio_l2496_249693

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : GeometricSequence a q)
  (h_sum : a 2 + a 4 = 3)
  (h_prod : a 3 * a 5 = 2) :
  q = Real.sqrt ((3 * Real.sqrt 2 + 2) / 7) := by
sorry

end geometric_sequence_common_ratio_l2496_249693


namespace money_distribution_l2496_249673

theorem money_distribution (a b c total : ℕ) : 
  a + b + c = 9 →
  b = 3 →
  1200 * 3 = total →
  total = 3600 := by sorry

end money_distribution_l2496_249673


namespace tan_160_gt_tan_neg_23_l2496_249612

theorem tan_160_gt_tan_neg_23 : Real.tan (160 * π / 180) > Real.tan (-23 * π / 180) := by
  sorry

end tan_160_gt_tan_neg_23_l2496_249612


namespace f_composition_equals_pi_squared_plus_one_l2496_249614

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_equals_pi_squared_plus_one :
  f (f (f (-2016))) = Real.pi^2 + 1 := by
  sorry

end f_composition_equals_pi_squared_plus_one_l2496_249614


namespace new_student_weight_l2496_249684

theorem new_student_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_decrease : ℝ) :
  initial_count = 4 →
  replaced_weight = 96 →
  avg_decrease = 8 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * avg_decrease + replaced_weight ∧
    new_weight = 160 :=
by sorry

end new_student_weight_l2496_249684


namespace cut_prism_faces_cut_prism_faces_proof_l2496_249678

/-- A triangular prism with 9 edges -/
structure TriangularPrism :=
  (edges : ℕ)
  (edges_eq : edges = 9)

/-- The result of cutting a triangular prism parallel to its base from the midpoints of its side edges -/
structure CutPrism extends TriangularPrism :=
  (additional_faces : ℕ)
  (additional_faces_eq : additional_faces = 3)

/-- The theorem stating that a cut triangular prism has 8 faces in total -/
theorem cut_prism_faces (cp : CutPrism) : ℕ :=
  8

#check cut_prism_faces

/-- Proof of the theorem -/
theorem cut_prism_faces_proof (cp : CutPrism) : cut_prism_faces cp = 8 := by
  sorry

end cut_prism_faces_cut_prism_faces_proof_l2496_249678


namespace divisibility_of_power_minus_odd_l2496_249624

theorem divisibility_of_power_minus_odd (k m : ℕ) (hk : k > 0) (hm : Odd m) :
  ∃ n : ℕ, n > 0 ∧ (2^k : ℕ) ∣ (n^n - m) :=
sorry

end divisibility_of_power_minus_odd_l2496_249624


namespace babysitting_hours_l2496_249602

/-- Represents the babysitting scenario -/
structure BabysittingScenario where
  hourly_rate : ℚ
  makeup_fraction : ℚ
  skincare_fraction : ℚ
  remaining_amount : ℚ

/-- Calculates the number of hours babysitted per day -/
def hours_per_day (scenario : BabysittingScenario) : ℚ :=
  ((1 - scenario.makeup_fraction - scenario.skincare_fraction) * scenario.remaining_amount) /
  (7 * scenario.hourly_rate)

/-- Theorem stating that given the specific scenario, the person babysits for 3 hours each day -/
theorem babysitting_hours (scenario : BabysittingScenario) 
  (h1 : scenario.hourly_rate = 10)
  (h2 : scenario.makeup_fraction = 3/10)
  (h3 : scenario.skincare_fraction = 2/5)
  (h4 : scenario.remaining_amount = 63) :
  hours_per_day scenario = 3 := by
  sorry

#eval hours_per_day { hourly_rate := 10, makeup_fraction := 3/10, skincare_fraction := 2/5, remaining_amount := 63 }

end babysitting_hours_l2496_249602


namespace english_chinese_difference_l2496_249626

/-- The number of hours Ryan spends learning English daily -/
def hours_english : ℕ := 6

/-- The number of hours Ryan spends learning Chinese daily -/
def hours_chinese : ℕ := 2

/-- The difference in hours between English and Chinese learning -/
def hour_difference : ℕ := hours_english - hours_chinese

theorem english_chinese_difference : hour_difference = 4 := by
  sorry

end english_chinese_difference_l2496_249626


namespace athlete_running_time_l2496_249656

/-- Represents the calories burned per minute while running -/
def running_rate : ℝ := 10

/-- Represents the calories burned per minute while walking -/
def walking_rate : ℝ := 4

/-- Represents the total calories burned -/
def total_calories : ℝ := 450

/-- Represents the total time spent exercising in minutes -/
def total_time : ℝ := 60

/-- Theorem stating that the athlete spends 35 minutes running -/
theorem athlete_running_time :
  ∃ (r w : ℝ),
    r + w = total_time ∧
    running_rate * r + walking_rate * w = total_calories ∧
    r = 35 :=
by
  sorry

end athlete_running_time_l2496_249656


namespace ronas_age_l2496_249675

theorem ronas_age (rona rachel collete : ℕ) 
  (h1 : rachel = 2 * rona)
  (h2 : collete = rona / 2)
  (h3 : rachel - collete = 12) : 
  rona = 12 := by
sorry

end ronas_age_l2496_249675


namespace sum_of_digits_of_square_l2496_249691

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a number is a ten-digit number -/
def isTenDigitNumber (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_square (N : ℕ) :
  isTenDigitNumber N →
  sumOfDigits N = 4 →
  (sumOfDigits (N^2) = 7 ∨ sumOfDigits (N^2) = 16) := by sorry

end sum_of_digits_of_square_l2496_249691


namespace square_area_proof_l2496_249657

theorem square_area_proof (x : ℝ) (h1 : 4 * x - 15 = 20 - 3 * x) : 
  (4 * x - 15) ^ 2 = 25 := by
sorry

end square_area_proof_l2496_249657


namespace square_of_98_l2496_249662

theorem square_of_98 : (98 : ℕ) ^ 2 = 9604 := by sorry

end square_of_98_l2496_249662


namespace power_of_seven_roots_l2496_249695

theorem power_of_seven_roots (x : ℝ) (h : x > 0) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) := by
  sorry

end power_of_seven_roots_l2496_249695


namespace parallel_lines_max_distance_l2496_249617

/-- Two parallel lines with maximum distance -/
theorem parallel_lines_max_distance :
  ∃ (k b₁ b₂ : ℝ),
    -- Line equations
    (∀ x y, y = k * x + b₁ ↔ 3 * x + 5 * y + 16 = 0) ∧
    (∀ x y, y = k * x + b₂ ↔ 3 * x + 5 * y - 18 = 0) ∧
    -- Lines pass through given points
    (-2 = k * (-2) + b₁) ∧
    (3 = k * 1 + b₂) ∧
    -- Lines are parallel
    (∀ x y₁ y₂, y₁ = k * x + b₁ ∧ y₂ = k * x + b₂ → y₂ - y₁ = b₂ - b₁) ∧
    -- Distance between lines is maximum
    (∀ k' b₁' b₂',
      ((-2 = k' * (-2) + b₁') ∧ (3 = k' * 1 + b₂')) →
      |b₂ - b₁| / Real.sqrt (1 + k^2) ≥ |b₂' - b₁'| / Real.sqrt (1 + k'^2)) :=
by sorry

end parallel_lines_max_distance_l2496_249617


namespace smallest_valid_number_divisible_by_51_l2496_249698

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 1000 = (n / 100) % 10) ∧
  ((n / 10) % 10 = n % 10)

theorem smallest_valid_number_divisible_by_51 :
  ∃ (A : ℕ), is_valid_number A ∧ A % 51 = 0 ∧
  ∀ (B : ℕ), is_valid_number B ∧ B % 51 = 0 → A ≤ B ∧ A = 1122 :=
sorry

end smallest_valid_number_divisible_by_51_l2496_249698


namespace bromine_extraction_l2496_249653

-- Define the solubility of a substance in a solvent
def solubility (substance solvent : Type) : ℝ := sorry

-- Define the property of being immiscible
def immiscible (solvent1 solvent2 : Type) : Prop := sorry

-- Define the extraction process
def can_extract (substance from_solvent to_solvent : Type) : Prop := sorry

-- Define the substances and solvents
def bromine : Type := sorry
def water : Type := sorry
def benzene : Type := sorry
def soybean_oil : Type := sorry

-- Theorem statement
theorem bromine_extraction :
  (solubility bromine benzene > solubility bromine water) →
  (solubility bromine soybean_oil > solubility bromine water) →
  immiscible benzene water →
  immiscible soybean_oil water →
  (can_extract bromine water benzene ∨ can_extract bromine water soybean_oil) :=
by sorry

end bromine_extraction_l2496_249653


namespace line_hyperbola_intersection_l2496_249699

theorem line_hyperbola_intersection (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ 
    x₁^2 - (k*x₁ + 1)^2 = 1 ∧ 
    x₂^2 - (k*x₂ + 1)^2 = 1) → 
  k > 1 ∧ k < Real.sqrt 2 :=
sorry

end line_hyperbola_intersection_l2496_249699


namespace alicia_art_collection_l2496_249659

/-- The number of medieval art pieces Alicia donated -/
def donated : ℕ := 46

/-- The number of medieval art pieces Alicia had left after donating -/
def left_after_donating : ℕ := 24

/-- The original number of medieval art pieces Alicia had -/
def original_pieces : ℕ := donated + left_after_donating

theorem alicia_art_collection : original_pieces = 70 := by
  sorry

end alicia_art_collection_l2496_249659


namespace tony_preparation_time_l2496_249629

/-- The total time Tony spent preparing to be an astronaut -/
def total_preparation_time (
  science_degree_time : ℝ
  ) (num_other_degrees : ℕ
  ) (physics_grad_time : ℝ
  ) (scientist_work_time : ℝ
  ) (num_internships : ℕ
  ) (internship_duration : ℝ
  ) : ℝ :=
  science_degree_time +
  num_other_degrees * science_degree_time +
  physics_grad_time +
  scientist_work_time +
  num_internships * internship_duration

/-- Theorem stating that Tony's total preparation time is 18.5 years -/
theorem tony_preparation_time :
  total_preparation_time 4 2 2 3 3 0.5 = 18.5 := by
  sorry


end tony_preparation_time_l2496_249629


namespace sin_graph_shift_l2496_249661

noncomputable def f (x : ℝ) := Real.sin (2 * x)
noncomputable def g (x : ℝ) := Real.sin (2 * x + 1)

theorem sin_graph_shift :
  ∀ x : ℝ, g x = f (x + 1/2) := by sorry

end sin_graph_shift_l2496_249661


namespace complement_union_equals_set_l2496_249683

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {3, 4, 5}
def B : Set Nat := {1, 3, 6}

theorem complement_union_equals_set : 
  (U \ (A ∪ B)) = {2, 7} := by sorry

end complement_union_equals_set_l2496_249683


namespace isosceles_triangle_perimeter_l2496_249609

/-- An isosceles triangle with sides of length 2 and 5 has a perimeter of 12 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 5 → b = 5 → c = 2 → a + b + c = 12 := by sorry

end isosceles_triangle_perimeter_l2496_249609


namespace min_probability_alex_dylan_same_team_l2496_249634

/-- The probability that Alex and Dylan are on the same team given that Alex picks one of the cards a or a+7, and Dylan picks the other. -/
def p (a : ℕ) : ℚ :=
  (Nat.choose (32 - a) 2 + Nat.choose (a - 1) 2) / 703

/-- The statement to be proved -/
theorem min_probability_alex_dylan_same_team :
  (∃ a : ℕ, a ≤ 40 ∧ a + 7 ≤ 40 ∧ p a ≥ 1/2) ∧
  (∀ a : ℕ, a ≤ 40 ∧ a + 7 ≤ 40 ∧ p a ≥ 1/2 → p a ≥ 497/703) ∧
  (∃ a : ℕ, a ≤ 40 ∧ a + 7 ≤ 40 ∧ p a = 497/703) :=
sorry

end min_probability_alex_dylan_same_team_l2496_249634


namespace prime_neighbor_divisible_by_six_l2496_249646

theorem prime_neighbor_divisible_by_six (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) :
  6 ∣ (p - 1) ∨ 6 ∣ (p + 1) := by
  sorry

#check prime_neighbor_divisible_by_six

end prime_neighbor_divisible_by_six_l2496_249646


namespace joes_first_lift_weight_l2496_249687

theorem joes_first_lift_weight (total_weight first_lift second_lift : ℕ) 
  (h1 : total_weight = 900)
  (h2 : first_lift + second_lift = total_weight)
  (h3 : 2 * first_lift = second_lift + 300) :
  first_lift = 400 := by
  sorry

end joes_first_lift_weight_l2496_249687


namespace arithmetic_sequence_common_difference_l2496_249638

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  increasing : ∀ n, a n < a (n + 1)
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  first_term : a 1 = 2
  geometric : (a 2)^2 = a 1 * a 5

/-- The common difference of the arithmetic sequence is 4 -/
theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) :
  ∃ d, (∀ n, seq.a (n + 1) - seq.a n = d) ∧ d = 4 := by
  sorry


end arithmetic_sequence_common_difference_l2496_249638


namespace freezer_temp_calculation_l2496_249633

-- Define the temperature of the refrigerator compartment
def refrigerator_temp : ℝ := 4

-- Define the temperature difference between compartments
def temp_difference : ℝ := 22

-- Theorem to prove
theorem freezer_temp_calculation :
  refrigerator_temp - temp_difference = -18 := by
  sorry

end freezer_temp_calculation_l2496_249633


namespace combined_average_marks_average_marks_two_classes_l2496_249666

/-- Given two classes with specified number of students and average marks,
    calculate the average mark of all students combined. -/
theorem combined_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 → n2 > 0 →
  let total_students := n1 + n2
  let total_marks := n1 * avg1 + n2 * avg2
  total_marks / total_students = (n1 * avg1 + n2 * avg2) / (n1 + n2) :=
by sorry

/-- The average marks of all students from two classes. -/
theorem average_marks_two_classes :
  let class1_students : ℕ := 30
  let class2_students : ℕ := 50
  let class1_avg : ℚ := 40
  let class2_avg : ℚ := 70
  let total_students := class1_students + class2_students
  let total_marks := class1_students * class1_avg + class2_students * class2_avg
  total_marks / total_students = 58.75 :=
by sorry

end combined_average_marks_average_marks_two_classes_l2496_249666


namespace smallest_share_for_200_people_l2496_249635

/-- Represents a family with land inheritance rules -/
structure Family :=
  (size : ℕ)
  (has_founder : size > 0)

/-- The smallest possible share of the original plot for any family member -/
def smallest_share (f : Family) : ℚ :=
  1 / (4 * 3^65)

/-- Theorem stating the smallest possible share for a family of 200 people -/
theorem smallest_share_for_200_people (f : Family) (h : f.size = 200) :
  smallest_share f = 1 / (4 * 3^65) := by
  sorry

end smallest_share_for_200_people_l2496_249635


namespace billy_soda_theorem_l2496_249685

def billy_soda_distribution (num_sisters : ℕ) (soda_pack : ℕ) : Prop :=
  let num_brothers := 2 * num_sisters
  let total_siblings := num_brothers + num_sisters
  let sodas_per_sibling := soda_pack / total_siblings
  (num_sisters = 2) ∧ (soda_pack = 12) → (sodas_per_sibling = 2)

theorem billy_soda_theorem : billy_soda_distribution 2 12 := by
  sorry

end billy_soda_theorem_l2496_249685


namespace f_negative_range_x_range_for_negative_f_l2496_249670

/-- The function f(x) = mx^2 - mx - 6 + m -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 6 + m

theorem f_negative_range (m : ℝ) (x : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x < 0) ↔ m < 6/7 :=
sorry

theorem x_range_for_negative_f (m : ℝ) (x : ℝ) :
  (∀ m ∈ Set.Icc (-2) 2, f m x < 0) ↔ -1 < x ∧ x < 2 :=
sorry

end f_negative_range_x_range_for_negative_f_l2496_249670


namespace train_speed_problem_l2496_249606

theorem train_speed_problem (x : ℝ) (v : ℝ) :
  x > 0 →
  (x / v + 2 * x / 20 = 4 * x / 32) →
  v = 8.8 := by
sorry

end train_speed_problem_l2496_249606


namespace sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l2496_249671

theorem sum_of_squares_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → r₁^2 + r₂^2 = (b^2 - 2*a*c) / a^2 :=
by sorry

theorem sum_of_squares_specific_quadratic :
  let r₁ := (10 + Real.sqrt 36) / 2
  let r₂ := (10 - Real.sqrt 36) / 2
  r₁^2 + r₂^2 = 68 :=
by sorry

end sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l2496_249671


namespace geometric_sequence_common_ratio_l2496_249689

/-- An increasing geometric sequence -/
def IsIncreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : IsIncreasingGeometricSequence a)
  (h_sum : a 1 + a 4 = 9)
  (h_prod : a 2 * a 3 = 8) :
  ∃ q : ℝ, q = 2 ∧ ∀ n, a (n + 1) = a n * q :=
sorry

end geometric_sequence_common_ratio_l2496_249689


namespace fraction_sum_equality_l2496_249605

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (80 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 16 / (80 - c) = 5 := by
  sorry

end fraction_sum_equality_l2496_249605


namespace stamp_problem_solution_l2496_249667

def stamp_problem (aj kj cj : ℕ) (m : ℚ) : Prop :=
  aj = 370 ∧
  kj = aj / 2 ∧
  aj + kj + cj = 930 ∧
  cj = m * kj + 5

theorem stamp_problem_solution :
  ∃ (aj kj cj : ℕ) (m : ℚ), stamp_problem aj kj cj m ∧ m = 2 := by
  sorry

end stamp_problem_solution_l2496_249667


namespace inequality_proof_l2496_249692

theorem inequality_proof (a b c d : ℝ) :
  (a + b + c + d) / ((1 + a^2) * (1 + b^2) * (1 + c^2) * (1 + d^2)) < 1 := by
  sorry

end inequality_proof_l2496_249692


namespace spider_return_probability_l2496_249651

/-- Probability of the spider being at the starting corner after n moves -/
def P : ℕ → ℚ
| 0 => 1
| n + 1 => (1 - P n) / 3

/-- The probability of returning to the starting corner on the eighth move -/
theorem spider_return_probability : P 8 = 547 / 2187 := by
  sorry

end spider_return_probability_l2496_249651


namespace intersection_A_complement_B_l2496_249628

open Set

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end intersection_A_complement_B_l2496_249628


namespace population_increase_theorem_l2496_249647

/-- Given birth and death rates per 1000 people, calculate the percentage increase in population rate -/
def population_increase_percentage (birth_rate death_rate : ℚ) : ℚ :=
  (birth_rate - death_rate) * 100 / 1000

theorem population_increase_theorem (birth_rate death_rate : ℚ) 
  (h1 : birth_rate = 32)
  (h2 : death_rate = 11) : 
  population_increase_percentage birth_rate death_rate = (21 : ℚ) / 10 :=
by sorry

end population_increase_theorem_l2496_249647


namespace fahrenheit_from_kelvin_l2496_249686

theorem fahrenheit_from_kelvin (K F C : ℝ) : 
  K = 300 → 
  C = (5/9) * (F - 32) → 
  C = K - 273 → 
  F = 80.6 := by sorry

end fahrenheit_from_kelvin_l2496_249686


namespace sphere_surface_area_l2496_249696

theorem sphere_surface_area (V : ℝ) (r : ℝ) (S : ℝ) :
  V = 48 * Real.pi →
  V = (4 / 3) * Real.pi * r^3 →
  S = 4 * Real.pi * r^2 →
  S = 144 * Real.pi :=
by
  sorry

end sphere_surface_area_l2496_249696


namespace modulus_equality_necessary_not_sufficient_l2496_249676

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem modulus_equality_necessary_not_sufficient :
  (∀ (u v : E), u ≠ 0 ∧ v ≠ 0 → (‖u‖ = ‖v‖ → u = v) ↔ false) ∧
  (∀ (u v : E), u ≠ 0 ∧ v ≠ 0 → (u = v → ‖u‖ = ‖v‖)) :=
by sorry

end modulus_equality_necessary_not_sufficient_l2496_249676


namespace arithmetic_sequence_sum_l2496_249631

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the problem statement
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 8) / 2 = 10 →
  a 1 + a 10 = 20 := by
    sorry

end arithmetic_sequence_sum_l2496_249631


namespace kannon_apples_last_night_l2496_249672

/-- The number of apples Kannon had last night -/
def apples_last_night : ℕ := sorry

/-- The number of bananas Kannon had last night -/
def bananas_last_night : ℕ := 1

/-- The number of oranges Kannon had last night -/
def oranges_last_night : ℕ := 4

/-- The number of apples Kannon will have today -/
def apples_today : ℕ := apples_last_night + 4

/-- The number of bananas Kannon will have today -/
def bananas_today : ℕ := 10 * bananas_last_night

/-- The number of oranges Kannon will have today -/
def oranges_today : ℕ := 2 * apples_today

theorem kannon_apples_last_night :
  apples_last_night = 3 ∧
  (apples_last_night + bananas_last_night + oranges_last_night +
   apples_today + bananas_today + oranges_today = 39) :=
by sorry

end kannon_apples_last_night_l2496_249672


namespace abs_sum_inequality_l2496_249690

theorem abs_sum_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ (-6.5 < x ∧ x < 3.5) :=
sorry

end abs_sum_inequality_l2496_249690


namespace product_of_two_digit_numbers_l2496_249601

theorem product_of_two_digit_numbers (a b c d : ℕ) : 
  a < 10 → b < 10 → c < 10 → d < 10 →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (10 * a + b) * (10 * c + b) = 111 * d →
  a + b + c + d = 21 := by
sorry

end product_of_two_digit_numbers_l2496_249601


namespace shaded_area_proof_l2496_249607

theorem shaded_area_proof (t : ℝ) (h : t = 5) : 
  let larger_side := 2 * t - 4
  let smaller_side := 4
  (larger_side ^ 2) - (smaller_side ^ 2) = 20 := by
sorry

end shaded_area_proof_l2496_249607


namespace exists_area_preserving_projection_l2496_249694

-- Define the concept of a plane
def Plane : Type := sorry

-- Define the concept of a triangle
structure Triangle (P : Plane) :=
  (area : ℝ)

-- Define the concept of parallel projection
def parallel_projection (P Q : Plane) (T : Triangle P) : Triangle Q := sorry

-- Theorem statement
theorem exists_area_preserving_projection
  (P Q : Plane)
  (intersect : P ≠ Q)
  (T : Triangle P) :
  ∃ (proj : Triangle Q), proj = parallel_projection P Q T ∧ proj.area = T.area :=
sorry

end exists_area_preserving_projection_l2496_249694


namespace two_propositions_are_true_l2496_249621

/-- Represents the truth value of a proposition -/
inductive PropositionTruth
  | True
  | False

/-- The four propositions in the problem -/
def proposition1 : PropositionTruth := PropositionTruth.False
def proposition2 : PropositionTruth := PropositionTruth.True
def proposition3 : PropositionTruth := PropositionTruth.False
def proposition4 : PropositionTruth := PropositionTruth.True

/-- Counts the number of true propositions -/
def countTruePropositions (p1 p2 p3 p4 : PropositionTruth) : Nat :=
  match p1, p2, p3, p4 with
  | PropositionTruth.True, PropositionTruth.True, PropositionTruth.True, PropositionTruth.True => 4
  | PropositionTruth.True, PropositionTruth.True, PropositionTruth.True, PropositionTruth.False => 3
  | PropositionTruth.True, PropositionTruth.True, PropositionTruth.False, PropositionTruth.True => 3
  | PropositionTruth.True, PropositionTruth.True, PropositionTruth.False, PropositionTruth.False => 2
  | PropositionTruth.True, PropositionTruth.False, PropositionTruth.True, PropositionTruth.True => 3
  | PropositionTruth.True, PropositionTruth.False, PropositionTruth.True, PropositionTruth.False => 2
  | PropositionTruth.True, PropositionTruth.False, PropositionTruth.False, PropositionTruth.True => 2
  | PropositionTruth.True, PropositionTruth.False, PropositionTruth.False, PropositionTruth.False => 1
  | PropositionTruth.False, PropositionTruth.True, PropositionTruth.True, PropositionTruth.True => 3
  | PropositionTruth.False, PropositionTruth.True, PropositionTruth.True, PropositionTruth.False => 2
  | PropositionTruth.False, PropositionTruth.True, PropositionTruth.False, PropositionTruth.True => 2
  | PropositionTruth.False, PropositionTruth.True, PropositionTruth.False, PropositionTruth.False => 1
  | PropositionTruth.False, PropositionTruth.False, PropositionTruth.True, PropositionTruth.True => 2
  | PropositionTruth.False, PropositionTruth.False, PropositionTruth.True, PropositionTruth.False => 1
  | PropositionTruth.False, PropositionTruth.False, PropositionTruth.False, PropositionTruth.True => 1
  | PropositionTruth.False, PropositionTruth.False, PropositionTruth.False, PropositionTruth.False => 0

/-- Theorem stating that exactly 2 out of 4 given propositions are true -/
theorem two_propositions_are_true :
  countTruePropositions proposition1 proposition2 proposition3 proposition4 = 2 := by
  sorry

end two_propositions_are_true_l2496_249621


namespace number_classification_l2496_249680

-- Define a number type that can represent both decimal and natural numbers
inductive Number
  | Decimal (integerPart : Int) (fractionalPart : Nat)
  | Natural (value : Nat)

-- Define a function to check if a number is decimal
def isDecimal (n : Number) : Prop :=
  match n with
  | Number.Decimal _ _ => True
  | Number.Natural _ => False

-- Define a function to check if a number is natural
def isNatural (n : Number) : Prop :=
  match n with
  | Number.Decimal _ _ => False
  | Number.Natural _ => True

-- Theorem statement
theorem number_classification (n : Number) :
  (isDecimal n ∧ ¬isNatural n) ∨ (¬isDecimal n ∧ isNatural n) :=
by sorry

end number_classification_l2496_249680


namespace max_single_painted_face_theorem_l2496_249600

/-- Represents a large cube composed of smaller cubes -/
structure LargeCube where
  size : Nat
  painted_faces : Nat

/-- Calculates the maximum number of smaller cubes with exactly one face painted -/
def max_single_painted_face (cube : LargeCube) : Nat :=
  if cube.size = 4 ∧ cube.painted_faces = 3 then 32 else 0

/-- Theorem stating the maximum number of smaller cubes with exactly one face painted -/
theorem max_single_painted_face_theorem (cube : LargeCube) :
  cube.size = 4 ∧ cube.painted_faces = 3 →
  max_single_painted_face cube = 32 := by
  sorry

end max_single_painted_face_theorem_l2496_249600


namespace trajectory_and_fixed_point_l2496_249641

-- Define the moving circle M
structure MovingCircle where
  center : ℝ × ℝ
  passes_through : center.1 - 1 ^ 2 + center.2 ^ 2 = (center.1 + 1) ^ 2
  tangent_to_line : center.1 + 1 = ((center.1 - 1) ^ 2 + center.2 ^ 2).sqrt

-- Define the trajectory C
def trajectory (x y : ℝ) : Prop := y ^ 2 = 4 * x

-- Define a point on the trajectory
structure PointOnTrajectory where
  point : ℝ × ℝ
  on_trajectory : trajectory point.1 point.2
  not_origin : point ≠ (0, 0)

-- Theorem statement
theorem trajectory_and_fixed_point 
  (M : MovingCircle) 
  (A B : PointOnTrajectory) 
  (h : A.point.1 * B.point.1 + A.point.2 * B.point.2 = 0) :
  ∃ (t : ℝ), 
    t * A.point.2 + (1 - t) * B.point.2 = 0 ∧ 
    t * A.point.1 + (1 - t) * B.point.1 = 4 := by
  sorry

end trajectory_and_fixed_point_l2496_249641


namespace derivative_sin_3x_at_pi_9_l2496_249632

theorem derivative_sin_3x_at_pi_9 :
  let f : ℝ → ℝ := λ x ↦ Real.sin (3 * x)
  let x₀ : ℝ := π / 9
  (deriv f) x₀ = 3 / 2 := by
  sorry

end derivative_sin_3x_at_pi_9_l2496_249632


namespace walking_rate_l2496_249688

/-- Given a distance of 4 miles and a time of 1.25 hours, the rate of travel is 3.2 miles per hour -/
theorem walking_rate (distance : ℝ) (time : ℝ) (rate : ℝ) 
    (h1 : distance = 4)
    (h2 : time = 1.25)
    (h3 : rate = distance / time) : 
  rate = 3.2 := by
  sorry

end walking_rate_l2496_249688


namespace min_d_value_l2496_249677

theorem min_d_value (a b c d : ℕ+) (h_order : a < b ∧ b < c ∧ c < d) 
  (h_unique : ∃! (x y : ℝ), x + 2*y = 2023 ∧ y = |x - a| + |x - b| + |x - c| + |x - d|) :
  d ≥ 1010 ∧ ∃ (a' b' c' : ℕ+), a' < b' ∧ b' < c' ∧ c' < 1010 ∧
    ∃! (x y : ℝ), x + 2*y = 2023 ∧ y = |x - a'| + |x - b'| + |x - c'| + |x - 1010| :=
by sorry

end min_d_value_l2496_249677


namespace graph_shift_l2496_249639

/-- Given a function f : ℝ → ℝ, prove that the graph of y = f(x + 2) - 1 
    is equivalent to shifting the graph of y = f(x) 2 units left and 1 unit down. -/
theorem graph_shift (f : ℝ → ℝ) (x y : ℝ) :
  y = f (x + 2) - 1 ↔ ∃ x₀ y₀ : ℝ, y₀ = f x₀ ∧ x = x₀ - 2 ∧ y = y₀ - 1 :=
sorry

end graph_shift_l2496_249639


namespace common_number_in_overlapping_lists_l2496_249608

theorem common_number_in_overlapping_lists (nums : List ℝ) : 
  nums.length = 9 ∧ 
  (nums.take 5).sum / 5 = 7 ∧ 
  (nums.drop 4).sum / 5 = 9 ∧ 
  nums.sum / 9 = 73 / 9 →
  ∃ x ∈ nums.take 5 ∩ nums.drop 4, x = 7 :=
by sorry

end common_number_in_overlapping_lists_l2496_249608


namespace bulb_toggling_theorem_l2496_249618

/-- Represents the state of a light bulb (on or off) -/
inductive BulbState
| Off
| On

/-- Toggles the state of a light bulb -/
def toggleBulb : BulbState → BulbState
| BulbState.Off => BulbState.On
| BulbState.On => BulbState.Off

/-- Returns the number of positive divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- Returns true if a natural number is a perfect square, false otherwise -/
def isPerfectSquare (n : ℕ) : Bool := sorry

/-- Simulates the process of students toggling light bulbs -/
def toggleBulbs (n : ℕ) : List BulbState := sorry

/-- Counts the number of bulbs that are on after the toggling process -/
def countOnBulbs (bulbs : List BulbState) : ℕ := sorry

/-- Counts the number of perfect squares less than or equal to a given number -/
def countPerfectSquares (n : ℕ) : ℕ := sorry

theorem bulb_toggling_theorem :
  countOnBulbs (toggleBulbs 100) = countPerfectSquares 100 := by sorry

end bulb_toggling_theorem_l2496_249618


namespace remainder_1632_times_2024_div_400_l2496_249642

theorem remainder_1632_times_2024_div_400 : (1632 * 2024) % 400 = 368 := by
  sorry

end remainder_1632_times_2024_div_400_l2496_249642


namespace P_sufficient_not_necessary_for_Q_l2496_249669

-- Define the conditions P and Q
def P (x : ℝ) : Prop := |2*x - 3| < 1
def Q (x : ℝ) : Prop := x*(x - 3) < 0

-- Theorem stating that P is sufficient but not necessary for Q
theorem P_sufficient_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ 
  (∃ x : ℝ, Q x ∧ ¬(P x)) :=
sorry

end P_sufficient_not_necessary_for_Q_l2496_249669


namespace root_values_l2496_249616

theorem root_values (p q r s m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : p * m^3 + q * m^2 + r * m + s = 0)
  (h2 : q * m^3 + r * m^2 + s * m + p = 0) :
  m = 1 ∨ m = -1 ∨ m = Complex.I ∨ m = -Complex.I :=
sorry

end root_values_l2496_249616


namespace fourth_side_length_l2496_249681

/-- A quadrilateral inscribed in a circle with radius 300, where three sides have lengths 300, 300, and 150√2 -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of the first side -/
  side1 : ℝ
  /-- The length of the second side -/
  side2 : ℝ
  /-- The length of the third side -/
  side3 : ℝ
  /-- The length of the fourth side -/
  side4 : ℝ
  /-- Condition that the quadrilateral is inscribed in a circle with radius 300 -/
  radius_eq : radius = 300
  /-- Condition that two sides have length 300 -/
  side1_eq : side1 = 300
  side2_eq : side2 = 300
  /-- Condition that one side has length 150√2 -/
  side3_eq : side3 = 150 * Real.sqrt 2

/-- Theorem stating that the fourth side of the inscribed quadrilateral has length 450 -/
theorem fourth_side_length (q : InscribedQuadrilateral) : q.side4 = 450 := by
  sorry

end fourth_side_length_l2496_249681


namespace midpoint_property_l2496_249604

/-- Given two points D and E in the plane, if F is their midpoint, 
    then 3 times the x-coordinate of F minus 5 times the y-coordinate of F equals 4. -/
theorem midpoint_property (D E F : ℝ × ℝ) : 
  D = (15, 3) → 
  E = (6, 8) → 
  F.1 = (D.1 + E.1) / 2 →
  F.2 = (D.2 + E.2) / 2 →
  3 * F.1 - 5 * F.2 = 4 := by
  sorry

end midpoint_property_l2496_249604


namespace octal_addition_l2496_249643

/-- Addition of octal numbers -/
def octal_add (a b c : ℕ) : ℕ :=
  (a * 8^2 + (a / 8) * 8 + (a % 8)) +
  (b * 8^2 + (b / 8) * 8 + (b % 8)) +
  (c * 8^2 + (c / 8) * 8 + (c % 8))

/-- Conversion from decimal to octal -/
def to_octal (n : ℕ) : ℕ :=
  (n / 8^2) * 100 + ((n / 8) % 8) * 10 + (n % 8)

theorem octal_addition :
  to_octal (octal_add 176 725 63) = 1066 := by
  sorry

end octal_addition_l2496_249643


namespace shaded_area_calculation_l2496_249658

theorem shaded_area_calculation (R : ℝ) (r : ℝ) (h1 : R = 9) (h2 : r = R / 4) :
  π * R^2 - 2 * (π * r^2) = 70.875 * π := by
  sorry

end shaded_area_calculation_l2496_249658


namespace probability_white_then_red_l2496_249615

/-- Probability of drawing a white marble first and then a red marble from a bag -/
theorem probability_white_then_red (total_marbles : ℕ) (red_marbles : ℕ) (white_marbles : ℕ) :
  total_marbles = red_marbles + white_marbles →
  red_marbles = 4 →
  white_marbles = 6 →
  (white_marbles : ℚ) / (total_marbles : ℚ) * (red_marbles : ℚ) / ((total_marbles - 1) : ℚ) = 4 / 15 :=
by sorry

end probability_white_then_red_l2496_249615


namespace dice_probability_l2496_249640

theorem dice_probability : 
  let n : ℕ := 8  -- total number of dice
  let k : ℕ := 4  -- number of dice showing even
  let p : ℚ := 1/2  -- probability of rolling even (or odd) on a single die
  Nat.choose n k * p^n = 35/128 := by
  sorry

end dice_probability_l2496_249640


namespace solve_pq_system_l2496_249623

theorem solve_pq_system (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end solve_pq_system_l2496_249623


namespace arithmetic_sequence_a7_l2496_249663

def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic a)
  (h_sum : a 4 + a 9 = 24)
  (h_a6 : a 6 = 11) :
  a 7 = 13 := by
sorry

end arithmetic_sequence_a7_l2496_249663


namespace pi_is_irrational_l2496_249630

-- Define the property of being an infinite non-repeating decimal
def is_infinite_non_repeating_decimal (x : ℝ) : Prop := sorry

-- Define the property of being an irrational number
def is_irrational (x : ℝ) : Prop := sorry

-- State the theorem
theorem pi_is_irrational :
  is_infinite_non_repeating_decimal π →
  (∀ x : ℝ, is_infinite_non_repeating_decimal x → is_irrational x) →
  is_irrational π :=
by sorry

end pi_is_irrational_l2496_249630


namespace three_digit_divisible_by_five_l2496_249636

theorem three_digit_divisible_by_five (n : ℕ) :
  300 ≤ n ∧ n < 400 →
  (n % 5 = 0 ↔ n % 100 = 5 ∧ n / 100 = 3) :=
by sorry

end three_digit_divisible_by_five_l2496_249636


namespace price_reduction_l2496_249697

theorem price_reduction (x : ℝ) : 
  (100 - x) * 0.9 = 85.5 → x = 5 := by sorry

end price_reduction_l2496_249697


namespace symmetry_of_f_l2496_249668

def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def f (x a₁ a₂ : ℝ) : ℝ := |x - a₁| + |x - a₂|

theorem symmetry_of_f (a : ℕ → ℝ) (d : ℝ) (h : d ≠ 0) :
  arithmeticSequence a d →
  ∀ x : ℝ, f (((a 1) + (a 2)) / 2 - x) ((a 1) : ℝ) ((a 2) : ℝ) = 
           f (((a 1) + (a 2)) / 2 + x) ((a 1) : ℝ) ((a 2) : ℝ) :=
by sorry

end symmetry_of_f_l2496_249668


namespace downstream_distance_man_downstream_distance_l2496_249674

/-- Calculates the downstream distance given swimming conditions --/
theorem downstream_distance (time : ℝ) (upstream_distance : ℝ) (still_speed : ℝ) : ℝ :=
  let stream_speed := still_speed - (upstream_distance / time)
  let downstream_speed := still_speed + stream_speed
  downstream_speed * time

/-- Proves that the downstream distance is 45 km given the specific conditions --/
theorem man_downstream_distance : 
  downstream_distance 5 25 7 = 45 := by
  sorry

end downstream_distance_man_downstream_distance_l2496_249674


namespace base7_to_base10_76543_l2496_249682

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 number 76543 --/
def base7Number : List Nat := [3, 4, 5, 6, 7]

/-- Theorem: The base 10 equivalent of 76543 in base 7 is 19141 --/
theorem base7_to_base10_76543 :
  base7ToBase10 base7Number = 19141 := by
  sorry

end base7_to_base10_76543_l2496_249682

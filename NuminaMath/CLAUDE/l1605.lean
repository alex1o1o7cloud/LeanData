import Mathlib

namespace NUMINAMATH_CALUDE_self_inverse_cube_mod_15_l1605_160548

theorem self_inverse_cube_mod_15 (a : ℤ) (h : a * a ≡ 1 [ZMOD 15]) :
  a^3 ≡ 1 [ZMOD 15] := by
  sorry

end NUMINAMATH_CALUDE_self_inverse_cube_mod_15_l1605_160548


namespace NUMINAMATH_CALUDE_grass_field_path_problem_l1605_160528

/-- Calculates the area of a rectangular path around a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem grass_field_path_problem (field_length field_width path_width cost_per_unit : ℝ)
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5)
  (h4 : cost_per_unit = 2) :
  path_area field_length field_width path_width = 675 ∧
  construction_cost (path_area field_length field_width path_width) cost_per_unit = 1350 := by
  sorry

#check grass_field_path_problem

end NUMINAMATH_CALUDE_grass_field_path_problem_l1605_160528


namespace NUMINAMATH_CALUDE_elimination_method_l1605_160575

theorem elimination_method (x y : ℝ) :
  (5 * x - 3 * y = -5) ∧ (5 * x + 4 * y = -1) → 7 * y = 4 :=
by sorry

end NUMINAMATH_CALUDE_elimination_method_l1605_160575


namespace NUMINAMATH_CALUDE_complement_of_A_l1605_160502

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

theorem complement_of_A : (Aᶜ : Set ℝ) = {x | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1605_160502


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l1605_160531

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem minimum_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (a 7 = a 6 + 2 * a 5) →
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) →
  (∀ k l : ℕ, 1 / k + 9 / l ≥ 11 / 4) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l1605_160531


namespace NUMINAMATH_CALUDE_equality_check_l1605_160578

theorem equality_check : 
  (2^3 ≠ 3^2) ∧ 
  (-(-2) = |-2|) ∧ 
  ((-2)^2 ≠ -2^2) ∧ 
  ((2/3)^2 ≠ 2^2/3) := by
  sorry

end NUMINAMATH_CALUDE_equality_check_l1605_160578


namespace NUMINAMATH_CALUDE_seats_needed_for_zoo_trip_l1605_160571

theorem seats_needed_for_zoo_trip (total_children : ℕ) (children_per_seat : ℕ) (seats_needed : ℕ) : 
  total_children = 58 → children_per_seat = 2 → seats_needed = total_children / children_per_seat → seats_needed = 29 := by
  sorry

end NUMINAMATH_CALUDE_seats_needed_for_zoo_trip_l1605_160571


namespace NUMINAMATH_CALUDE_cat_video_length_is_correct_l1605_160544

/-- The length of the cat video in minutes -/
def cat_video_length : ℝ := 4

/-- The total time spent watching videos in minutes -/
def total_watching_time : ℝ := 36

/-- Theorem stating that the cat video length is correct given the conditions -/
theorem cat_video_length_is_correct :
  let dog_video_length := 2 * cat_video_length
  let gorilla_video_length := 2 * (cat_video_length + dog_video_length)
  cat_video_length + dog_video_length + gorilla_video_length = total_watching_time :=
by sorry

end NUMINAMATH_CALUDE_cat_video_length_is_correct_l1605_160544


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l1605_160545

theorem smallest_number_with_remainders : ∃ n : ℕ,
  (n % 10 = 9) ∧
  (n % 9 = 8) ∧
  (n % 8 = 7) ∧
  (n % 7 = 6) ∧
  (n % 6 = 5) ∧
  (n % 5 = 4) ∧
  (n % 4 = 3) ∧
  (n % 3 = 2) ∧
  (n % 2 = 1) ∧
  (∀ m : ℕ, m < n →
    ¬((m % 10 = 9) ∧
      (m % 9 = 8) ∧
      (m % 8 = 7) ∧
      (m % 7 = 6) ∧
      (m % 6 = 5) ∧
      (m % 5 = 4) ∧
      (m % 4 = 3) ∧
      (m % 3 = 2) ∧
      (m % 2 = 1))) ∧
  n = 2519 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l1605_160545


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1605_160521

theorem quadratic_roots_property (a b : ℝ) : 
  (3 * a^2 + 4 * a - 7 = 0) → 
  (3 * b^2 + 4 * b - 7 = 0) → 
  (a - 2) * (b - 2) = 13/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1605_160521


namespace NUMINAMATH_CALUDE_valid_count_is_48_l1605_160549

/-- Represents a three-digit number with the last two digits being the same -/
structure ThreeDigitNumber where
  first : Nat
  last : Nat
  first_is_digit : first ≤ 9
  last_is_digit : last ≤ 9

/-- Checks if a ThreeDigitNumber is valid according to the problem conditions -/
def isValid (n : ThreeDigitNumber) : Prop :=
  (100 * n.first + 11 * n.last) % 3 = 0 ∧
  n.first + 2 * n.last ≤ 18

/-- The count of valid ThreeDigitNumbers -/
def validCount : Nat :=
  (ThreeDigitNumber.mk 1 0 (by norm_num) (by norm_num) ::
   ThreeDigitNumber.mk 1 3 (by norm_num) (by norm_num) ::
   ThreeDigitNumber.mk 1 6 (by norm_num) (by norm_num) ::
   -- ... (other valid ThreeDigitNumbers)
   []).length

theorem valid_count_is_48 : validCount = 48 := by
  sorry

#eval validCount

end NUMINAMATH_CALUDE_valid_count_is_48_l1605_160549


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1605_160534

theorem perpendicular_lines (b : ℝ) : 
  (∀ x y : ℝ, 3 * y + 2 * x - 4 = 0 ∧ 4 * y + b * x - 6 = 0 → 
   (2 / 3) * (b / 4) = 1) → 
  b = -6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1605_160534


namespace NUMINAMATH_CALUDE_sequence_inequality_l1605_160584

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h_n : n ≥ 2)
  (h_pos : ∀ k, 0 ≤ k ∧ k ≤ n → 0 < a k)
  (h_eq : ∀ k, 1 ≤ k ∧ k < n → (a (k-1) + a k) * (a k + a (k+1)) = a (k-1) - a (k+1)) :
  a n < 1 / (n - 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1605_160584


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1605_160547

theorem polynomial_remainder (x : ℝ) : 
  let p : ℝ → ℝ := λ x => 8*x^4 - 18*x^3 + 27*x^2 - 31*x + 14
  let d : ℝ → ℝ := λ x => 4*x - 8
  ∃ q : ℝ → ℝ, p = λ x => d x * q x + 30 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1605_160547


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l1605_160559

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y + 2 = 0 ∨ x + 2*y - 2 = 0

-- Define the point M
def point_M : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem ellipse_and_line_properties :
  -- Given conditions
  let focal_distance : ℝ := 2
  let eccentricity : ℝ := 1/2
  -- Prove the following
  ∀ (A B : ℝ × ℝ),
    -- A and B are on the ellipse
    ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 →
    -- A, B, and M are collinear
    ∃ (k : ℝ), (A.1 - point_M.1) = k * (B.1 - point_M.1) ∧ 
               (A.2 - point_M.2) = k * (B.2 - point_M.2) →
    -- AM = 2MB
    (A.1 - point_M.1)^2 + (A.2 - point_M.2)^2 = 
      4 * ((point_M.1 - B.1)^2 + (point_M.2 - B.2)^2) →
    -- The line passing through A, B, and M is line_l
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧ line_l point_M.1 point_M.2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l1605_160559


namespace NUMINAMATH_CALUDE_red_light_probability_l1605_160580

theorem red_light_probability (p : ℝ) (h1 : p = 1 / 3) :
  let probability_green := 1 - p
  let probability_red := p
  let probability_first_red_at_second := probability_green * probability_red
  probability_first_red_at_second = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_red_light_probability_l1605_160580


namespace NUMINAMATH_CALUDE_total_trees_after_planting_l1605_160561

/-- The number of walnut trees initially in the park -/
def initial_trees : ℕ := 4

/-- The number of new walnut trees to be planted -/
def new_trees : ℕ := 6

/-- Theorem: The total number of walnut trees after planting is 10 -/
theorem total_trees_after_planting : 
  initial_trees + new_trees = 10 := by sorry

end NUMINAMATH_CALUDE_total_trees_after_planting_l1605_160561


namespace NUMINAMATH_CALUDE_arc_length_central_angle_l1605_160532

theorem arc_length_central_angle (r : ℝ) (θ : ℝ) (h : θ = π / 2) :
  let circum := 2 * π * r
  let arc_length := (θ / (2 * π)) * circum
  r = 15 → arc_length = 7.5 * π := by
  sorry

end NUMINAMATH_CALUDE_arc_length_central_angle_l1605_160532


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l1605_160538

/-- Represents a box of stationery --/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Represents Alice's usage of stationery --/
def alice_usage (box : StationeryBox) : Prop :=
  box.sheets - 2 * box.envelopes = 80

/-- Represents Bob's usage of stationery --/
def bob_usage (box : StationeryBox) : Prop :=
  4 * box.envelopes = box.sheets ∧ box.envelopes ≥ 35

theorem stationery_box_sheets :
  ∃ (box : StationeryBox), alice_usage box ∧ bob_usage box ∧ box.sheets = 160 := by
  sorry

#check stationery_box_sheets

end NUMINAMATH_CALUDE_stationery_box_sheets_l1605_160538


namespace NUMINAMATH_CALUDE_stream_speed_verify_stream_speed_l1605_160560

/-- The speed of a stream given a swimmer's still water speed and relative upstream/downstream times -/
theorem stream_speed (still_speed : ℝ) (upstream_time_ratio : ℝ) : ℝ := by
  sorry

/-- Verify that the stream speed is correct for the given conditions -/
theorem verify_stream_speed :
  let still_speed := 7.5
  let upstream_time_ratio := 2
  let stream_speed := stream_speed still_speed upstream_time_ratio
  stream_speed = 2.5 ∧
  (still_speed - stream_speed) / (still_speed + stream_speed) = 1 / upstream_time_ratio := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_verify_stream_speed_l1605_160560


namespace NUMINAMATH_CALUDE_union_A_B_intersection_complement_A_B_l1605_160518

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 4 < x ∧ x < 10}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | 3 ≤ x ∧ x < 10} := by sorry

-- Theorem for the intersection of complement of A and B
theorem intersection_complement_A_B : (Set.univ \ A) ∩ B = {x | 7 ≤ x ∧ x < 10} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_complement_A_B_l1605_160518


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1605_160508

theorem greatest_divisor_with_remainders : Nat.gcd (60 - 6) (190 - 10) = 18 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1605_160508


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l1605_160554

theorem floor_abs_negative_real : ⌊|(-25.7 : ℝ)|⌋ = 25 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l1605_160554


namespace NUMINAMATH_CALUDE_min_value_theorem_l1605_160555

theorem min_value_theorem (a b : ℝ) (h : a * b = 1) :
  4 * a^2 + 9 * b^2 ≥ 12 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1605_160555


namespace NUMINAMATH_CALUDE_road_trip_time_calculation_l1605_160501

/-- Represents the road trip problem -/
theorem road_trip_time_calculation 
  (freeway_distance : ℝ) 
  (mountain_distance : ℝ) 
  (mountain_time : ℝ) 
  (speed_ratio : ℝ) :
  freeway_distance = 120 →
  mountain_distance = 25 →
  mountain_time = 75 →
  speed_ratio = 4 →
  let mountain_speed := mountain_distance / mountain_time
  let freeway_speed := speed_ratio * mountain_speed
  let freeway_time := freeway_distance / freeway_speed
  freeway_time + mountain_time = 165 :=
by sorry

end NUMINAMATH_CALUDE_road_trip_time_calculation_l1605_160501


namespace NUMINAMATH_CALUDE_malcolm_instagram_followers_l1605_160541

/-- Represents the number of followers on various social media platforms --/
structure SocialMediaFollowers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ

/-- Calculates the total number of followers across all platforms --/
def totalFollowers (smf : SocialMediaFollowers) : ℕ :=
  smf.instagram + smf.facebook + smf.twitter + smf.tiktok + smf.youtube

/-- Theorem stating that Malcolm has 240 followers on Instagram --/
theorem malcolm_instagram_followers :
  ∃ (smf : SocialMediaFollowers),
    smf.facebook = 500 ∧
    smf.twitter = (smf.instagram + smf.facebook) / 2 ∧
    smf.tiktok = 3 * smf.twitter ∧
    smf.youtube = smf.tiktok + 510 ∧
    totalFollowers smf = 3840 ∧
    smf.instagram = 240 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_instagram_followers_l1605_160541


namespace NUMINAMATH_CALUDE_largest_expression_l1605_160581

theorem largest_expression : ∀ (a b c d e : ℕ),
  a = 3 + 1 + 2 + 8 →
  b = 3 * 1 + 2 + 8 →
  c = 3 + 1 * 2 + 8 →
  d = 3 + 1 + 2 * 8 →
  e = 3 * 1 * 2 * 8 →
  e ≥ a ∧ e ≥ b ∧ e ≥ c ∧ e ≥ d :=
by sorry

end NUMINAMATH_CALUDE_largest_expression_l1605_160581


namespace NUMINAMATH_CALUDE_hildasAge_l1605_160591

def guesses : List Nat := [25, 29, 31, 33, 37, 39, 42, 45, 48, 50]

def isComposite (n : Nat) : Prop := ∃ a b, a > 1 ∧ b > 1 ∧ a * b = n

def countHighGuesses (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (λ g => g > age)).length

def offByTwo (age : Nat) (guesses : List Nat) : Nat :=
  (guesses.filter (λ g => g = age - 2 ∨ g = age + 2)).length

theorem hildasAge :
  ∃ age : Nat,
    age ∈ guesses ∧
    isComposite age ∧
    countHighGuesses age guesses ≥ guesses.length / 4 ∧
    offByTwo age guesses = 2 ∧
    age = 45 := by sorry

end NUMINAMATH_CALUDE_hildasAge_l1605_160591


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1605_160537

/-- A polynomial of degree 3 with a parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + k * x - 12

/-- The divisor x - 3 -/
def g (x : ℝ) : ℝ := x - 3

/-- The potential divisor 3x^2 + 4 -/
def h (x : ℝ) : ℝ := 3 * x^2 + 4

theorem polynomial_divisibility (k : ℝ) :
  (∃ q : ℝ → ℝ, ∀ x, f k x = g x * q x) →
  (∃ r : ℝ → ℝ, ∀ x, f k x = h x * r x) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1605_160537


namespace NUMINAMATH_CALUDE_inequality_proof_l1605_160516

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1605_160516


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1605_160511

theorem polynomial_factorization (x : ℤ) :
  5 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (5 * x^2 + 94 * x + 385) * (x + 3) * (x + 14) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1605_160511


namespace NUMINAMATH_CALUDE_work_multiple_l1605_160577

/-- Given that some number of people can complete a work in 24 days,
    this theorem proves that 4 times that number of people
    can complete half the work in 6 days. -/
theorem work_multiple (p : ℕ) (w : ℝ) : 
  (p * 24 : ℝ) = w → (4 * p * 6 : ℝ) = w / 2 := by sorry

end NUMINAMATH_CALUDE_work_multiple_l1605_160577


namespace NUMINAMATH_CALUDE_restaurant_group_cost_l1605_160598

/-- Calculates the total cost for a group to eat at a restaurant where kids eat free. -/
def group_meal_cost (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

/-- Proves that the total cost for a group of 11 people with 2 kids is $72,
    given that adult meals cost $8 and kids eat free. -/
theorem restaurant_group_cost :
  group_meal_cost 11 2 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_cost_l1605_160598


namespace NUMINAMATH_CALUDE_sum_of_solutions_prove_sum_of_solutions_l1605_160557

theorem sum_of_solutions : ℕ → Prop :=
  fun s => ∃ (S : Finset ℕ), 
    (∀ x ∈ S, (5 * x + 2 > 3 * (x - 1)) ∧ ((1/2) * x - 1 ≤ 7 - (3/2) * x)) ∧
    (∀ x : ℕ, (5 * x + 2 > 3 * (x - 1)) ∧ ((1/2) * x - 1 ≤ 7 - (3/2) * x) → x ∈ S) ∧
    (Finset.sum S id = s) ∧
    s = 10

theorem prove_sum_of_solutions : sum_of_solutions 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_prove_sum_of_solutions_l1605_160557


namespace NUMINAMATH_CALUDE_roots_cubic_equation_l1605_160533

theorem roots_cubic_equation (x₁ x₂ : ℝ) (h : x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0) :
  x₁^3 - 4*x₂^2 + 19 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_cubic_equation_l1605_160533


namespace NUMINAMATH_CALUDE_system_solution_correct_l1605_160520

theorem system_solution_correct (x y : ℝ) : 
  (x = 2 ∧ y = -2) → (x + 2*y = -2 ∧ 2*x + y = 2) := by
sorry

end NUMINAMATH_CALUDE_system_solution_correct_l1605_160520


namespace NUMINAMATH_CALUDE_triangle_tangent_inequality_l1605_160543

theorem triangle_tangent_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.tan A ^ 2 + Real.tan B ^ 2 + Real.tan C ^ 2 ≥ 
  Real.tan A * Real.tan B + Real.tan B * Real.tan C + Real.tan C * Real.tan A :=
by sorry

end NUMINAMATH_CALUDE_triangle_tangent_inequality_l1605_160543


namespace NUMINAMATH_CALUDE_principal_calculation_l1605_160594

/-- Given a principal amount, prove that it equals 2600 if the simple interest
    at 4% for 5 years is 2080 less than the principal. -/
theorem principal_calculation (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2080 → P = 2600 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l1605_160594


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_target_l1605_160515

/-- A function f satisfying the given condition for all non-zero real x -/
noncomputable def f : ℝ → ℝ := sorry

/-- The condition that f satisfies for all non-zero real x -/
axiom f_condition (x : ℝ) (hx : x ≠ 0) : 2 * f x + f (1 / x) = 5 * x + 4

/-- The value we're looking for -/
def target_value : ℝ := 2004

/-- The theorem to prove -/
theorem sum_of_roots_equals_target (x : ℝ) :
  let a : ℝ := 1
  let b : ℝ := -((3 * target_value - 4) / 10)
  let c : ℝ := 5 / 2
  x^2 + b*x + c = 0 → x + (-b/a) = (3 * target_value - 4) / 10 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_target_l1605_160515


namespace NUMINAMATH_CALUDE_circle_non_intersect_l1605_160565

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the line
def line (k : ℤ) (x y : ℝ) : Prop := y = k*x - 2

-- Define the condition for non-intersection
def non_intersect (k : ℤ) (l : ℝ) : Prop :=
  ∀ x y : ℝ, line k x y →
    (x - 1)^2 + y^2 > (1 + l)^2

-- Main theorem
theorem circle_non_intersect :
  ∃ k : ℤ, ∀ l : ℝ, l > 0 → non_intersect k l ∧ k = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_non_intersect_l1605_160565


namespace NUMINAMATH_CALUDE_range_of_a_l1605_160504

theorem range_of_a (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 / x + 1 / y = 1) :
  (∀ a : ℝ, x + y + a > 0) ↔ ∀ a : ℝ, a > -3 - 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1605_160504


namespace NUMINAMATH_CALUDE_g_sum_symmetric_l1605_160512

-- Define the function g
def g (p q r : ℝ) (x : ℝ) : ℝ := p * x^8 + q * x^6 - r * x^4 + 5

-- State the theorem
theorem g_sum_symmetric (p q r : ℝ) :
  g p q r 12 = 3 → g p q r 12 + g p q r (-12) = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_symmetric_l1605_160512


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l1605_160539

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

-- Define the derivative of the curve
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_slope_implies_a (a : ℝ) :
  curve_derivative a 1 = 2 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l1605_160539


namespace NUMINAMATH_CALUDE_average_speed_to_first_summit_l1605_160517

/-- Proves that the average speed to the first summit is equal to the overall average speed
    given the total journey time, overall average speed, and time to first summit. -/
theorem average_speed_to_first_summit
  (total_time : ℝ)
  (overall_avg_speed : ℝ)
  (time_to_first_summit : ℝ)
  (h_total_time : total_time = 8)
  (h_overall_avg_speed : overall_avg_speed = 3)
  (h_time_to_first_summit : time_to_first_summit = 3) :
  (overall_avg_speed * time_to_first_summit) / time_to_first_summit = overall_avg_speed :=
by sorry

#check average_speed_to_first_summit

end NUMINAMATH_CALUDE_average_speed_to_first_summit_l1605_160517


namespace NUMINAMATH_CALUDE_encryption_3859_l1605_160582

def encrypt_digit (d : Nat) : Nat :=
  (d^3 + 1) % 10

def encrypt_number (n : List Nat) : List Nat :=
  n.map encrypt_digit

theorem encryption_3859 :
  encrypt_number [3, 8, 5, 9] = [8, 3, 6, 0] := by
  sorry

end NUMINAMATH_CALUDE_encryption_3859_l1605_160582


namespace NUMINAMATH_CALUDE_unique_nonnegative_solution_l1605_160513

theorem unique_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -4*x := by sorry

end NUMINAMATH_CALUDE_unique_nonnegative_solution_l1605_160513


namespace NUMINAMATH_CALUDE_mork_tax_rate_l1605_160519

theorem mork_tax_rate (mork_income : ℝ) (mork_rate : ℝ) : 
  mork_income > 0 →
  mork_rate * mork_income + 0.2 * (4 * mork_income) = 0.25 * (5 * mork_income) →
  mork_rate = 0.45 := by
sorry

end NUMINAMATH_CALUDE_mork_tax_rate_l1605_160519


namespace NUMINAMATH_CALUDE_max_parts_is_ten_l1605_160562

/-- Represents a Viennese pretzel lying on a table -/
structure ViennesePretzel where
  loops : ℕ
  intersections : ℕ

/-- Represents a straight cut through the pretzel -/
structure StraightCut where
  intersectionsCut : ℕ

/-- The number of parts resulting from a straight cut -/
def numParts (p : ViennesePretzel) (c : StraightCut) : ℕ :=
  c.intersectionsCut + 1

/-- The maximum number of intersections that can be cut by a single straight line -/
def maxIntersectionsCut (p : ViennesePretzel) : ℕ := 9

/-- Theorem stating that the maximum number of parts is 10 -/
theorem max_parts_is_ten (p : ViennesePretzel) :
  ∃ c : StraightCut, numParts p c = 10 ∧
  ∀ c' : StraightCut, numParts p c' ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_parts_is_ten_l1605_160562


namespace NUMINAMATH_CALUDE_system_solution_equation_solution_l1605_160576

-- System of equations
theorem system_solution :
  ∃! (x y : ℝ), x + 2*y = 9 ∧ 3*x - 2*y = 3 ∧ x = 3 ∧ y = 3 := by sorry

-- Single equation
theorem equation_solution :
  ∃! x : ℝ, (2 - x) / (x - 3) + 3 = 2 / (3 - x) ∧ x = 5/2 := by sorry

end NUMINAMATH_CALUDE_system_solution_equation_solution_l1605_160576


namespace NUMINAMATH_CALUDE_grocery_solution_l1605_160564

/-- Represents the grocery shopping problem --/
def grocery_problem (mustard_oil_price : ℝ) (mustard_oil_amount : ℝ) 
  (pasta_price : ℝ) (sauce_price : ℝ) (sauce_amount : ℝ) 
  (initial_money : ℝ) (remaining_money : ℝ) : Prop :=
  ∃ (pasta_amount : ℝ),
    mustard_oil_price * mustard_oil_amount + 
    pasta_price * pasta_amount + 
    sauce_price * sauce_amount = 
    initial_money - remaining_money ∧
    pasta_amount = 3

/-- Theorem stating the solution to the grocery problem --/
theorem grocery_solution : 
  grocery_problem 13 2 4 5 1 50 7 := by
  sorry

end NUMINAMATH_CALUDE_grocery_solution_l1605_160564


namespace NUMINAMATH_CALUDE_sams_puppies_l1605_160530

theorem sams_puppies (initial_spotted : ℕ) (initial_nonspotted : ℕ) 
  (given_away_spotted : ℕ) (given_away_nonspotted : ℕ) 
  (remaining_spotted : ℕ) (remaining_nonspotted : ℕ) : ℕ :=
  by
  have h1 : initial_spotted = 8 := by sorry
  have h2 : initial_nonspotted = 5 := by sorry
  have h3 : given_away_spotted = 2 := by sorry
  have h4 : given_away_nonspotted = 3 := by sorry
  have h5 : remaining_spotted = 6 := by sorry
  have h6 : remaining_nonspotted = 2 := by sorry
  have h7 : initial_spotted - given_away_spotted = remaining_spotted := by sorry
  have h8 : initial_nonspotted - given_away_nonspotted = remaining_nonspotted := by sorry
  exact initial_spotted + initial_nonspotted

end NUMINAMATH_CALUDE_sams_puppies_l1605_160530


namespace NUMINAMATH_CALUDE_smallest_repeating_block_7_11_l1605_160535

/-- The length of the smallest repeating block in the decimal expansion of 7/11 -/
def repeating_block_length_7_11 : ℕ := 2

/-- The fraction we're considering -/
def fraction : ℚ := 7 / 11

theorem smallest_repeating_block_7_11 :
  repeating_block_length_7_11 = 2 ∧
  ∃ (a b : ℕ), fraction = (a : ℚ) / (10^repeating_block_length_7_11 - 1 : ℚ) +
                (b : ℚ) / (10^repeating_block_length_7_11 : ℚ) ∧
                0 ≤ a ∧ a < 10^repeating_block_length_7_11 - 1 ∧
                0 ≤ b ∧ b < 10^repeating_block_length_7_11 ∧
                ∀ (n : ℕ), n < repeating_block_length_7_11 →
                  ¬∃ (c d : ℕ), fraction = (c : ℚ) / (10^n - 1 : ℚ) +
                                (d : ℚ) / (10^n : ℚ) ∧
                                0 ≤ c ∧ c < 10^n - 1 ∧
                                0 ≤ d ∧ d < 10^n := by
  sorry

#eval repeating_block_length_7_11

end NUMINAMATH_CALUDE_smallest_repeating_block_7_11_l1605_160535


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1605_160597

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y)^2 + (y + 1/(2*x))^2 ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y)^2 + (y + 1/(2*x))^2 = 3 + 2 * Real.sqrt 2 ↔ 
  x = Real.sqrt 2 / 2 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1605_160597


namespace NUMINAMATH_CALUDE_houses_with_neither_feature_l1605_160570

theorem houses_with_neither_feature (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ) 
  (h1 : total = 85) 
  (h2 : garage = 50) 
  (h3 : pool = 40) 
  (h4 : both = 35) : 
  total - (garage + pool - both) = 30 := by
  sorry

end NUMINAMATH_CALUDE_houses_with_neither_feature_l1605_160570


namespace NUMINAMATH_CALUDE_division_problem_l1605_160593

theorem division_problem (dividend quotient remainder : ℕ) (h1 : dividend = 686) (h2 : quotient = 19) (h3 : remainder = 2) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 36 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1605_160593


namespace NUMINAMATH_CALUDE_jesse_carpet_need_l1605_160522

/-- The amount of additional carpet Jesse needs to cover two rooms -/
def additional_carpet_needed (jesse_carpet area_room_a area_room_b : ℝ) : ℝ :=
  area_room_a + area_room_b - jesse_carpet

/-- Proof that Jesse needs 94 more square feet of carpet -/
theorem jesse_carpet_need : 
  let jesse_carpet : ℝ := 18
  let room_a_length : ℝ := 4
  let room_a_width : ℝ := 20
  let area_room_a : ℝ := room_a_length * room_a_width
  let area_room_b : ℝ := area_room_a / 2.5
  additional_carpet_needed jesse_carpet area_room_a area_room_b = 94
  := by sorry

end NUMINAMATH_CALUDE_jesse_carpet_need_l1605_160522


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1605_160526

theorem inequality_equivalence (x : ℝ) : 
  Real.sqrt ((1 / (2 - x) + 1) ^ 2) ≥ 2 ↔ 
  (x ≥ 1 ∧ x < 2) ∨ (x > 2 ∧ x ≤ 7/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1605_160526


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l1605_160585

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l1605_160585


namespace NUMINAMATH_CALUDE_a_greater_than_one_l1605_160596

-- Define an increasing function on the real numbers
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem a_greater_than_one
  (f : ℝ → ℝ)
  (h_increasing : IncreasingFunction f)
  (h_inequality : f (a + 1) < f (2 * a)) :
  a > 1 :=
sorry

end NUMINAMATH_CALUDE_a_greater_than_one_l1605_160596


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1605_160579

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 10 * 2 / 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1605_160579


namespace NUMINAMATH_CALUDE_ellipse_properties_l1605_160509

/-- An ellipse with center at the origin, foci on the x-axis, and eccentricity 1/2 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_pos : 0 < b ∧ b < a
  h_ecc : c / a = 1 / 2
  h_rel : a^2 = b^2 + c^2

/-- A line passing through a point with a given angle -/
structure TangentLine where
  c : ℝ
  angle : ℝ
  h_angle : angle = π / 3

/-- The main theorem about the ellipse and its properties -/
theorem ellipse_properties (E : Ellipse) (L : TangentLine) :
  (E.a = 2 ∧ E.b = Real.sqrt 3 ∧ E.c = 1) ∧
  (∀ k m : ℝ, ∃ x y : ℝ,
    x^2 / 4 + y^2 / 3 = 1 ∧
    y = k * x + m ∧
    (x - 2)^2 + y^2 = 4 →
    k * (2 / 7) + m = 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1605_160509


namespace NUMINAMATH_CALUDE_inequality_proof_l1605_160589

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1605_160589


namespace NUMINAMATH_CALUDE_prob_red_then_black_value_l1605_160510

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (hTotal : total_cards = 52)
  (hRed : red_cards = 26)
  (hBlack : black_cards = 26)
  (hSum : red_cards + black_cards = total_cards)

/-- The probability of drawing a red card first and a black card second -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards * (d.black_cards : ℚ) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a red card first and a black card second -/
theorem prob_red_then_black_value (d : Deck) : prob_red_then_black d = 13 / 51 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_black_value_l1605_160510


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l1605_160550

theorem initial_number_of_persons (average_increase : ℝ) (weight_difference : ℝ) : 
  average_increase = 1.5 → weight_difference = 12 → 
  (average_increase * initial_persons = weight_difference) → initial_persons = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l1605_160550


namespace NUMINAMATH_CALUDE_distance_traveled_l1605_160506

/-- Calculates the actual distance traveled given the conditions of the problem -/
def actual_distance (actual_speed hours_walked : ℝ) : ℝ :=
  actual_speed * hours_walked

/-- Represents the additional distance that would be covered at the higher speed -/
def additional_distance (actual_speed higher_speed hours_walked : ℝ) : ℝ :=
  (higher_speed - actual_speed) * hours_walked

theorem distance_traveled (actual_speed higher_speed additional : ℝ) 
  (h1 : actual_speed = 12)
  (h2 : higher_speed = 20)
  (h3 : additional = 30) :
  ∃ (hours_walked : ℝ), 
    additional_distance actual_speed higher_speed hours_walked = additional ∧ 
    actual_distance actual_speed hours_walked = 45 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l1605_160506


namespace NUMINAMATH_CALUDE_sequence_term_equation_l1605_160592

def sequence_term (n : ℕ+) : ℕ := 9 * (n - 1) + n

theorem sequence_term_equation (n : ℕ+) : sequence_term n = 10 * n - 9 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_equation_l1605_160592


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l1605_160574

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l1605_160574


namespace NUMINAMATH_CALUDE_mixture_ratio_l1605_160524

/-- Given a mixture with initial volume and ratio, and additional water added, 
    calculate the new ratio of components -/
theorem mixture_ratio (initial_volume : ℚ) (milk_ratio water_ratio juice_ratio : ℕ) 
                      (added_water : ℚ) : 
  initial_volume = 60 ∧ 
  milk_ratio = 3 ∧ 
  water_ratio = 2 ∧ 
  juice_ratio = 1 ∧ 
  added_water = 24 →
  ∃ (new_milk new_water new_juice : ℚ),
    new_milk / 2 = 15 ∧
    new_water / 2 = 22 ∧
    new_juice / 2 = 5 ∧
    new_milk + new_water + new_juice = initial_volume + added_water :=
by sorry

end NUMINAMATH_CALUDE_mixture_ratio_l1605_160524


namespace NUMINAMATH_CALUDE_correct_remaining_contents_l1605_160586

/-- Represents the contents of a cup with coffee and milk -/
structure CupContents where
  coffee : ℚ
  milk : ℚ

/-- Calculates the remaining contents in the cup after mixing and removing some mixture -/
def remainingContents (initialCoffee : ℚ) (addedMilk : ℚ) (removedMixture : ℚ) : CupContents :=
  let totalVolume := initialCoffee + addedMilk
  let coffeeRatio := initialCoffee / totalVolume
  let milkRatio := addedMilk / totalVolume
  let remainingVolume := totalVolume - removedMixture
  { coffee := coffeeRatio * remainingVolume,
    milk := milkRatio * remainingVolume }

/-- Theorem stating the correct remaining contents after mixing and removing -/
theorem correct_remaining_contents :
  let result := remainingContents 1 (1/4) (1/4)
  result.coffee = 4/5 ∧ result.milk = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_correct_remaining_contents_l1605_160586


namespace NUMINAMATH_CALUDE_pizza_slice_volume_l1605_160573

/-- The volume of a slice of pizza -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_slices : ℕ) 
  (h1 : thickness = 1/4)
  (h2 : diameter = 16)
  (h3 : num_slices = 8) :
  (π * (diameter/2)^2 * thickness) / num_slices = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_volume_l1605_160573


namespace NUMINAMATH_CALUDE_sum_15_l1605_160527

/-- An arithmetic progression with sum of first n terms S_n -/
structure ArithmeticProgression where
  S : ℕ → ℝ  -- Sum function

/-- The sum of the first 5 terms is 3 -/
axiom sum_5 (ap : ArithmeticProgression) : ap.S 5 = 3

/-- The sum of the first 10 terms is 12 -/
axiom sum_10 (ap : ArithmeticProgression) : ap.S 10 = 12

/-- Theorem: If S_5 = 3 and S_10 = 12, then S_15 = 39 -/
theorem sum_15 (ap : ArithmeticProgression) : ap.S 15 = 39 := by
  sorry


end NUMINAMATH_CALUDE_sum_15_l1605_160527


namespace NUMINAMATH_CALUDE_percentage_85_89_is_40_3_l1605_160566

/-- Represents the frequency distribution of test scores in a class -/
structure ScoreDistribution where
  range_90_100 : Nat
  range_85_89 : Nat
  range_75_84 : Nat
  range_65_74 : Nat
  range_below_65 : Nat

/-- Calculates the percentage of students in a specific score range -/
def percentageInRange (dist : ScoreDistribution) (rangeCount : Nat) : Rat :=
  let totalStudents := dist.range_90_100 + dist.range_85_89 + dist.range_75_84 + 
                       dist.range_65_74 + dist.range_below_65
  (rangeCount : Rat) / (totalStudents : Rat) * 100

/-- The main theorem stating that the percentage of students in the 85%-89% range is 40/3% -/
theorem percentage_85_89_is_40_3 (dist : ScoreDistribution) 
    (h : dist = ScoreDistribution.mk 6 4 7 10 3) : 
    percentageInRange dist dist.range_85_89 = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_85_89_is_40_3_l1605_160566


namespace NUMINAMATH_CALUDE_chinese_english_difference_l1605_160568

/-- The number of hours Ryan spends learning English daily -/
def english_hours : ℕ := 6

/-- The number of hours Ryan spends learning Chinese daily -/
def chinese_hours : ℕ := 7

/-- The difference in hours between Chinese and English learning time -/
def learning_difference : ℕ := chinese_hours - english_hours

theorem chinese_english_difference :
  learning_difference = 1 :=
by sorry

end NUMINAMATH_CALUDE_chinese_english_difference_l1605_160568


namespace NUMINAMATH_CALUDE_sum_47_58_base5_l1605_160599

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_47_58_base5 : toBase5 (47 + 58) = [4, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_sum_47_58_base5_l1605_160599


namespace NUMINAMATH_CALUDE_parabola_perpendicular_range_l1605_160552

/-- Given a parabola y = x^2 with a fixed point A(-1, 1) and two moving points P and Q on the parabola,
    if PA ⊥ PQ, then the x-coordinate of Q is in (-∞, -3] ∪ [1, +∞) -/
theorem parabola_perpendicular_range (a x : ℝ) :
  let P : ℝ × ℝ := (a, a^2)
  let Q : ℝ × ℝ := (x, x^2)
  let A : ℝ × ℝ := (-1, 1)
  (a + 1) * (x - a) + (a^2 - 1) * (x^2 - a^2) = 0 →
  x ≤ -3 ∨ x ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_range_l1605_160552


namespace NUMINAMATH_CALUDE_percentage_calculation_l1605_160546

theorem percentage_calculation (N : ℝ) (P : ℝ) 
  (h1 : N = 125) 
  (h2 : N = (P / 100) * N + 105) : 
  P = 16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1605_160546


namespace NUMINAMATH_CALUDE_function_properties_imply_solution_set_l1605_160542

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def satisfies_negation_property (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

def matches_linear_on_interval (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = (1/2) * x

def solution_set (k : ℤ) : ℝ := 4 * k - 1

theorem function_properties_imply_solution_set 
  (f : ℝ → ℝ) 
  (h1 : is_odd_function f) 
  (h2 : satisfies_negation_property f) 
  (h3 : matches_linear_on_interval f) :
  ∀ x, f x = -(1/2) ↔ ∃ k : ℤ, x = solution_set k :=
sorry

end NUMINAMATH_CALUDE_function_properties_imply_solution_set_l1605_160542


namespace NUMINAMATH_CALUDE_coconut_grove_yield_l1605_160529

/-- Calculates the yield of the third group of trees in a coconut grove --/
theorem coconut_grove_yield (x : ℕ) (Y : ℕ) : x = 6 →
  ((x + 3) * 60 + x * 120 + (x - 3) * Y) / (3 * x) = 100 →
  Y = 180 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_yield_l1605_160529


namespace NUMINAMATH_CALUDE_remainder_4n_squared_mod_13_l1605_160563

theorem remainder_4n_squared_mod_13 (n : ℤ) (h : n % 13 = 7) : (4 * n^2) % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4n_squared_mod_13_l1605_160563


namespace NUMINAMATH_CALUDE_ellipse_param_sum_l1605_160583

/-- An ellipse with given foci and distance sum -/
structure Ellipse where
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  distance_sum : ℝ

/-- Standard form parameters of an ellipse -/
structure EllipseParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Calculate the parameters of the ellipse given its foci and distance sum -/
def calculate_ellipse_params (e : Ellipse) : EllipseParams :=
  sorry

/-- The main theorem to be proved -/
theorem ellipse_param_sum (e : Ellipse) (p : EllipseParams) :
  e.f1 = (0, 1) →
  e.f2 = (6, 1) →
  e.distance_sum = 10 →
  p = calculate_ellipse_params e →
  p.h + p.k + p.a + p.b = 13 :=
sorry

end NUMINAMATH_CALUDE_ellipse_param_sum_l1605_160583


namespace NUMINAMATH_CALUDE_students_speaking_neither_language_l1605_160567

theorem students_speaking_neither_language (total : ℕ) (english : ℕ) (telugu : ℕ) (both : ℕ) :
  total = 150 →
  english = 55 →
  telugu = 85 →
  both = 20 →
  total - (english + telugu - both) = 30 :=
by sorry

end NUMINAMATH_CALUDE_students_speaking_neither_language_l1605_160567


namespace NUMINAMATH_CALUDE_sqrt_squared_2a_minus_1_l1605_160553

theorem sqrt_squared_2a_minus_1 (a : ℝ) (h : a ≥ (1/2 : ℝ)) :
  Real.sqrt ((2*a - 1)^2) = 2*a - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_squared_2a_minus_1_l1605_160553


namespace NUMINAMATH_CALUDE_shopping_trip_proof_l1605_160507

def shopping_trip (initial_amount bag_price lunch_price : ℚ) : Prop :=
  let shoe_price : ℚ := 45
  let remaining : ℚ := 78
  initial_amount = 158 ∧
  bag_price = shoe_price - 17 ∧
  initial_amount = shoe_price + bag_price + lunch_price + remaining ∧
  lunch_price / bag_price = 1/4

theorem shopping_trip_proof : ∃ bag_price lunch_price : ℚ, shopping_trip 158 bag_price lunch_price :=
sorry

end NUMINAMATH_CALUDE_shopping_trip_proof_l1605_160507


namespace NUMINAMATH_CALUDE_mary_bought_14_apples_l1605_160523

/-- The number of apples Mary bought initially -/
def apples : ℕ := sorry

/-- The number of oranges Mary bought initially -/
def oranges : ℕ := 9

/-- The number of blueberries Mary bought initially -/
def blueberries : ℕ := 6

/-- The number of fruits Mary has left after eating one of each -/
def fruits_left : ℕ := 26

/-- Theorem stating that Mary bought 14 apples initially -/
theorem mary_bought_14_apples : 
  apples = 14 ∧ 
  apples + oranges + blueberries - 3 = fruits_left :=
sorry

end NUMINAMATH_CALUDE_mary_bought_14_apples_l1605_160523


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1605_160587

/-- Given a right triangle with legs a and b, if the volume of the cone formed by
    rotating the triangle about leg a is 1000π cm³ and the volume of the cone formed by
    rotating the triangle about leg b is 2430π cm³, then the length of the hypotenuse c
    is approximately 28.12 cm. -/
theorem right_triangle_hypotenuse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / 3 * π * b^2 * a = 1000 * π) →
  (1 / 3 * π * a^2 * b = 2430 * π) →
  abs (Real.sqrt (a^2 + b^2) - 28.12) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1605_160587


namespace NUMINAMATH_CALUDE_last_card_in_box_three_l1605_160500

/-- The number of boxes -/
def num_boxes : ℕ := 7

/-- The total number of cards -/
def total_cards : ℕ := 2015

/-- The length of a complete cycle -/
def cycle_length : ℕ := 12

/-- Function to determine the box number for a given card number -/
def box_number (card : ℕ) : ℕ :=
  let cycle_position := card % cycle_length
  if cycle_position ≤ num_boxes
  then cycle_position
  else 2 * num_boxes - cycle_position

/-- Theorem stating that the 2015th card will be placed in box 3 -/
theorem last_card_in_box_three :
  box_number total_cards = 3 := by
  sorry


end NUMINAMATH_CALUDE_last_card_in_box_three_l1605_160500


namespace NUMINAMATH_CALUDE_inequality_relations_l1605_160572

theorem inequality_relations (a d e : ℝ) 
  (h1 : a < 0) (h2 : a < d) (h3 : d < e) : 
  (a * d < d * e) ∧ 
  (a * e < d * e) ∧ 
  (a + d < d + e) ∧ 
  (e / a < 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relations_l1605_160572


namespace NUMINAMATH_CALUDE_parabola_directrix_l1605_160569

/-- Given a parabola with equation y = -1/4 * x^2, its directrix has the equation y = 1 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = -1/4 * x^2) → (∃ (k : ℝ), k = 1 ∧ k = y + 1/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1605_160569


namespace NUMINAMATH_CALUDE_function_bounds_l1605_160590

theorem function_bounds (x : ℝ) : 
  0.95 ≤ (x^4 + x^2 + 5) / ((x^2 + 1)^2) ∧ (x^4 + x^2 + 5) / ((x^2 + 1)^2) ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_function_bounds_l1605_160590


namespace NUMINAMATH_CALUDE_smallest_common_factor_l1605_160536

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 3 → gcd (12*m - 3) (8*m + 9) = 1) ∧ 
  gcd (12*3 - 3) (8*3 + 9) > 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l1605_160536


namespace NUMINAMATH_CALUDE_mcdonald_farm_eggs_l1605_160525

/-- Calculates the total number of eggs needed per month for a community -/
def total_eggs_per_month (saly_weekly : ℕ) (ben_weekly : ℕ) (weeks_per_month : ℕ) : ℕ :=
  let ked_weekly := ben_weekly / 2
  let total_weekly := saly_weekly + ben_weekly + ked_weekly
  total_weekly * weeks_per_month

/-- Proves that the total eggs needed per month is 124 given the specific requirements -/
theorem mcdonald_farm_eggs : total_eggs_per_month 10 14 4 = 124 := by
  sorry

end NUMINAMATH_CALUDE_mcdonald_farm_eggs_l1605_160525


namespace NUMINAMATH_CALUDE_solution_value_l1605_160503

theorem solution_value (a b : ℝ) (h : a - 2*b = 7) : -a + 2*b + 1 = -6 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1605_160503


namespace NUMINAMATH_CALUDE_xy_minimum_l1605_160588

theorem xy_minimum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1/2) :
  x * y ≥ 16 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 1/y₀ = 1/2 ∧ x₀ * y₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_xy_minimum_l1605_160588


namespace NUMINAMATH_CALUDE_book_reading_time_l1605_160595

theorem book_reading_time (total_books : ℕ) (books_per_week : ℕ) (weeks : ℕ) : 
  total_books = 30 → books_per_week = 6 → weeks * books_per_week = total_books → weeks = 5 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l1605_160595


namespace NUMINAMATH_CALUDE_prime_sum_divides_power_sum_l1605_160540

theorem prime_sum_divides_power_sum (p q : ℕ) : 
  Prime p → Prime q → q = p + 2 → (p + q) ∣ (p^q + q^p) := by
sorry

end NUMINAMATH_CALUDE_prime_sum_divides_power_sum_l1605_160540


namespace NUMINAMATH_CALUDE_inequality_proof_l1605_160556

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) :
  x^2 * y^2 + |x^2 - y^2| ≤ π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1605_160556


namespace NUMINAMATH_CALUDE_handshake_frames_remaining_l1605_160558

theorem handshake_frames_remaining (d₁ d₂ : ℕ) 
  (h₁ : d₁ % 9 = 4) 
  (h₂ : d₂ % 9 = 6) : 
  (d₁ * d₂) % 9 = 6 := by
sorry

end NUMINAMATH_CALUDE_handshake_frames_remaining_l1605_160558


namespace NUMINAMATH_CALUDE_power_function_is_odd_l1605_160505

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (c : ℝ) (α : ℝ), ∀ x, f x = c * x^α

-- Define an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem power_function_is_odd (α : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (α - 2) * x^α
  isPowerFunction f → isOddFunction f := by
  sorry


end NUMINAMATH_CALUDE_power_function_is_odd_l1605_160505


namespace NUMINAMATH_CALUDE_wire_length_for_max_area_circular_sector_l1605_160551

/-- The length of wire needed to create a circular sector with maximum area --/
theorem wire_length_for_max_area_circular_sector (r : ℝ) (h : r = 4) :
  2 * π * r + 2 * r = 8 * π + 8 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_for_max_area_circular_sector_l1605_160551


namespace NUMINAMATH_CALUDE_linear_function_m_range_l1605_160514

/-- A linear function y = (m-1)x + (4m-3) whose graph lies in the first, second, and fourth quadrants -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x + (4 * m - 3)

/-- The slope of the linear function is negative -/
def slope_negative (m : ℝ) : Prop := m - 1 < 0

/-- The y-intercept of the linear function is positive -/
def y_intercept_positive (m : ℝ) : Prop := 4 * m - 3 > 0

/-- The graph of the linear function lies in the first, second, and fourth quadrants -/
def graph_in_first_second_fourth_quadrants (m : ℝ) : Prop :=
  slope_negative m ∧ y_intercept_positive m

theorem linear_function_m_range :
  ∀ m : ℝ, graph_in_first_second_fourth_quadrants m → 3/4 < m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_m_range_l1605_160514

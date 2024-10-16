import Mathlib

namespace NUMINAMATH_CALUDE_framed_painting_ratio_l3209_320959

theorem framed_painting_ratio : 
  ∀ (y : ℝ),
  y > 0 →
  (20 + 2*y) * (30 + 6*y) = 2 * 20 * 30 →
  (min (20 + 2*y) (30 + 6*y)) / (max (20 + 2*y) (30 + 6*y)) = 4/7 :=
by sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l3209_320959


namespace NUMINAMATH_CALUDE_fraction_equality_l3209_320936

theorem fraction_equality (a b : ℚ) (h : (a - b) / b = 2 / 3) : a / b = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3209_320936


namespace NUMINAMATH_CALUDE_apartment_utilities_cost_l3209_320961

/-- Represents the monthly costs and driving distance for an apartment --/
structure Apartment where
  rent : ℝ
  utilities : ℝ
  driveMiles : ℝ

/-- Calculates the total monthly cost for an apartment --/
def totalMonthlyCost (apt : Apartment) (workdays : ℝ) (driveCostPerMile : ℝ) : ℝ :=
  apt.rent + apt.utilities + (apt.driveMiles * workdays * driveCostPerMile)

/-- The problem statement --/
theorem apartment_utilities_cost 
  (apt1 : Apartment)
  (apt2 : Apartment)
  (workdays : ℝ)
  (driveCostPerMile : ℝ)
  (totalCostDifference : ℝ)
  (h1 : apt1.rent = 800)
  (h2 : apt1.utilities = 260)
  (h3 : apt1.driveMiles = 31)
  (h4 : apt2.rent = 900)
  (h5 : apt2.driveMiles = 21)
  (h6 : workdays = 20)
  (h7 : driveCostPerMile = 0.58)
  (h8 : totalMonthlyCost apt1 workdays driveCostPerMile - 
        totalMonthlyCost apt2 workdays driveCostPerMile = totalCostDifference)
  (h9 : totalCostDifference = 76) :
  apt2.utilities = 200 := by
  sorry


end NUMINAMATH_CALUDE_apartment_utilities_cost_l3209_320961


namespace NUMINAMATH_CALUDE_stream_current_rate_l3209_320920

/-- The rate of the stream's current in miles per hour -/
def w : ℝ := 3

/-- The man's rowing speed in still water in miles per hour -/
def r : ℝ := 6

/-- The distance traveled downstream and upstream in miles -/
def d : ℝ := 18

/-- Theorem stating that given the conditions, the stream's current is 3 mph -/
theorem stream_current_rate : 
  (d / (r + w) + 4 = d / (r - w)) ∧ 
  (d / (3 * r + w) + 2 = d / (3 * r - w)) → 
  w = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_current_rate_l3209_320920


namespace NUMINAMATH_CALUDE_zara_age_l3209_320928

def guesses : List Nat := [26, 29, 31, 34, 37, 39, 42, 45, 47, 50, 52]

def is_prime (n : Nat) : Prop := Nat.Prime n

def more_than_half_low (age : Nat) : Prop :=
  (guesses.filter (· < age)).length > guesses.length / 2

def three_off_by_one (age : Nat) : Prop :=
  (guesses.filter (fun x => x = age - 1 ∨ x = age + 1)).length = 3

theorem zara_age : ∃! age : Nat, 
  age ∈ guesses ∧
  is_prime age ∧
  more_than_half_low age ∧
  three_off_by_one age ∧
  age = 47 :=
sorry

end NUMINAMATH_CALUDE_zara_age_l3209_320928


namespace NUMINAMATH_CALUDE_intersection_determinant_l3209_320903

theorem intersection_determinant (a : ℝ) :
  (∃! p : ℝ × ℝ, a * p.1 + p.2 + 3 = 0 ∧ p.1 + p.2 + 2 = 0 ∧ 2 * p.1 - p.2 + 1 = 0) →
  Matrix.det !![a, 1, 3; 1, 1, 2; 2, -1, 1] = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_determinant_l3209_320903


namespace NUMINAMATH_CALUDE_pet_store_cats_count_l3209_320977

theorem pet_store_cats_count (siamese : Float) (house : Float) (added : Float) :
  siamese = 13.0 → house = 5.0 → added = 10.0 →
  siamese + house + added = 28.0 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cats_count_l3209_320977


namespace NUMINAMATH_CALUDE_probability_three_odd_dice_l3209_320980

theorem probability_three_odd_dice (n : ℕ) (p : ℝ) : 
  n = 5 →                          -- number of dice
  p = 1 / 2 →                      -- probability of rolling an odd number on a single die
  (Nat.choose n 3 : ℝ) * p^3 * (1 - p)^(n - 3) = 5 / 16 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_odd_dice_l3209_320980


namespace NUMINAMATH_CALUDE_count_sequences_100_l3209_320945

/-- The number of sequences of length n, where each sequence contains at least one 4 or 5,
    and any two consecutive members differ by no more than 2. -/
def count_sequences (n : ℕ) : ℕ :=
  5^n - 3^n

/-- The theorem stating that the number of valid sequences of length 100 is 5^100 - 3^100. -/
theorem count_sequences_100 :
  count_sequences 100 = 5^100 - 3^100 :=
by sorry

end NUMINAMATH_CALUDE_count_sequences_100_l3209_320945


namespace NUMINAMATH_CALUDE_anderson_pet_food_weight_l3209_320994

/-- Calculates the total weight of pet food in ounces -/
def total_pet_food_ounces (cat_food_bags : ℕ) (cat_food_weight : ℕ) 
                          (dog_food_bags : ℕ) (dog_food_extra_weight : ℕ) 
                          (ounces_per_pound : ℕ) : ℕ :=
  let total_cat_food := cat_food_bags * cat_food_weight
  let dog_food_weight := cat_food_weight + dog_food_extra_weight
  let total_dog_food := dog_food_bags * dog_food_weight
  let total_weight := total_cat_food + total_dog_food
  total_weight * ounces_per_pound

/-- Theorem: The total weight of pet food Mrs. Anderson bought is 256 ounces -/
theorem anderson_pet_food_weight : 
  total_pet_food_ounces 2 3 2 2 16 = 256 := by
  sorry

end NUMINAMATH_CALUDE_anderson_pet_food_weight_l3209_320994


namespace NUMINAMATH_CALUDE_log_inequality_relation_l3209_320925

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_inequality_relation :
  (∀ x y : ℝ, x > 0 → y > 0 → (log x < log y → x < y)) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x < y ∧ ¬(log x < log y)) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_relation_l3209_320925


namespace NUMINAMATH_CALUDE_single_tile_replacement_impossible_l3209_320949

/-- Represents the two types of tiles used for paving -/
inductive TileType
  | Rectangle4x1
  | Square2x2

/-- Represents a rectangular floor -/
structure Floor :=
  (width : ℕ)
  (height : ℕ)
  (tiling : List (TileType))

/-- Checks if a tiling is valid for the given floor -/
def is_valid_tiling (floor : Floor) : Prop :=
  sorry

/-- Represents the operation of replacing a single tile -/
def replace_single_tile (floor : Floor) (old_type new_type : TileType) : Floor :=
  sorry

/-- The main theorem stating that replacing a single tile
    with a different type always results in an invalid tiling -/
theorem single_tile_replacement_impossible (floor : Floor) :
  ∀ (old_type new_type : TileType),
    old_type ≠ new_type →
    is_valid_tiling floor →
    ¬(is_valid_tiling (replace_single_tile floor old_type new_type)) :=
  sorry

end NUMINAMATH_CALUDE_single_tile_replacement_impossible_l3209_320949


namespace NUMINAMATH_CALUDE_withdrawal_amount_l3209_320960

def initial_balance : ℕ := 65
def deposit : ℕ := 15
def final_balance : ℕ := 76

theorem withdrawal_amount : 
  initial_balance + deposit - final_balance = 4 := by
  sorry

end NUMINAMATH_CALUDE_withdrawal_amount_l3209_320960


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3209_320993

-- Problem 1
theorem problem_1 (m n : ℝ) : 2 * m * n^2 * (1/4 * m * n) = 1/2 * m^2 * n^3 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : (2 * a^3 * b^2 + a^2 * b) / (a * b) = 2 * a^2 * b + a := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) : (2 * x + 3) * (x - 1) = 2 * x^2 + x - 3 := by sorry

-- Problem 4
theorem problem_4 (x y : ℝ) : (x + y)^2 - 2 * y * (x - y) = x^2 + 3 * y^2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3209_320993


namespace NUMINAMATH_CALUDE_two_numbers_sum_65_l3209_320906

theorem two_numbers_sum_65 
  (S : Finset ℕ) 
  (A B : Finset ℕ) 
  (hS : S = Finset.range 64) 
  (hA : A ⊆ S) 
  (hB : B ⊆ S) 
  (hAcard : A.card = 16) 
  (hBcard : B.card = 16) 
  (hAodd : ∀ a ∈ A, Odd a) 
  (hBeven : ∀ b ∈ B, Even b) 
  (hsum : A.sum id = B.sum id) : 
  ∃ (x y : ℕ), x ∈ A ∪ B ∧ y ∈ A ∪ B ∧ x + y = 65 :=
sorry

end NUMINAMATH_CALUDE_two_numbers_sum_65_l3209_320906


namespace NUMINAMATH_CALUDE_min_value_of_b_is_negative_two_l3209_320966

/-- The function that represents b in terms of a, where y = 2x + b is a tangent line to y = a ln x --/
noncomputable def b (a : ℝ) : ℝ := a * Real.log (a / 2) - a

/-- The theorem stating that the minimum value of b is -2 when a > 0 --/
theorem min_value_of_b_is_negative_two :
  ∀ a : ℝ, a > 0 → (∀ x : ℝ, x > 0 → b x ≥ b 2) ∧ b 2 = -2 := by sorry

end NUMINAMATH_CALUDE_min_value_of_b_is_negative_two_l3209_320966


namespace NUMINAMATH_CALUDE_complex_square_root_expression_l3209_320912

theorem complex_square_root_expression : 71 * Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_expression_l3209_320912


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3209_320937

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3209_320937


namespace NUMINAMATH_CALUDE_adult_meals_count_l3209_320985

/-- The number of meals that can feed children -/
def childMeals : ℕ := 90

/-- The number of adults who have their meal -/
def adultsMealed : ℕ := 35

/-- The number of children that can be fed with remaining food after some adults eat -/
def remainingChildMeals : ℕ := 45

/-- The number of meals initially available for adults -/
def adultMeals : ℕ := 80

theorem adult_meals_count :
  adultMeals = childMeals - remainingChildMeals + adultsMealed :=
by sorry

end NUMINAMATH_CALUDE_adult_meals_count_l3209_320985


namespace NUMINAMATH_CALUDE_safe_code_exists_l3209_320952

def is_valid_code (n : ℕ) : Prop :=
  (n ≥ 1000000 ∧ n ≤ 9999999) ∧  -- seven-digit number
  (∀ d, d ∈ n.digits 10 → d = 2 ∨ d = 3) ∧  -- digits are 2 or 3
  ((n.digits 10).filter (· = 2)).length > ((n.digits 10).filter (· = 3)).length ∧  -- more 2s than 3s
  n % 3 = 0 ∧ n % 4 = 0  -- divisible by 3 and 4

theorem safe_code_exists : ∃ n : ℕ, is_valid_code n :=
sorry

end NUMINAMATH_CALUDE_safe_code_exists_l3209_320952


namespace NUMINAMATH_CALUDE_xiao_ming_test_average_l3209_320941

theorem xiao_ming_test_average (first_two_avg : ℝ) (last_three_total : ℝ) :
  first_two_avg = 85 →
  last_three_total = 270 →
  (2 * first_two_avg + last_three_total) / 5 = 88 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_test_average_l3209_320941


namespace NUMINAMATH_CALUDE_equivalent_representations_l3209_320999

theorem equivalent_representations : 
  (16 : ℚ) / 20 = 24 / 30 ∧ 
  (16 : ℚ) / 20 = 80 / 100 ∧ 
  (16 : ℚ) / 20 = 4 / 5 ∧ 
  (16 : ℚ) / 20 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_representations_l3209_320999


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3209_320926

/-- An arithmetic sequence {a_n} where a_1 = 1/3, a_2 + a_5 = 4, and a_n = 33 has n = 50 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) (n : ℕ) 
  (h_arith : ∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) 
  (h_a1 : a 1 = 1/3)
  (h_sum : a 2 + a 5 = 4)
  (h_an : a n = 33) :
  n = 50 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3209_320926


namespace NUMINAMATH_CALUDE_bill_toys_count_l3209_320990

/-- The number of toys Bill and Hash have together -/
def total_toys : ℕ := 99

/-- The number of toys Bill has -/
def bill_toys : ℕ := 60

/-- The number of toys Hash has -/
def hash_toys : ℕ := total_toys - bill_toys

theorem bill_toys_count :
  (hash_toys = bill_toys / 2 + 9) ∧ (bill_toys + hash_toys = total_toys) →
  bill_toys = 60 := by
  sorry

end NUMINAMATH_CALUDE_bill_toys_count_l3209_320990


namespace NUMINAMATH_CALUDE_planted_field_fraction_l3209_320983

theorem planted_field_fraction (a b c x : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_legs : a = 6 ∧ b = 8) (h_distance : (a - 0.6*x) * (b - 0.8*x) / 2 = 3) :
  (a * b / 2 - x^2) / (a * b / 2) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_planted_field_fraction_l3209_320983


namespace NUMINAMATH_CALUDE_ben_walking_time_l3209_320900

/-- Given that Ben walks at a constant speed and covers 3 km in 2 hours,
    prove that the time required to walk 12 km is 480 minutes. -/
theorem ben_walking_time (speed : ℝ) (h1 : speed > 0) : 
  (3 : ℝ) / speed = 2 → (12 : ℝ) / speed * 60 = 480 := by
sorry

end NUMINAMATH_CALUDE_ben_walking_time_l3209_320900


namespace NUMINAMATH_CALUDE_tommy_steaks_l3209_320910

/-- The number of steaks needed for a family dinner -/
def steaks_needed (family_members : ℕ) (pounds_per_member : ℕ) (ounces_per_steak : ℕ) : ℕ :=
  let total_ounces := family_members * pounds_per_member * 16
  (total_ounces + ounces_per_steak - 1) / ounces_per_steak

/-- Theorem: Tommy needs to buy 4 steaks for his family -/
theorem tommy_steaks : steaks_needed 5 1 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tommy_steaks_l3209_320910


namespace NUMINAMATH_CALUDE_second_encounter_correct_l3209_320950

/-- Represents the highway with speed limit signs and monitoring devices -/
structure Highway where
  speed_limit_start : ℕ := 3
  speed_limit_interval : ℕ := 4
  monitoring_start : ℕ := 10
  monitoring_interval : ℕ := 9
  first_encounter : ℕ := 19

/-- The kilometer mark of the second simultaneous encounter -/
def second_encounter (h : Highway) : ℕ := 55

/-- Theorem stating that the second encounter occurs at 55 km -/
theorem second_encounter_correct (h : Highway) : 
  second_encounter h = 55 := by sorry

end NUMINAMATH_CALUDE_second_encounter_correct_l3209_320950


namespace NUMINAMATH_CALUDE_river_joe_collection_l3209_320918

/-- Represents the total money collected by River Joe's Seafood Diner --/
def total_money_collected (catfish_price popcorn_shrimp_price : ℚ) 
  (total_orders popcorn_shrimp_orders : ℕ) : ℚ :=
  let catfish_orders := total_orders - popcorn_shrimp_orders
  catfish_price * catfish_orders + popcorn_shrimp_price * popcorn_shrimp_orders

/-- Proves that River Joe collected $133.50 given the specified conditions --/
theorem river_joe_collection : 
  total_money_collected 6 3.5 26 9 = 133.5 := by
  sorry

#eval total_money_collected 6 3.5 26 9

end NUMINAMATH_CALUDE_river_joe_collection_l3209_320918


namespace NUMINAMATH_CALUDE_total_pages_in_paper_l3209_320986

/-- Represents the number of days Stacy has to complete her history paper -/
def days_to_complete : ℕ := 7

/-- Represents the number of pages Stacy needs to write per day -/
def pages_per_day : ℕ := 9

/-- Theorem stating that the total number of pages in Stacy's history paper is 63 -/
theorem total_pages_in_paper : days_to_complete * pages_per_day = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_in_paper_l3209_320986


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l3209_320997

theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 30 →
  length = 3 * breadth →
  area = length * breadth →
  area = 2700 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l3209_320997


namespace NUMINAMATH_CALUDE_only_courses_form_set_l3209_320929

-- Define a type for the universe of discourse
def Universe : Type := Unit

-- Define predicates for each option
def likes_airplanes (x : Universe) : Prop := sorry
def is_sufficiently_small_negative (x : ℝ) : Prop := sorry
def has_poor_eyesight (x : Universe) : Prop := sorry
def is_course_of_class_on_day (x : Universe) : Prop := sorry

-- Define what it means for a predicate to form a well-defined set
def forms_well_defined_set {α : Type} (P : α → Prop) : Prop := sorry

-- State the theorem
theorem only_courses_form_set :
  ¬(forms_well_defined_set likes_airplanes) ∧
  ¬(forms_well_defined_set is_sufficiently_small_negative) ∧
  ¬(forms_well_defined_set has_poor_eyesight) ∧
  (forms_well_defined_set is_course_of_class_on_day) :=
sorry

end NUMINAMATH_CALUDE_only_courses_form_set_l3209_320929


namespace NUMINAMATH_CALUDE_second_person_speed_l3209_320996

/-- Given two people traveling between points A and B, prove the speed of the second person. -/
theorem second_person_speed 
  (distance : ℝ) 
  (speed_first : ℝ) 
  (travel_time : ℝ) 
  (h1 : distance = 600) 
  (h2 : speed_first = 70) 
  (h3 : travel_time = 4) : 
  ∃ speed_second : ℝ, speed_second = 80 ∧ 
  speed_first * travel_time + speed_second * travel_time = distance :=
by
  sorry

#check second_person_speed

end NUMINAMATH_CALUDE_second_person_speed_l3209_320996


namespace NUMINAMATH_CALUDE_sum_a_b_equals_14_l3209_320947

theorem sum_a_b_equals_14 (a b c d : ℝ) 
  (h1 : b + c = 9) 
  (h2 : c + d = 3) 
  (h3 : a + d = 8) : 
  a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_14_l3209_320947


namespace NUMINAMATH_CALUDE_jason_average_messages_l3209_320930

/-- The average number of text messages sent over five days -/
def average_messages (monday : ℕ) (tuesday : ℕ) (wed_to_fri : ℕ) (days : ℕ) : ℚ :=
  (monday + tuesday + 3 * wed_to_fri : ℚ) / days

theorem jason_average_messages :
  let monday := 220
  let tuesday := monday / 2
  let wed_to_fri := 50
  let days := 5
  average_messages monday tuesday wed_to_fri days = 96 := by
sorry

end NUMINAMATH_CALUDE_jason_average_messages_l3209_320930


namespace NUMINAMATH_CALUDE_larger_number_proof_l3209_320958

theorem larger_number_proof (L S : ℕ) (hL : L > S) :
  L - S = 1365 → L = 6 * S + 15 → L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3209_320958


namespace NUMINAMATH_CALUDE_derivative_value_at_one_l3209_320923

noncomputable section

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

def is_derivative (f f' : ℝ → ℝ) : Prop :=
  ∀ x, HasDerivAt f (f' x) x

theorem derivative_value_at_one :
  (∀ x, f x = (f' 1) * x^3 - 2 * x^2 + 3) →
  is_derivative f f' →
  f' 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_derivative_value_at_one_l3209_320923


namespace NUMINAMATH_CALUDE_extreme_value_and_inequality_l3209_320955

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x + 1)

theorem extreme_value_and_inequality (a : ℝ) :
  (∃ x, f x = -1 ∧ ∀ y, f y ≥ f x) ∧
  (∀ x > 0, f x ≥ x + Real.log x + a + 1) ↔ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_and_inequality_l3209_320955


namespace NUMINAMATH_CALUDE_inequality_proof_l3209_320905

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  let p := x + y + z
  let q := x*y + y*z + z*x
  let r := x*y*z
  (p^2 ≥ 3*q) ∧
  (p^3 ≥ 27*r) ∧
  (p*q ≥ 9*r) ∧
  (q^2 ≥ 3*p*r) ∧
  (p^2*q + 3*p*r ≥ 4*q^2) ∧
  (p^3 + 9*r ≥ 4*p*q) ∧
  (p*q^2 ≥ 2*p^2*r + 3*q*r) ∧
  (p*q^2 + 3*q*r ≥ 4*p^2*r) ∧
  (2*q^3 + 9*r^2 ≥ 7*p*q*r) ∧
  (p^4 + 4*q^2 + 6*p*r ≥ 5*p^2*q) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3209_320905


namespace NUMINAMATH_CALUDE_odd_product_minus_one_divisible_by_four_l3209_320969

theorem odd_product_minus_one_divisible_by_four (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) : 
  (4 ∣ a * b - 1) ∨ (4 ∣ b * c - 1) ∨ (4 ∣ c * a - 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_product_minus_one_divisible_by_four_l3209_320969


namespace NUMINAMATH_CALUDE_circle_center_is_three_halves_thirty_seven_fourths_l3209_320991

/-- A circle passes through (0, 9) and is tangent to y = x^2 at (3, 9) -/
def CircleTangentToParabola (center : ℝ × ℝ) : Prop :=
  let (a, b) := center
  -- Circle passes through (0, 9)
  (a^2 + (b - 9)^2 = a^2 + (b - 9)^2) ∧
  -- Circle is tangent to y = x^2 at (3, 9)
  ((a - 3)^2 + (b - 9)^2 = (a - 0)^2 + (b - 9)^2) ∧
  -- Tangent line to parabola at (3, 9) is perpendicular to line from (3, 9) to center
  ((b - 9) / (a - 3) = -1 / (2 * 3))

/-- The center of the circle is (3/2, 37/4) -/
theorem circle_center_is_three_halves_thirty_seven_fourths :
  CircleTangentToParabola (3/2, 37/4) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_is_three_halves_thirty_seven_fourths_l3209_320991


namespace NUMINAMATH_CALUDE_tour_budget_l3209_320992

/-- Given a tour scenario, proves that the total budget for the original tour is 360 units -/
theorem tour_budget (original_days : ℕ) (extension_days : ℕ) (expense_reduction : ℕ) : 
  original_days = 20 → 
  extension_days = 4 → 
  expense_reduction = 3 →
  (original_days * (original_days + extension_days)) / extension_days = 360 :=
by
  sorry

#check tour_budget

end NUMINAMATH_CALUDE_tour_budget_l3209_320992


namespace NUMINAMATH_CALUDE_diane_stamp_arrangements_l3209_320967

/-- Represents a collection of stamps with their quantities -/
def StampCollection := List (Nat × Nat)

/-- Represents an arrangement of stamps -/
def StampArrangement := List Nat

/-- Returns true if the arrangement sums to the target value -/
def isValidArrangement (arrangement : StampArrangement) (target : Nat) : Bool :=
  arrangement.sum = target

/-- Returns true if the arrangement is possible given the stamp collection -/
def isPossibleArrangement (arrangement : StampArrangement) (collection : StampCollection) : Bool :=
  sorry

/-- Counts the number of unique arrangements given a stamp collection and target sum -/
def countUniqueArrangements (collection : StampCollection) (target : Nat) : Nat :=
  sorry

/-- Diane's stamp collection -/
def dianeCollection : StampCollection :=
  [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]

theorem diane_stamp_arrangements :
  countUniqueArrangements dianeCollection 12 = 30 := by sorry

end NUMINAMATH_CALUDE_diane_stamp_arrangements_l3209_320967


namespace NUMINAMATH_CALUDE_ratio_equality_l3209_320976

theorem ratio_equality (x y z : ℝ) (h : x / 3 = y / 4 ∧ y / 4 = z / 5) :
  (x + y - z) / (2 * x - y + z) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3209_320976


namespace NUMINAMATH_CALUDE_minimum_excellent_all_exams_l3209_320948

theorem minimum_excellent_all_exams (total_students : ℕ) 
  (excellent_first : ℕ) (excellent_second : ℕ) (excellent_third : ℕ) 
  (h_total : total_students = 200)
  (h_first : excellent_first = (80 : ℝ) / 100 * total_students)
  (h_second : excellent_second = (70 : ℝ) / 100 * total_students)
  (h_third : excellent_third = (59 : ℝ) / 100 * total_students) :
  ∃ (excellent_all : ℕ), 
    excellent_all ≥ 18 ∧ 
    (∀ (n : ℕ), n < excellent_all → 
      ∃ (m1 m2 m3 m12 m13 m23 : ℕ),
        m1 + m2 + m3 + m12 + m13 + m23 + n > total_students ∨
        m1 + m12 + m13 + n > excellent_first ∨
        m2 + m12 + m23 + n > excellent_second ∨
        m3 + m13 + m23 + n > excellent_third) :=
sorry

end NUMINAMATH_CALUDE_minimum_excellent_all_exams_l3209_320948


namespace NUMINAMATH_CALUDE_arcsin_plus_arccos_eq_pi_sixth_l3209_320968

theorem arcsin_plus_arccos_eq_pi_sixth (x : ℝ) :
  Real.arcsin x + Real.arccos (3 * x) = π / 6 → x = Real.sqrt (3 / 124) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_plus_arccos_eq_pi_sixth_l3209_320968


namespace NUMINAMATH_CALUDE_first_day_of_month_l3209_320984

/-- Days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after n days -/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => dayAfter (nextDay d) n

/-- Theorem: If the 30th day of a month is a Wednesday, then the 1st day of that month is a Tuesday -/
theorem first_day_of_month (d : DayOfWeek) : 
  dayAfter d 29 = DayOfWeek.Wednesday → d = DayOfWeek.Tuesday := by
  sorry


end NUMINAMATH_CALUDE_first_day_of_month_l3209_320984


namespace NUMINAMATH_CALUDE_factor_expression_l3209_320970

theorem factor_expression (x : ℝ) : 18 * x^2 + 9 * x - 3 = 3 * (6 * x^2 + 3 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3209_320970


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l3209_320933

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 10000000 ∧ n ≤ 99999999) ∧
  (∃ (a b c d : ℕ), 
    a + b + c + d = 12 ∧
    List.count 4 (Nat.digits 10 n) = 2 ∧
    List.count 0 (Nat.digits 10 n) = 2 ∧
    List.count 2 (Nat.digits 10 n) = 2 ∧
    List.count 6 (Nat.digits 10 n) = 2)

def largest_valid_number : ℕ := 66442200
def smallest_valid_number : ℕ := 20024466

theorem sum_of_largest_and_smallest :
  is_valid_number largest_valid_number ∧
  is_valid_number smallest_valid_number ∧
  (∀ n : ℕ, is_valid_number n → n ≤ largest_valid_number) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ smallest_valid_number) ∧
  largest_valid_number + smallest_valid_number = 86466666 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l3209_320933


namespace NUMINAMATH_CALUDE_weight_replacement_l3209_320987

theorem weight_replacement (n : ℕ) (avg_increase weight_new : ℝ) :
  n = 9 ∧ 
  avg_increase = 5.5 ∧
  weight_new = 135.5 →
  (n * avg_increase + weight_new - n * avg_increase) = 86 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l3209_320987


namespace NUMINAMATH_CALUDE_difference_divisible_by_nine_l3209_320989

theorem difference_divisible_by_nine (a b : ℤ) :
  ∃ k : ℤ, (3*a + 2)^2 - (3*b + 2)^2 = 9*k := by
  sorry

end NUMINAMATH_CALUDE_difference_divisible_by_nine_l3209_320989


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3209_320979

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a_2 + a_4 = 20 and a_3 + a_5 = 40, then q = 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 : a 2 + a 4 = 20) 
  (h3 : a 3 + a 5 = 40) : 
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3209_320979


namespace NUMINAMATH_CALUDE_binomial_divisibility_theorem_l3209_320982

theorem binomial_divisibility_theorem (k : ℕ) (h : k > 1) :
  ∃ n : ℕ, n > 0 ∧ 
    (n ∣ Nat.choose n k) ∧ 
    (∀ m : ℕ, 2 ≤ m → m < k → ¬(n ∣ Nat.choose n m)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_theorem_l3209_320982


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3209_320938

theorem trigonometric_inequality (A B C : Real) (h : A + B + C = Real.pi) :
  (Real.cos (A - B))^2 + (Real.cos (B - C))^2 + (Real.cos (C - A))^2 ≥ 
  24 * Real.cos A * Real.cos B * Real.cos C :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3209_320938


namespace NUMINAMATH_CALUDE_equation_solution_l3209_320931

theorem equation_solution (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ -1) :
  (x / (x - 1) = 4 / (x^2 - 1) + 1) ↔ (x = 3) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3209_320931


namespace NUMINAMATH_CALUDE_stickers_distribution_l3209_320962

/-- Calculates the number of stickers each of the other students received -/
def stickers_per_other_student (total_stickers : ℕ) (friends : ℕ) (stickers_per_friend : ℕ) 
  (leftover_stickers : ℕ) (total_students : ℕ) : ℕ :=
  let stickers_given_to_friends := friends * stickers_per_friend
  let total_stickers_given := total_stickers - leftover_stickers
  let stickers_for_others := total_stickers_given - stickers_given_to_friends
  let other_students := total_students - 1 - friends
  stickers_for_others / other_students

theorem stickers_distribution (total_stickers : ℕ) (friends : ℕ) (stickers_per_friend : ℕ) 
  (leftover_stickers : ℕ) (total_students : ℕ)
  (h1 : total_stickers = 50)
  (h2 : friends = 5)
  (h3 : stickers_per_friend = 4)
  (h4 : leftover_stickers = 8)
  (h5 : total_students = 17) :
  stickers_per_other_student total_stickers friends stickers_per_friend leftover_stickers total_students = 2 := by
  sorry

end NUMINAMATH_CALUDE_stickers_distribution_l3209_320962


namespace NUMINAMATH_CALUDE_remainder_101_37_mod_100_l3209_320954

theorem remainder_101_37_mod_100 : 101^37 ≡ 1 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_37_mod_100_l3209_320954


namespace NUMINAMATH_CALUDE_cannot_obtain_123_l3209_320939

/-- Represents an arithmetic expression using numbers 1, 2, 3, 4, 5 and operations +, -, * -/
inductive Expr
| Num : Fin 5 → Expr
| Add : Expr → Expr → Expr
| Sub : Expr → Expr → Expr
| Mul : Expr → Expr → Expr

/-- Evaluates an arithmetic expression -/
def eval : Expr → Int
| Expr.Num n => n.val.succ
| Expr.Add e1 e2 => eval e1 + eval e2
| Expr.Sub e1 e2 => eval e1 - eval e2
| Expr.Mul e1 e2 => eval e1 * eval e2

/-- Theorem stating that it's impossible to obtain 123 using the given constraints -/
theorem cannot_obtain_123 : ¬ ∃ e : Expr, eval e = 123 := by
  sorry

end NUMINAMATH_CALUDE_cannot_obtain_123_l3209_320939


namespace NUMINAMATH_CALUDE_larger_number_is_eleven_l3209_320946

theorem larger_number_is_eleven (x y : ℝ) (h1 : y - x = 2) (h2 : x + y = 20) : 
  max x y = 11 := by
sorry

end NUMINAMATH_CALUDE_larger_number_is_eleven_l3209_320946


namespace NUMINAMATH_CALUDE_running_speed_calculation_l3209_320981

def walking_speed : ℝ := 4
def total_distance : ℝ := 20
def total_time : ℝ := 3.75

theorem running_speed_calculation (R : ℝ) :
  (total_distance / 2 / walking_speed) + (total_distance / 2 / R) = total_time →
  R = 8 := by
  sorry

end NUMINAMATH_CALUDE_running_speed_calculation_l3209_320981


namespace NUMINAMATH_CALUDE_equality_transitivity_add_polynomial_to_equation_l3209_320978

-- Statement 1: Transitivity of equality
theorem equality_transitivity (a b c : ℝ) (h1 : a = b) (h2 : b = c) : a = c := by
  sorry

-- Statement 5: Adding a polynomial to both sides of an equation
theorem add_polynomial_to_equation (f g p : ℝ → ℝ) (h : ∀ x, f x = g x) : 
  ∀ x, f x + p x = g x + p x := by
  sorry

end NUMINAMATH_CALUDE_equality_transitivity_add_polynomial_to_equation_l3209_320978


namespace NUMINAMATH_CALUDE_total_boxes_eq_sum_l3209_320963

/-- The total number of boxes Kaylee needs to sell -/
def total_boxes : ℕ := sorry

/-- The number of lemon biscuit boxes sold -/
def lemon_boxes : ℕ := 12

/-- The number of chocolate biscuit boxes sold -/
def chocolate_boxes : ℕ := 5

/-- The number of oatmeal biscuit boxes sold -/
def oatmeal_boxes : ℕ := 4

/-- The additional number of boxes Kaylee needs to sell -/
def additional_boxes : ℕ := 12

/-- Theorem stating that the total number of boxes is equal to the sum of all sold boxes and additional boxes -/
theorem total_boxes_eq_sum :
  total_boxes = lemon_boxes + chocolate_boxes + oatmeal_boxes + additional_boxes := by
  sorry

end NUMINAMATH_CALUDE_total_boxes_eq_sum_l3209_320963


namespace NUMINAMATH_CALUDE_equation_solution_l3209_320916

theorem equation_solution (x : ℝ) : (40 / 80 = Real.sqrt (x / 80)) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3209_320916


namespace NUMINAMATH_CALUDE_phones_left_theorem_l3209_320974

/-- Calculates the number of phones left in the factory after doubling production and selling a quarter --/
def phones_left_in_factory (last_year_production : ℕ) : ℕ :=
  let this_year_production := 2 * last_year_production
  let sold_phones := this_year_production / 4
  this_year_production - sold_phones

/-- Theorem stating that given last year's production of 5000 phones, 
    if this year's production is doubled and a quarter of it is sold, 
    then the number of phones left in the factory is 7500 --/
theorem phones_left_theorem : phones_left_in_factory 5000 = 7500 := by
  sorry

end NUMINAMATH_CALUDE_phones_left_theorem_l3209_320974


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l3209_320973

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from a pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  (area pan.panDimensions) / (area pan.pieceDimensions)

theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 24, width := 15 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 60 := by sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l3209_320973


namespace NUMINAMATH_CALUDE_factor_63x_plus_54_l3209_320956

theorem factor_63x_plus_54 : ∀ x : ℝ, 63 * x + 54 = 9 * (7 * x + 6) := by
  sorry

end NUMINAMATH_CALUDE_factor_63x_plus_54_l3209_320956


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l3209_320935

theorem largest_solution_of_equation (a b c d : ℤ) (x : ℝ) :
  (4 * x / 5 - 2 = 5 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (∀ y, (4 * y / 5 - 2 = 5 / y) → y ≤ x) →
  (x = (5 + 5 * Real.sqrt 5) / 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l3209_320935


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3209_320911

theorem fractional_equation_solution :
  ∃ x : ℝ, (2 * x) / (x - 3) = 1 ∧ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3209_320911


namespace NUMINAMATH_CALUDE_flag_designs_count_l3209_320921

/-- The number of colors available for the flag design -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The number of different flag designs possible -/
def num_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the number of different flag designs is 27 -/
theorem flag_designs_count : num_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_designs_count_l3209_320921


namespace NUMINAMATH_CALUDE_bc_fraction_of_ad_l3209_320943

-- Define the points
variable (A B C D : ℝ)

-- Define the conditions
axiom on_line_segment : B ≤ A ∧ B ≤ D ∧ C ≤ A ∧ C ≤ D

-- Define the length relationships
axiom length_AB : A - B = 3 * (D - B)
axiom length_AC : A - C = 7 * (D - C)

-- Theorem to prove
theorem bc_fraction_of_ad : (C - B) = (1/8) * (A - D) := by sorry

end NUMINAMATH_CALUDE_bc_fraction_of_ad_l3209_320943


namespace NUMINAMATH_CALUDE_exactly_two_pass_probability_l3209_320988

def prob_A : ℚ := 4/5
def prob_B : ℚ := 3/4
def prob_C : ℚ := 3/4

theorem exactly_two_pass_probability : 
  let prob_AB := prob_A * prob_B * (1 - prob_C)
  let prob_AC := prob_A * (1 - prob_B) * prob_C
  let prob_BC := (1 - prob_A) * prob_B * prob_C
  prob_AB + prob_AC + prob_BC = 33/80 :=
by sorry

end NUMINAMATH_CALUDE_exactly_two_pass_probability_l3209_320988


namespace NUMINAMATH_CALUDE_f_prime_at_zero_l3209_320913

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) * Real.exp x

theorem f_prime_at_zero : 
  (deriv f) 0 = 3 := by sorry

end NUMINAMATH_CALUDE_f_prime_at_zero_l3209_320913


namespace NUMINAMATH_CALUDE_inverse_proportion_point_ordering_l3209_320995

/-- Given three points A(-3, y₁), B(-2, y₂), C(3, y₃) on the graph of y = -2/x,
    prove that y₃ < y₁ < y₂ -/
theorem inverse_proportion_point_ordering (y₁ y₂ y₃ : ℝ) : 
  y₁ = -2 / (-3) → y₂ = -2 / (-2) → y₃ = -2 / 3 → y₃ < y₁ ∧ y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_ordering_l3209_320995


namespace NUMINAMATH_CALUDE_soccer_league_teams_l3209_320922

/-- The number of teams in the soccer league -/
def n : ℕ := 9

/-- The total number of matches played in the league -/
def total_matches : ℕ := 36

/-- Theorem stating that the number of teams is correct given the conditions -/
theorem soccer_league_teams :
  n * (n - 1) / 2 = total_matches :=
by sorry

end NUMINAMATH_CALUDE_soccer_league_teams_l3209_320922


namespace NUMINAMATH_CALUDE_ben_spending_l3209_320902

-- Define the prices and quantities
def apple_price : ℚ := 2
def apple_quantity : ℕ := 7
def milk_price : ℚ := 4
def milk_quantity : ℕ := 4
def bread_price : ℚ := 3
def bread_quantity : ℕ := 3
def sugar_price : ℚ := 6
def sugar_quantity : ℕ := 3

-- Define the discounts
def dairy_discount : ℚ := 0.25
def coupon_discount : ℚ := 10
def coupon_threshold : ℚ := 50

-- Define the total spending function
def total_spending : ℚ :=
  let apple_cost := apple_price * apple_quantity
  let milk_cost := milk_price * milk_quantity * (1 - dairy_discount)
  let bread_cost := bread_price * bread_quantity
  let sugar_cost := sugar_price * sugar_quantity
  let subtotal := apple_cost + milk_cost + bread_cost + sugar_cost
  if subtotal ≥ coupon_threshold then subtotal - coupon_discount else subtotal

-- Theorem to prove
theorem ben_spending :
  total_spending = 43 :=
sorry

end NUMINAMATH_CALUDE_ben_spending_l3209_320902


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3209_320934

theorem solve_exponential_equation :
  ∃ y : ℝ, 4^(3*y) = (64 : ℝ)^(1/3) ∧ y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3209_320934


namespace NUMINAMATH_CALUDE_solution_set_f_geq_1_max_value_f_minus_x_squared_plus_x_l3209_320927

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem 1: The solution set of f(x) ≥ 1 is {x | x ≥ 1}
theorem solution_set_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

-- Theorem 2: The maximum value of f(x) - x^2 + x is 5/4
theorem max_value_f_minus_x_squared_plus_x :
  ∃ (x : ℝ), ∀ (y : ℝ), f y - y^2 + y ≤ f x - x^2 + x ∧ f x - x^2 + x = 5/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_1_max_value_f_minus_x_squared_plus_x_l3209_320927


namespace NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l3209_320901

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem extreme_values_and_monotonicity 
  (a b c : ℝ) 
  (h1 : ∃ (y : ℝ), y = f a b c (-2) ∧ (∀ x, f a b c x ≤ y))
  (h2 : ∃ (y : ℝ), y = f a b c 1 ∧ (∀ x, f a b c x ≤ y))
  (h3 : ∀ x ∈ Set.Icc (-1) 2, f a b c x < c^2) :
  (a = 3/2 ∧ b = -6) ∧ 
  (∀ x < -2, ∀ y ∈ Set.Ioo x (-2), f a b c x < f a b c y) ∧
  (∀ x ∈ Set.Ioo (-2) 1, ∀ y ∈ Set.Ioo x 1, f a b c x > f a b c y) ∧
  (∀ x > 1, ∀ y ∈ Set.Ioo 1 x, f a b c x > f a b c y) ∧
  (c > 2 ∨ c < -1) := by
  sorry


end NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l3209_320901


namespace NUMINAMATH_CALUDE_calen_current_pencils_l3209_320908

-- Define the number of pencils for each person
def candy_pencils : ℕ := 9
def caleb_pencils : ℕ := 2 * candy_pencils - 3
def calen_original_pencils : ℕ := caleb_pencils + 5
def calen_lost_pencils : ℕ := 10

-- Theorem to prove
theorem calen_current_pencils :
  calen_original_pencils - calen_lost_pencils = 10 := by
  sorry

end NUMINAMATH_CALUDE_calen_current_pencils_l3209_320908


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l3209_320975

-- Define the line
def line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

-- Define the tangent point
def tangent_point : ℝ × ℝ := (2, -1)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 9

-- Theorem statement
theorem circle_tangent_to_line :
  ∀ (x y : ℝ),
  line x y →
  (∃ (t : ℝ), (x, y) = (2 + t * 3, -1 - t * 4)) →
  circle_equation x y :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l3209_320975


namespace NUMINAMATH_CALUDE_smallest_divisor_is_number_itself_l3209_320914

def form_number (a b : Nat) (digit : Nat) : Nat :=
  a * 1000 + digit * 100 + b

theorem smallest_divisor_is_number_itself :
  let complete_number := form_number 761 829 3
  complete_number % complete_number = 0 ∧
  ∀ d : Nat, d > 0 ∧ d < complete_number → complete_number % d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_is_number_itself_l3209_320914


namespace NUMINAMATH_CALUDE_expected_value_binomial_l3209_320915

/-- The number of missile launches -/
def n : ℕ := 10

/-- The probability of an accident in a single launch -/
def p : ℝ := 0.01

/-- The random variable representing the number of accidents -/
def ξ : Nat → ℝ := sorry

theorem expected_value_binomial :
  Finset.sum (Finset.range (n + 1)) (fun k => k * (n.choose k : ℝ) * p^k * (1 - p)^(n - k)) = n * p :=
sorry

end NUMINAMATH_CALUDE_expected_value_binomial_l3209_320915


namespace NUMINAMATH_CALUDE_multi_digit_square_has_even_digit_l3209_320907

/-- A multi-digit number is a natural number with at least two digits. -/
def is_multi_digit (n : ℕ) : Prop := n ≥ 10

/-- A number is a perfect square if it's equal to some natural number squared. -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- A digit is even if it's divisible by 2. -/
def is_even_digit (d : ℕ) : Prop := d % 10 % 2 = 0

/-- For any multi-digit perfect square, there exists at least one even digit. -/
theorem multi_digit_square_has_even_digit (n : ℕ) 
  (h1 : is_multi_digit n) (h2 : is_perfect_square n) : 
  ∃ d : ℕ, d < n ∧ is_even_digit d :=
sorry

end NUMINAMATH_CALUDE_multi_digit_square_has_even_digit_l3209_320907


namespace NUMINAMATH_CALUDE_opposite_of_three_l3209_320919

theorem opposite_of_three : -(3 : ℤ) = -3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_three_l3209_320919


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l3209_320998

theorem binomial_expansion_properties (n : ℕ) :
  (∀ x : ℝ, x > 0 → 
    Nat.choose n 2 = Nat.choose n 6) →
  (n = 8 ∧ 
   ∀ k : ℕ, k ≤ n → (8 : ℝ) - (3 / 2 : ℝ) * k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l3209_320998


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l3209_320964

-- Define a triangle with side lengths a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_angle_ratio (t : Triangle) 
  (h1 : t.a^2 = t.b * (t.b + t.c)) -- Given condition
  (h2 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0) -- Angles are positive
  (h3 : t.A + t.B + t.C = π) -- Sum of angles in a triangle
  (h4 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0) -- Side lengths are positive
  : t.B / t.A = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_ratio_l3209_320964


namespace NUMINAMATH_CALUDE_sin_2alpha_over_cos_squared_l3209_320909

theorem sin_2alpha_over_cos_squared (α : Real) 
  (h : Real.sin α = 3 * Real.cos α) : 
  Real.sin (2 * α) / (Real.cos α)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_over_cos_squared_l3209_320909


namespace NUMINAMATH_CALUDE_max_value_abc_l3209_320917

theorem max_value_abc (a b c : ℝ) (h : a + 3 * b + c = 6) :
  (∀ x y z : ℝ, x + 3 * y + z = 6 → a * b + a * c + b * c ≥ x * y + x * z + y * z) →
  a * b + a * c + b * c = 12 :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l3209_320917


namespace NUMINAMATH_CALUDE_smaller_solution_form_l3209_320904

theorem smaller_solution_form : ∃ (p q : ℤ),
  ∃ (x : ℝ),
    x^(1/4) + (40 - x)^(1/4) = 2 ∧
    x = p - Real.sqrt q ∧
    ∀ (y : ℝ), y^(1/4) + (40 - y)^(1/4) = 2 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_form_l3209_320904


namespace NUMINAMATH_CALUDE_mother_extra_rides_l3209_320944

/-- The number of times Billy rode his bike -/
def billy_rides : ℕ := 17

/-- The number of times John rode his bike -/
def john_rides : ℕ := 2 * billy_rides

/-- The number of times the mother rode her bike -/
def mother_rides (x : ℕ) : ℕ := john_rides + x

/-- The total number of times they all rode their bikes -/
def total_rides : ℕ := 95

/-- Theorem stating that the mother rode her bike 10 times more than John -/
theorem mother_extra_rides : 
  ∃ x : ℕ, x = 10 ∧ mother_rides x = john_rides + x ∧ 
  billy_rides + john_rides + mother_rides x = total_rides :=
sorry

end NUMINAMATH_CALUDE_mother_extra_rides_l3209_320944


namespace NUMINAMATH_CALUDE_andy_max_cookies_l3209_320932

theorem andy_max_cookies (total_cookies : ℕ) (andy alexa alice : ℕ) : 
  total_cookies = 36 →
  alexa = 3 * andy →
  alice = 2 * andy →
  total_cookies = andy + alexa + alice →
  andy ≤ 6 ∧ ∃ (n : ℕ), n = 6 ∧ n = andy := by
  sorry

end NUMINAMATH_CALUDE_andy_max_cookies_l3209_320932


namespace NUMINAMATH_CALUDE_sqrt_two_thirds_times_sqrt_six_l3209_320924

theorem sqrt_two_thirds_times_sqrt_six : Real.sqrt (2/3) * Real.sqrt 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_thirds_times_sqrt_six_l3209_320924


namespace NUMINAMATH_CALUDE_mandys_data_plan_charge_l3209_320972

/-- The normal monthly charge for Mandy's data plan -/
def normal_charge : ℝ := 30

/-- The total amount Mandy paid for 6 months -/
def total_paid : ℝ := 175

/-- The extra fee charged in the fourth month -/
def extra_fee : ℝ := 15

theorem mandys_data_plan_charge :
  (normal_charge / 3) +  -- First month (promotional rate)
  (normal_charge + extra_fee) +  -- Fourth month (with extra fee)
  (4 * normal_charge) =  -- Other four months
  total_paid := by sorry

end NUMINAMATH_CALUDE_mandys_data_plan_charge_l3209_320972


namespace NUMINAMATH_CALUDE_min_value_theorem_l3209_320957

theorem min_value_theorem (a : ℝ) (m n : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : 2*m - 1 + n = 0) :
  (4:ℝ)^m + 2^n ≥ 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3209_320957


namespace NUMINAMATH_CALUDE_ratio_sum_squares_theorem_l3209_320971

theorem ratio_sum_squares_theorem (x y z : ℝ) : 
  y = 2 * x ∧ z = 3 * x ∧ x^2 + y^2 + z^2 = 2744 → x + y + z = 84 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_theorem_l3209_320971


namespace NUMINAMATH_CALUDE_greatest_digit_sum_base_seven_l3209_320965

/-- Represents a positive integer in base 7 --/
def BaseSevenRepresentation := List Nat

/-- Converts a natural number to its base-seven representation --/
def toBaseSeven (n : Nat) : BaseSevenRepresentation :=
  sorry

/-- Calculates the sum of digits in a base-seven representation --/
def sumDigits (repr : BaseSevenRepresentation) : Nat :=
  sorry

/-- The upper bound for the problem --/
def upperBound : Nat := 2401

theorem greatest_digit_sum_base_seven :
  ∃ (max : Nat), ∀ (n : Nat), n < upperBound →
    sumDigits (toBaseSeven n) ≤ max ∧
    ∃ (m : Nat), m < upperBound ∧ sumDigits (toBaseSeven m) = max ∧
    max = 12 :=
  sorry

end NUMINAMATH_CALUDE_greatest_digit_sum_base_seven_l3209_320965


namespace NUMINAMATH_CALUDE_rowing_distance_problem_l3209_320953

/-- Proves that the distance to a destination is 72 km given specific rowing conditions -/
theorem rowing_distance_problem (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) : 
  rowing_speed = 10 → 
  current_speed = 2 → 
  total_time = 15 → 
  (rowing_speed + current_speed) * (rowing_speed - current_speed) * total_time / 
    (rowing_speed + current_speed + rowing_speed - current_speed) = 72 := by
  sorry

#check rowing_distance_problem

end NUMINAMATH_CALUDE_rowing_distance_problem_l3209_320953


namespace NUMINAMATH_CALUDE_cos_angle_AMB_formula_l3209_320940

/-- Regular square pyramid with vertex A and square base BCDE -/
structure RegularSquarePyramid where
  s : ℝ  -- side length of the base
  h : ℝ  -- height of the pyramid
  l : ℝ  -- slant height of the pyramid

/-- Point M is the midpoint of diagonal BD -/
def midpoint_M (p : RegularSquarePyramid) : ℝ × ℝ × ℝ := sorry

/-- Angle AMB in the regular square pyramid -/
def angle_AMB (p : RegularSquarePyramid) : ℝ := sorry

theorem cos_angle_AMB_formula (p : RegularSquarePyramid) :
  Real.cos (angle_AMB p) = (p.l^2 + p.h^2) / (2 * p.l * Real.sqrt (p.h^2 + p.s^2 / 2)) :=
sorry

end NUMINAMATH_CALUDE_cos_angle_AMB_formula_l3209_320940


namespace NUMINAMATH_CALUDE_smallest_interesting_number_l3209_320951

/-- A natural number is interesting if 2n is a perfect square and 15n is a perfect cube. -/
def IsInteresting (n : ℕ) : Prop :=
  ∃ k m : ℕ, 2 * n = k^2 ∧ 15 * n = m^3

/-- 1800 is the smallest interesting natural number. -/
theorem smallest_interesting_number : 
  IsInteresting 1800 ∧ ∀ n < 1800, ¬IsInteresting n := by
  sorry

#check smallest_interesting_number

end NUMINAMATH_CALUDE_smallest_interesting_number_l3209_320951


namespace NUMINAMATH_CALUDE_expand_binomial_product_l3209_320942

theorem expand_binomial_product (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomial_product_l3209_320942

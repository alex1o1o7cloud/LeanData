import Mathlib

namespace angle_supplement_l2526_252682

theorem angle_supplement (θ : ℝ) : 
  (90 - θ = 30) → (180 - θ = 120) := by
  sorry

end angle_supplement_l2526_252682


namespace green_team_opponent_score_l2526_252602

/-- The final score of a team's opponent given the team's score and lead -/
def opponent_score (team_score : ℕ) (lead : ℕ) : ℕ :=
  team_score - lead

/-- Theorem: Given Green Team's score of 39 and lead of 29, their opponent's score is 10 -/
theorem green_team_opponent_score :
  opponent_score 39 29 = 10 := by
  sorry

end green_team_opponent_score_l2526_252602


namespace expression_defined_iff_l2526_252644

theorem expression_defined_iff (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x - 2)) / (Real.sqrt (x - 1))) ↔ x ≥ 2 := by
  sorry

end expression_defined_iff_l2526_252644


namespace product_of_max_min_elements_l2526_252667

def S : Set ℝ := {z | ∃ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 5 ∧ z = 5/x + y}

theorem product_of_max_min_elements (M N : ℝ) : 
  (∀ z ∈ S, z ≤ M) ∧ (M ∈ S) ∧ (∀ z ∈ S, N ≤ z) ∧ (N ∈ S) →
  M * N = 20 * Real.sqrt 5 := by
  sorry

end product_of_max_min_elements_l2526_252667


namespace log_f_geq_one_f_geq_a_iff_a_leq_one_l2526_252629

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x - a|

-- Part 1: Prove that log f(x) ≥ 1 when a = -8
theorem log_f_geq_one (x : ℝ) : Real.log (f (-8) x) ≥ 1 := by
  sorry

-- Part 2: Prove that f(x) ≥ a for all x ∈ ℝ if and only if a ≤ 1
theorem f_geq_a_iff_a_leq_one :
  (∀ x : ℝ, f a x ≥ a) ↔ a ≤ 1 := by
  sorry

end log_f_geq_one_f_geq_a_iff_a_leq_one_l2526_252629


namespace park_area_l2526_252625

/-- A rectangular park with width one-third of length and perimeter 72 meters has area 243 square meters -/
theorem park_area (w : ℝ) (l : ℝ) : 
  w > 0 → l > 0 → w = l / 3 → 2 * (w + l) = 72 → w * l = 243 := by
  sorry

end park_area_l2526_252625


namespace sum_of_coefficients_l2526_252690

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (x - 2)^6 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5 + a₆*(x+1)^6) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 64 := by
sorry

end sum_of_coefficients_l2526_252690


namespace subway_security_comprehensive_l2526_252608

-- Define the type for survey options
inductive SurveyOption
| TouristSatisfaction
| SubwaySecurity
| YellowRiverFish
| LightBulbLifespan

-- Define what it means for a survey to be comprehensive
def is_comprehensive (survey : SurveyOption) : Prop :=
  match survey with
  | SurveyOption.SubwaySecurity => true
  | _ => false

-- Theorem statement
theorem subway_security_comprehensive :
  ∀ (survey : SurveyOption),
    is_comprehensive survey ↔ survey = SurveyOption.SubwaySecurity :=
by sorry

end subway_security_comprehensive_l2526_252608


namespace largest_divisible_n_l2526_252634

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ (n + 15) ∣ (n^3 + 150) ∧ ∀ (m : ℕ), m > n → ¬((m + 15) ∣ (m^3 + 150)) :=
by
  -- The largest such n is 2385
  use 2385
  sorry

end largest_divisible_n_l2526_252634


namespace abc_sum_mod_8_l2526_252609

theorem abc_sum_mod_8 (a b c : ℕ) : 
  0 < a ∧ a < 8 ∧ 
  0 < b ∧ b < 8 ∧ 
  0 < c ∧ c < 8 ∧ 
  (a * b * c) % 8 = 1 ∧ 
  (4 * b * c) % 8 = 3 ∧ 
  (5 * b) % 8 = (3 + b) % 8 
  → (a + b + c) % 8 = 2 := by
  sorry

end abc_sum_mod_8_l2526_252609


namespace f_properties_l2526_252641

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2 + 1

theorem f_properties :
  (∃ (x : ℝ), f x = Real.sqrt 2 ∧ ∀ (y : ℝ), f y ≤ Real.sqrt 2) ∧
  (∀ (θ : ℝ), f θ = 3/5 → Real.cos (2 * (π/4 - 2*θ)) = 16/25) :=
sorry

end f_properties_l2526_252641


namespace line_in_fourth_quadrant_l2526_252612

/-- A line passes through the fourth quadrant if it intersects both the negative x-axis and the positive y-axis. -/
def passes_through_fourth_quadrant (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ (a - 2) * x + a * y + 2 * a - 3 = 0

/-- The theorem stating the condition for the line to pass through the fourth quadrant. -/
theorem line_in_fourth_quadrant (a : ℝ) :
  passes_through_fourth_quadrant a ↔ a ∈ Set.Iio 0 ∪ Set.Ioi (3/2) :=
sorry

end line_in_fourth_quadrant_l2526_252612


namespace marias_gum_l2526_252676

/-- Represents the number of pieces of gum Maria has -/
def total_gum (initial : ℕ) (x : ℕ) (y : ℕ) : ℕ := initial + x + y

/-- Theorem stating the total number of pieces of gum Maria has -/
theorem marias_gum (x y : ℕ) (h1 : 10 ≤ x ∧ x < 100) (h2 : 10 ≤ y ∧ y < 100) :
  total_gum 58 x y = 58 + x + y := by sorry

end marias_gum_l2526_252676


namespace count_specific_coin_toss_sequences_l2526_252692

def coin_toss_sequences (n : ℕ) (th ht tt hh : ℕ) : ℕ :=
  Nat.choose 4 2 * Nat.choose 8 3 * Nat.choose 5 4 * Nat.choose 11 5

theorem count_specific_coin_toss_sequences :
  coin_toss_sequences 15 2 3 4 5 = 775360 := by
  sorry

end count_specific_coin_toss_sequences_l2526_252692


namespace closest_ratio_is_one_l2526_252665

/-- Represents the admission fee structure and total collection -/
structure AdmissionData where
  adult_fee : ℕ
  child_fee : ℕ
  total_collected : ℕ

/-- Finds the ratio of adults to children closest to 1 given admission data -/
def closest_ratio_to_one (data : AdmissionData) : ℚ :=
  sorry

/-- The main theorem stating that the closest ratio to 1 is exactly 1 for the given data -/
theorem closest_ratio_is_one :
  let data : AdmissionData := {
    adult_fee := 30,
    child_fee := 15,
    total_collected := 2700
  }
  closest_ratio_to_one data = 1 := by sorry

end closest_ratio_is_one_l2526_252665


namespace rectangle_area_proof_l2526_252675

theorem rectangle_area_proof (large_square_side : ℝ) 
  (rectangle_length rectangle_width : ℝ) 
  (small_square_side : ℝ) :
  large_square_side = 4 →
  rectangle_length = 1 →
  rectangle_width = 4 →
  small_square_side = 2 →
  large_square_side^2 - (rectangle_length * rectangle_width + small_square_side^2) = 8 :=
by
  sorry

#check rectangle_area_proof

end rectangle_area_proof_l2526_252675


namespace zebras_permutations_l2526_252666

theorem zebras_permutations :
  Nat.factorial 6 = 720 := by
  sorry

end zebras_permutations_l2526_252666


namespace book_cost_problem_l2526_252680

theorem book_cost_problem (book_price : ℝ) : 
  (3 * book_price = 45) → (7 * book_price = 105) := by
  sorry

end book_cost_problem_l2526_252680


namespace largest_stamps_per_page_l2526_252663

theorem largest_stamps_per_page (book1_stamps : ℕ) (book2_stamps : ℕ) :
  book1_stamps = 924 →
  book2_stamps = 1200 →
  ∃ (stamps_per_page : ℕ),
    stamps_per_page = Nat.gcd book1_stamps book2_stamps ∧
    stamps_per_page ≤ book1_stamps ∧
    stamps_per_page ≤ book2_stamps ∧
    ∀ (n : ℕ), n ∣ book1_stamps ∧ n ∣ book2_stamps → n ≤ stamps_per_page :=
by sorry

end largest_stamps_per_page_l2526_252663


namespace winston_gas_tank_capacity_l2526_252638

/-- Represents the gas tank of Winston's car -/
structure GasTank where
  initialGas : ℕ
  usedToStore : ℕ
  usedToDoctor : ℕ
  neededToRefill : ℕ

/-- Calculates the maximum capacity of the gas tank -/
def maxCapacity (tank : GasTank) : ℕ :=
  tank.initialGas - tank.usedToStore - tank.usedToDoctor + tank.neededToRefill

/-- Theorem stating that the maximum capacity of Winston's gas tank is 12 gallons -/
theorem winston_gas_tank_capacity :
  let tank : GasTank := {
    initialGas := 10,
    usedToStore := 6,
    usedToDoctor := 2,
    neededToRefill := 10
  }
  maxCapacity tank = 12 := by sorry

end winston_gas_tank_capacity_l2526_252638


namespace pear_apple_difference_l2526_252603

theorem pear_apple_difference :
  let red_apples : ℕ := 15
  let green_apples : ℕ := 8
  let pears : ℕ := 32
  let total_apples : ℕ := red_apples + green_apples
  pears - total_apples = 9 := by
  sorry

end pear_apple_difference_l2526_252603


namespace equation_linear_iff_k_eq_neg_two_l2526_252600

/-- The equation (k-2)x^(|k|-1) = k+1 is linear in x if and only if k = -2 -/
theorem equation_linear_iff_k_eq_neg_two :
  ∀ k : ℤ, (∃ a b : ℝ, ∀ x : ℝ, (k - 2) * x^(|k| - 1) = a * x + b) ↔ k = -2 := by
  sorry

end equation_linear_iff_k_eq_neg_two_l2526_252600


namespace correct_answer_is_nothing_l2526_252637

/-- Represents the possible answers to the Question of Questions -/
inductive Answer
| Something
| Nothing

/-- Represents a priest -/
structure Priest where
  knowsAnswer : Bool
  alwaysLies : Bool

/-- The response given by a priest -/
def priestResponse : Answer := Answer.Something

/-- Theorem: If a priest who knows the correct answer responds with "Something exists,"
    then the correct answer is "Nothing exists" -/
theorem correct_answer_is_nothing 
  (priest : Priest) 
  (h1 : priest.knowsAnswer = true) 
  (h2 : priest.alwaysLies = true) 
  (h3 : priestResponse = Answer.Something) : 
  Answer.Nothing = Answer.Nothing := by sorry


end correct_answer_is_nothing_l2526_252637


namespace b_97_mod_36_l2526_252613

def b (n : ℕ) : ℕ := 5^n + 7^n

theorem b_97_mod_36 : b 97 % 36 = 12 := by
  sorry

end b_97_mod_36_l2526_252613


namespace new_shoes_count_l2526_252639

theorem new_shoes_count (pairs_bought : ℕ) (shoes_per_pair : ℕ) : 
  pairs_bought = 3 → shoes_per_pair = 2 → pairs_bought * shoes_per_pair = 6 := by
  sorry

end new_shoes_count_l2526_252639


namespace student_ability_theorem_l2526_252664

-- Define the function
def f (x : ℝ) : ℝ := -0.1 * x^2 + 2.6 * x + 43

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 30 }

theorem student_ability_theorem :
  (∀ x ∈ domain, x ≤ 13 → ∀ y ∈ domain, x ≤ y → f x ≤ f y) ∧
  (∀ x ∈ domain, x ≥ 13 → ∀ y ∈ domain, x ≤ y → f x ≥ f y) ∧
  f 10 = 59 ∧
  (∀ x ∈ domain, f x ≤ f 13) := by
  sorry

end student_ability_theorem_l2526_252664


namespace sqrt_3_and_sqrt_1_3_same_type_l2526_252685

/-- Two quadratic radicals are of the same type if they have the same radicand after simplification -/
def same_type (a b : ℝ) : Prop :=
  ∃ (k₁ k₂ r : ℝ), k₁ > 0 ∧ k₂ > 0 ∧ r > 0 ∧ a = k₁ * Real.sqrt r ∧ b = k₂ * Real.sqrt r

/-- √3 and √(1/3) are of the same type -/
theorem sqrt_3_and_sqrt_1_3_same_type : same_type (Real.sqrt 3) (Real.sqrt (1/3)) := by
  sorry

end sqrt_3_and_sqrt_1_3_same_type_l2526_252685


namespace even_function_extension_l2526_252683

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The main theorem -/
theorem even_function_extension
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_neg : ∀ x < 0, f x = x - x^4) :
  ∀ x > 0, f x = -x - x^4 := by
  sorry

end even_function_extension_l2526_252683


namespace share_of_a_l2526_252628

theorem share_of_a (total : ℚ) (a b c : ℚ) : 
  total = 200 →
  total = a + b + c →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a = 60 := by
  sorry

end share_of_a_l2526_252628


namespace parallel_vectors_x_value_l2526_252631

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Given vectors a and b, prove that if they are parallel, then x = 2 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 4)
  are_parallel a b → x = 2 := by
  sorry

end parallel_vectors_x_value_l2526_252631


namespace chicken_rabbit_problem_l2526_252668

theorem chicken_rabbit_problem :
  ∀ (chickens rabbits : ℕ),
    chickens + rabbits = 35 →
    2 * chickens + 4 * rabbits = 94 →
    chickens = 23 ∧ rabbits = 12 := by
  sorry

end chicken_rabbit_problem_l2526_252668


namespace divisible_count_equality_l2526_252661

theorem divisible_count_equality (n : Nat) : n = 56000 →
  (Finset.filter (fun x => x % 7 = 0 ∧ x % 8 ≠ 0) (Finset.range (n + 1))).card =
  (Finset.filter (fun x => x % 8 = 0) (Finset.range (n + 1))).card := by
  sorry

end divisible_count_equality_l2526_252661


namespace parallelogram_base_length_l2526_252643

/-- A parallelogram with an area of 450 sq m and an altitude twice the corresponding base has a base length of 15 meters. -/
theorem parallelogram_base_length :
  ∀ (base altitude : ℝ),
  base > 0 →
  altitude > 0 →
  base * altitude = 450 →
  altitude = 2 * base →
  base = 15 := by
sorry

end parallelogram_base_length_l2526_252643


namespace nancy_alyssa_book_ratio_l2526_252616

def alyssa_books : ℕ := 36
def nancy_books : ℕ := 252

theorem nancy_alyssa_book_ratio :
  nancy_books / alyssa_books = 7 := by
  sorry

end nancy_alyssa_book_ratio_l2526_252616


namespace round_trip_average_speed_l2526_252646

-- Define the upstream and downstream speeds
def upstream_speed : ℝ := 6
def downstream_speed : ℝ := 3

-- Define the theorem
theorem round_trip_average_speed :
  let total_distance : ℝ → ℝ := λ d => 2 * d
  let upstream_time : ℝ → ℝ := λ d => d / upstream_speed
  let downstream_time : ℝ → ℝ := λ d => d / downstream_speed
  let total_time : ℝ → ℝ := λ d => upstream_time d + downstream_time d
  let average_speed : ℝ → ℝ := λ d => total_distance d / total_time d
  ∀ d : ℝ, d > 0 → average_speed d = 4 :=
by sorry

end round_trip_average_speed_l2526_252646


namespace least_number_of_beads_beads_divisibility_least_beads_l2526_252601

theorem least_number_of_beads (n : ℕ) : n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n → n ≥ 840 := by
  sorry

theorem beads_divisibility : 2 ∣ 840 ∧ 3 ∣ 840 ∧ 5 ∣ 840 ∧ 7 ∣ 840 ∧ 8 ∣ 840 := by
  sorry

theorem least_beads : ∃ (n : ℕ), n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n ∧ n = 840 := by
  sorry

end least_number_of_beads_beads_divisibility_least_beads_l2526_252601


namespace tonys_monthly_rent_l2526_252689

/-- Calculates the monthly rent for a cottage given its room sizes and cost per square foot. -/
def calculate_monthly_rent (master_area : ℕ) (guest_bedroom_area : ℕ) (num_guest_bedrooms : ℕ) (other_areas : ℕ) (cost_per_sqft : ℕ) : ℕ :=
  let total_area := master_area + (guest_bedroom_area * num_guest_bedrooms) + other_areas
  total_area * cost_per_sqft

/-- Theorem stating that Tony's monthly rent is $3000 given the specified conditions. -/
theorem tonys_monthly_rent : 
  calculate_monthly_rent 500 200 2 600 2 = 3000 := by
  sorry

#eval calculate_monthly_rent 500 200 2 600 2

end tonys_monthly_rent_l2526_252689


namespace total_devices_l2526_252652

theorem total_devices (computers televisions : ℕ) : 
  computers = 32 → televisions = 66 → computers + televisions = 98 := by
  sorry

end total_devices_l2526_252652


namespace weather_forecast_probability_l2526_252627

/-- The probability of success for a single trial -/
def p : ℝ := 0.8

/-- The number of trials -/
def n : ℕ := 3

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The probability of exactly k successes in n independent trials with probability p each -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem weather_forecast_probability :
  binomial_probability n k p = 0.384 := by
  sorry

end weather_forecast_probability_l2526_252627


namespace three_red_faces_count_total_cubes_count_l2526_252698

/-- Represents a rectangular solid composed of small cubes -/
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of corner cubes in a rectangular solid -/
def cornerCubes (solid : RectangularSolid) : ℕ :=
  8

/-- Theorem: In a 5 × 4 × 2 rectangular solid with painted outer surface,
    the number of small cubes with exactly 3 red faces is 8 -/
theorem three_red_faces_count :
  let solid : RectangularSolid := ⟨5, 4, 2⟩
  cornerCubes solid = 8 := by
  sorry

/-- Verifies that the total number of cubes is 40 -/
theorem total_cubes_count :
  let solid : RectangularSolid := ⟨5, 4, 2⟩
  solid.length * solid.width * solid.height = 40 := by
  sorry

end three_red_faces_count_total_cubes_count_l2526_252698


namespace a_lower_bound_l2526_252660

-- Define the inequality condition
def inequality_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 2 * x + 8 * x^3 + a^2 * Real.exp (2 * x) < 4 * x^2 + a * Real.exp x + a^3 * Real.exp (3 * x)

-- State the theorem
theorem a_lower_bound (a : ℝ) (h : inequality_condition a) : a > 2 / Real.exp 1 := by
  sorry

end a_lower_bound_l2526_252660


namespace inequality_solution_l2526_252642

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | 1 < x }
  else if 0 < a ∧ a < 2 then { x | 1 < x ∧ x < 2/a }
  else if a = 2 then ∅
  else if a > 2 then { x | 2/a < x ∧ x < 1 }
  else { x | x < 2/a ∨ 1 < x }

theorem inequality_solution (a : ℝ) :
  { x : ℝ | a * x^2 - (a + 2) * x + 2 < 0 } = solution_set a := by sorry

end inequality_solution_l2526_252642


namespace fourteenth_root_unity_l2526_252672

theorem fourteenth_root_unity : ∃ n : ℤ, 
  (Complex.tan (Real.pi / 7) + Complex.I) / (Complex.tan (Real.pi / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * n * Real.pi / 14)) :=
by sorry

end fourteenth_root_unity_l2526_252672


namespace public_library_book_count_l2526_252653

/-- The number of books in Oak Grove's public library -/
def public_library_books : ℕ := 7092 - 5106

/-- The total number of books in Oak Grove libraries -/
def total_books : ℕ := 7092

/-- The number of books in Oak Grove's school libraries -/
def school_library_books : ℕ := 5106

theorem public_library_book_count :
  public_library_books = 1986 ∧
  total_books = public_library_books + school_library_books :=
by sorry

end public_library_book_count_l2526_252653


namespace grape_price_calculation_l2526_252684

theorem grape_price_calculation (G : ℚ) : 
  (11 * G + 7 * 50 = 1428) → G = 98 := by
  sorry

end grape_price_calculation_l2526_252684


namespace oranges_picked_total_l2526_252633

/-- The number of oranges Mary picked -/
def mary_oranges : ℕ := 122

/-- The number of oranges Jason picked -/
def jason_oranges : ℕ := 105

/-- The total number of oranges picked -/
def total_oranges : ℕ := mary_oranges + jason_oranges

theorem oranges_picked_total :
  total_oranges = 227 :=
by sorry

end oranges_picked_total_l2526_252633


namespace chris_age_l2526_252677

theorem chris_age (a b c : ℚ) : 
  (a + b + c) / 3 = 10 →
  c - 5 = 2 * a →
  b + 4 = 3/4 * (a + 4) →
  c = 283/15 :=
by sorry

end chris_age_l2526_252677


namespace binomial_equation_solution_l2526_252671

def binomial (n k : ℕ) : ℕ := sorry

theorem binomial_equation_solution :
  ∀ x : ℕ, binomial 15 (2*x+1) = binomial 15 (x+2) → x = 1 ∨ x = 4 :=
by sorry

end binomial_equation_solution_l2526_252671


namespace rectangle_squares_sides_l2526_252694

/-- Represents the side lengths of squares in a rectangle divided into 6 squares. -/
structure SquareSides where
  s1 : ℝ
  s2 : ℝ
  s3 : ℝ
  s4 : ℝ
  s5 : ℝ
  s6 : ℝ

/-- Given a rectangle divided into 6 squares with specific conditions,
    proves that the side lengths of the squares are as calculated. -/
theorem rectangle_squares_sides (sides : SquareSides) 
    (h1 : sides.s1 = 18)
    (h2 : sides.s2 = 3) :
    sides.s3 = 15 ∧ 
    sides.s4 = 12 ∧ 
    sides.s5 = 12 ∧ 
    sides.s6 = 21 := by
  sorry


end rectangle_squares_sides_l2526_252694


namespace adams_laundry_l2526_252693

theorem adams_laundry (total_loads : ℕ) (remaining_loads : ℕ) (washed_loads : ℕ) : 
  total_loads = 14 → remaining_loads = 6 → washed_loads = total_loads - remaining_loads → washed_loads = 8 := by
  sorry

end adams_laundry_l2526_252693


namespace completing_square_quadratic_l2526_252645

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 + 8*x + 7 = 0) ↔ ((x + 4)^2 = 9) := by
  sorry

end completing_square_quadratic_l2526_252645


namespace initial_puppies_count_l2526_252610

/-- The number of puppies initially in the shelter -/
def initial_puppies : ℕ := sorry

/-- The number of additional puppies brought in -/
def additional_puppies : ℕ := 3

/-- The number of puppies adopted per day -/
def adoptions_per_day : ℕ := 3

/-- The number of days it takes for all puppies to be adopted -/
def days_to_adopt_all : ℕ := 2

/-- Theorem stating that the initial number of puppies is 3 -/
theorem initial_puppies_count : initial_puppies = 3 := by sorry

end initial_puppies_count_l2526_252610


namespace inequality_proof_l2526_252606

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  1 / a + 4 / (1 - a) ≥ 9 := by
  sorry

end inequality_proof_l2526_252606


namespace angelinas_speed_to_gym_l2526_252619

-- Define the constants
def distance_home_to_grocery : ℝ := 200
def distance_grocery_to_gym : ℝ := 300
def time_difference : ℝ := 50

-- Define the variables
variable (v : ℝ) -- Speed from home to grocery

-- Define the theorem
theorem angelinas_speed_to_gym :
  (distance_home_to_grocery / v) - (distance_grocery_to_gym / (2 * v)) = time_difference →
  2 * v = 2 := by
  sorry

end angelinas_speed_to_gym_l2526_252619


namespace parabola_hyperbola_triangle_l2526_252686

/-- Theorem: Value of p for a parabola and hyperbola forming an isosceles right triangle -/
theorem parabola_hyperbola_triangle (p a b : ℝ) : 
  p > 0 → a > 0 → b > 0 →
  (∀ x y, x^2 = 2*p*y) →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1) →
  (∃ x₁ y₁ x₂ y₂ x₃ y₃,
    -- Points form an isosceles right triangle
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₂ - x₃)^2 + (y₂ - y₃)^2 ∧
    (x₁ - x₂) * (x₂ - x₃) + (y₁ - y₂) * (y₂ - y₃) = 0 ∧
    -- Area of the triangle is 1
    abs ((x₁ - x₃) * (y₂ - y₃) - (x₂ - x₃) * (y₁ - y₃)) / 2 = 1 ∧
    -- Points lie on the directrix and asymptotes
    y₁ = -p/2 ∧ y₂ = -p/2 ∧
    y₁ = b/a * x₁ ∧ y₃ = -b/a * x₃) →
  p = 2 := by sorry

end parabola_hyperbola_triangle_l2526_252686


namespace sequence_a_integer_sequence_a_recurrence_l2526_252659

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) * (sequence_a (n + 1) + 1) / sequence_a n

theorem sequence_a_integer (n : ℕ) : ∃ k : ℤ, sequence_a n = k := by
  sorry

theorem sequence_a_recurrence (n : ℕ) : n ≥ 1 →
  sequence_a (n + 1) * sequence_a (n - 1) = sequence_a n * (sequence_a n + 1) := by
  sorry

end sequence_a_integer_sequence_a_recurrence_l2526_252659


namespace complement_A_intersect_B_l2526_252662

def U : Set ℕ := {x | 1 < x ∧ x < 6}
def A : Set ℕ := {2, 3}
def B : Set ℕ := {2, 4, 5}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {4, 5} := by sorry

end complement_A_intersect_B_l2526_252662


namespace total_disks_l2526_252624

/-- Represents the number of disks of each color in the bag -/
structure DiskCount where
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- The properties of the disk distribution in the bag -/
def validDiskCount (d : DiskCount) : Prop :=
  ∃ (x : ℕ),
    d.blue = 3 * x ∧
    d.yellow = 7 * x ∧
    d.green = 8 * x ∧
    d.green = d.blue + 15

/-- The theorem stating the total number of disks in the bag -/
theorem total_disks (d : DiskCount) (h : validDiskCount d) : 
  d.blue + d.yellow + d.green = 54 := by
  sorry


end total_disks_l2526_252624


namespace sample_size_of_500_selection_l2526_252696

/-- Represents a batch of CDs -/
structure CDBatch where
  size : ℕ

/-- Represents a sample of CDs -/
structure CDSample where
  size : ℕ
  source : CDBatch

/-- Defines a random selection of CDs from a batch -/
def randomSelection (batch : CDBatch) (n : ℕ) : CDSample :=
  { size := n
    source := batch }

/-- Theorem stating that the sample size of a random selection of 500 CDs is 500 -/
theorem sample_size_of_500_selection (batch : CDBatch) :
  (randomSelection batch 500).size = 500 := by
  sorry

end sample_size_of_500_selection_l2526_252696


namespace stepa_multiplication_l2526_252678

theorem stepa_multiplication (sequence : Fin 5 → ℕ) 
  (h1 : ∀ i : Fin 4, sequence (i.succ) = (3 * sequence i) / 2)
  (h2 : sequence 4 = 81) :
  ∃ (a b : ℕ), a * b = sequence 3 ∧ a = 6 ∧ b = 9 :=
by sorry

end stepa_multiplication_l2526_252678


namespace rectangle_area_with_fixed_dimension_l2526_252695

theorem rectangle_area_with_fixed_dimension (l w : ℕ) : 
  (2 * l + 2 * w = 200) →  -- perimeter is 200 cm
  (w = 30 ∨ l = 30) →      -- one dimension is fixed at 30 cm
  (l * w = 2100)           -- area is 2100 square cm
:= by sorry

end rectangle_area_with_fixed_dimension_l2526_252695


namespace geometric_sequence_sum_l2526_252630

/-- Given a geometric sequence {a_n} with a₁ = 2 and a₁ + a₃ + a₅ = 14,
    prove that 1/a₁ + 1/a₃ + 1/a₅ = 7/8 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
    (h_a1 : a 1 = 2) (h_sum : a 1 + a 3 + a 5 = 14) :
  1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8 := by
  sorry

end geometric_sequence_sum_l2526_252630


namespace divisible_by_120_l2526_252679

theorem divisible_by_120 (m : ℕ) : ∃ k : ℤ, (m ^ 5 : ℤ) - 5 * (m ^ 3) + 4 * m = 120 * k := by
  sorry

end divisible_by_120_l2526_252679


namespace remainder_theorem_l2526_252623

/-- The polynomial P(z) = 4z^4 - 9z^3 + 3z^2 - 17z + 7 -/
def P (z : ℂ) : ℂ := 4 * z^4 - 9 * z^3 + 3 * z^2 - 17 * z + 7

/-- The theorem stating that the remainder of P(z) divided by (z - 2) is -23 -/
theorem remainder_theorem :
  ∃ Q : ℂ → ℂ, P = (fun z ↦ (z - 2) * Q z + (-23)) := by sorry

end remainder_theorem_l2526_252623


namespace area_of_square_on_XY_l2526_252699

-- Define the triangle XYZ
structure RightTriangle where
  XY : ℝ
  YZ : ℝ
  XZ : ℝ
  right_angle : XZ^2 = XY^2 + YZ^2

-- Define the theorem
theorem area_of_square_on_XY (t : RightTriangle) 
  (sum_of_squares : t.XY^2 + t.YZ^2 + t.XZ^2 = 500) : 
  t.XY^2 = 125 := by
  sorry

end area_of_square_on_XY_l2526_252699


namespace storage_blocks_count_l2526_252651

/-- Calculates the number of blocks needed for a rectangular storage --/
def blocksNeeded (length width height thickness : ℕ) : ℕ :=
  let totalVolume := length * width * height
  let interiorLength := length - 2 * thickness
  let interiorWidth := width - 2 * thickness
  let interiorHeight := height - thickness
  let interiorVolume := interiorLength * interiorWidth * interiorHeight
  totalVolume - interiorVolume

/-- Theorem stating the number of blocks needed for the specific storage --/
theorem storage_blocks_count :
  blocksNeeded 20 15 10 2 = 1592 := by
  sorry

end storage_blocks_count_l2526_252651


namespace contrapositive_even_sum_l2526_252654

theorem contrapositive_even_sum (x y : ℤ) :
  (¬(Even (x + y)) → ¬(Even x ∧ Even y)) ↔
  (∀ x y : ℤ, Even x → Even y → Even (x + y)) :=
by sorry

end contrapositive_even_sum_l2526_252654


namespace clock_malfunction_theorem_l2526_252615

/-- Represents a time in 24-hour format -/
structure Time where
  hours : Fin 24
  minutes : Fin 60

/-- Represents the possible changes to a digit due to malfunction -/
inductive DigitChange
  | Increase
  | Decrease

/-- Applies the digit change to a number, wrapping around if necessary -/
def applyDigitChange (n : Fin 10) (change : DigitChange) : Fin 10 :=
  match change with
  | DigitChange.Increase => (n + 1) % 10
  | DigitChange.Decrease => (n + 9) % 10

/-- Applies changes to both digits of a two-digit number -/
def applyTwoDigitChange (n : Fin 100) (tens_change : DigitChange) (units_change : DigitChange) : Fin 100 :=
  let tens := n / 10
  let units := n % 10
  (applyDigitChange tens tens_change) * 10 + (applyDigitChange units units_change)

theorem clock_malfunction_theorem (malfunctioned_time : Time) 
    (h : malfunctioned_time.hours = 9 ∧ malfunctioned_time.minutes = 9) :
    ∃ (original_time : Time) (hours_tens_change hours_units_change minutes_tens_change minutes_units_change : DigitChange),
      original_time.hours = 18 ∧
      original_time.minutes = 18 ∧
      applyTwoDigitChange original_time.hours hours_tens_change hours_units_change = malfunctioned_time.hours ∧
      applyTwoDigitChange original_time.minutes minutes_tens_change minutes_units_change = malfunctioned_time.minutes :=
by sorry

end clock_malfunction_theorem_l2526_252615


namespace sum_of_three_squares_l2526_252632

theorem sum_of_three_squares (K : ℕ) (L : ℤ) (h : L % 8 = 7) :
  ¬ ∃ (a b c : ℤ), 4^K * L = a^2 + b^2 + c^2 := by
sorry

end sum_of_three_squares_l2526_252632


namespace arithmetic_progression_problem_l2526_252697

theorem arithmetic_progression_problem (a d : ℝ) : 
  -- The five numbers form a decreasing arithmetic progression
  (∀ i : Fin 5, (fun i => a - (2 - i) * d) i > (fun i => a - (2 - i.succ) * d) i.succ) →
  -- The sum of their cubes is zero
  ((a - 2*d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2*d)^3 = 0) →
  -- The sum of their fourth powers is 136
  ((a - 2*d)^4 + (a - d)^4 + a^4 + (a + d)^4 + (a + 2*d)^4 = 136) →
  -- The smallest number is -2√2
  a - 2*d = -2 * Real.sqrt 2 := by
sorry


end arithmetic_progression_problem_l2526_252697


namespace seconds_in_minutes_l2526_252669

/-- The number of seconds in one minute -/
def seconds_per_minute : ℝ := 60

/-- The number of minutes we want to convert to seconds -/
def minutes : ℝ := 12.5

/-- Theorem: The number of seconds in 12.5 minutes is 750 -/
theorem seconds_in_minutes : minutes * seconds_per_minute = 750 := by
  sorry

end seconds_in_minutes_l2526_252669


namespace intersection_of_A_and_complement_of_B_l2526_252648

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {2} := by sorry

end intersection_of_A_and_complement_of_B_l2526_252648


namespace triangle_similarity_implies_pc_length_l2526_252691

/-- Triangle ABC with sides AB, BC, and CA -/
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)

/-- Point P on the extension of BC -/
def P : Type := Unit

/-- The length of PC -/
def PC (t : Triangle) (p : P) : ℝ := sorry

/-- Similarity of triangles PAB and PCA -/
def similar_triangles (t : Triangle) (p : P) : Prop := sorry

theorem triangle_similarity_implies_pc_length 
  (t : Triangle) 
  (p : P) 
  (h1 : t.AB = 10) 
  (h2 : t.BC = 9) 
  (h3 : t.CA = 7) 
  (h4 : similar_triangles t p) : 
  PC t p = 1.5 := by sorry

end triangle_similarity_implies_pc_length_l2526_252691


namespace legs_sum_is_ten_l2526_252687

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  leg : ℝ
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = leg * Real.sqrt 2
  perimeter_eq : leg + leg + hypotenuse = 10 + hypotenuse

-- Theorem statement
theorem legs_sum_is_ten (t : IsoscelesRightTriangle) 
  (h : t.hypotenuse = 7.0710678118654755) : 
  t.leg + t.leg = 10 := by
  sorry

end legs_sum_is_ten_l2526_252687


namespace tom_catches_jerry_l2526_252658

/-- Represents the figure-eight track -/
structure Track :=
  (small_loop : ℝ)
  (large_loop : ℝ)
  (h_large_double_small : large_loop = 2 * small_loop)

/-- Represents the runners -/
structure Runner :=
  (speed : ℝ)

theorem tom_catches_jerry (track : Track) (tom jerry : Runner) 
  (h1 : tom.speed = track.small_loop / 10)
  (h2 : jerry.speed = track.small_loop / 20)
  (h3 : tom.speed = 2 * jerry.speed) :
  (2 * track.large_loop) / (tom.speed - jerry.speed) = 40 := by
  sorry

#check tom_catches_jerry

end tom_catches_jerry_l2526_252658


namespace line_perp_plane_properties_l2526_252640

-- Define a structure for a 3D space
structure Space3D where
  Point : Type
  Line : Type
  Plane : Type
  perpendicular : Line → Plane → Prop
  line_in_plane : Line → Plane → Prop
  line_perp_line : Line → Line → Prop

-- Define the theorem
theorem line_perp_plane_properties {S : Space3D} (a : S.Line) (M : S.Plane) :
  (S.perpendicular a M → ∀ (l : S.Line), S.line_in_plane l M → S.line_perp_line a l) ∧
  (∃ (b : S.Line) (N : S.Plane), (∀ (l : S.Line), S.line_in_plane l N → S.line_perp_line b l) ∧ ¬S.perpendicular b N) :=
sorry

end line_perp_plane_properties_l2526_252640


namespace baking_time_undetermined_l2526_252656

/-- Represents the cookie-making process with given information -/
structure CookieBaking where
  total_cookies : ℕ
  mixing_time : ℕ
  eaten_cookies : ℕ
  remaining_cookies : ℕ

/-- States that the baking time cannot be determined from the given information -/
theorem baking_time_undetermined (cb : CookieBaking) 
  (h1 : cb.total_cookies = 32)
  (h2 : cb.mixing_time = 24)
  (h3 : cb.eaten_cookies = 9)
  (h4 : cb.remaining_cookies = 23)
  (h5 : cb.total_cookies = cb.eaten_cookies + cb.remaining_cookies) :
  ¬ ∃ (baking_time : ℕ), baking_time = cb.mixing_time ∨ baking_time ≠ cb.mixing_time :=
by sorry


end baking_time_undetermined_l2526_252656


namespace opposite_violet_is_blue_l2526_252617

-- Define the colors
inductive Color
  | Orange
  | Black
  | Yellow
  | Violet
  | Blue
  | Pink

-- Define a cube
structure Cube where
  faces : Fin 6 → Color

-- Define the three views of the cube
def view1 (c : Cube) : Prop :=
  c.faces 0 = Color.Blue ∧ c.faces 1 = Color.Yellow ∧ c.faces 2 = Color.Orange

def view2 (c : Cube) : Prop :=
  c.faces 0 = Color.Blue ∧ c.faces 1 = Color.Pink ∧ c.faces 2 = Color.Orange

def view3 (c : Cube) : Prop :=
  c.faces 0 = Color.Blue ∧ c.faces 1 = Color.Black ∧ c.faces 2 = Color.Orange

-- Define the opposite face relation
def oppositeFace (i j : Fin 6) : Prop :=
  (i = 0 ∧ j = 5) ∨ (i = 1 ∧ j = 3) ∨ (i = 2 ∧ j = 4) ∨
  (i = 3 ∧ j = 1) ∨ (i = 4 ∧ j = 2) ∨ (i = 5 ∧ j = 0)

-- Theorem statement
theorem opposite_violet_is_blue (c : Cube) :
  (∀ i j : Fin 6, i ≠ j → c.faces i ≠ c.faces j) →
  view1 c → view2 c → view3 c →
  ∃ i j : Fin 6, oppositeFace i j ∧ c.faces i = Color.Violet ∧ c.faces j = Color.Blue :=
sorry

end opposite_violet_is_blue_l2526_252617


namespace smallest_n_square_and_cube_l2526_252647

/-- A perfect square is an integer that is the square of another integer. -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- A perfect cube is an integer that is the cube of another integer. -/
def IsPerfectCube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

/-- The smallest positive integer n such that 5n is a perfect square and 3n is a perfect cube. -/
theorem smallest_n_square_and_cube : 
  (∀ m : ℕ, m > 0 ∧ m < 1125 → ¬(IsPerfectSquare (5 * m) ∧ IsPerfectCube (3 * m))) ∧ 
  (IsPerfectSquare (5 * 1125) ∧ IsPerfectCube (3 * 1125)) := by
  sorry

end smallest_n_square_and_cube_l2526_252647


namespace book_sale_profit_l2526_252605

/-- Calculates the percent profit for a book sale given the cost, markup percentage, and discount percentage. -/
theorem book_sale_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) :
  cost = 50 ∧ markup_percent = 30 ∧ discount_percent = 10 →
  (((cost * (1 + markup_percent / 100)) * (1 - discount_percent / 100) - cost) / cost) * 100 = 17 := by
  sorry

end book_sale_profit_l2526_252605


namespace no_three_consecutive_squares_l2526_252618

/-- An arithmetic progression of natural numbers -/
structure ArithmeticProgression where
  terms : ℕ → ℕ
  common_difference : ℕ
  increasing : ∀ n, terms n < terms (n + 1)
  difference_property : ∀ n, terms (n + 1) - terms n = common_difference
  difference_ends_2019 : common_difference % 10000 = 2019

/-- Three consecutive squares in an arithmetic progression -/
def ThreeConsecutiveSquares (ap : ArithmeticProgression) (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 
    ap.terms n = a^2 ∧ 
    ap.terms (n + 1) = b^2 ∧ 
    ap.terms (n + 2) = c^2

theorem no_three_consecutive_squares (ap : ArithmeticProgression) :
  ¬ ∃ n, ThreeConsecutiveSquares ap n :=
sorry

end no_three_consecutive_squares_l2526_252618


namespace garage_sale_theorem_l2526_252673

def garage_sale_problem (treadmill_price : ℝ) (chest_price : ℝ) (tv_price : ℝ) (total_sales : ℝ) : Prop :=
  treadmill_price = 100 ∧
  chest_price = treadmill_price / 2 ∧
  tv_price = 3 * treadmill_price ∧
  (treadmill_price + chest_price + tv_price) / total_sales = 0.75 ∧
  total_sales = 600

theorem garage_sale_theorem :
  ∃ (treadmill_price chest_price tv_price total_sales : ℝ),
    garage_sale_problem treadmill_price chest_price tv_price total_sales :=
by
  sorry

end garage_sale_theorem_l2526_252673


namespace range_of_a_l2526_252621

open Real

theorem range_of_a (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = x * log x) →
  (∀ x, g x = x^3 + a*x^2 - x + 2) →
  (∀ x > 0, 2 * f x ≤ (deriv g) x + 2) →
  a ≥ -2 := by
  sorry

end range_of_a_l2526_252621


namespace workshop_wolf_prize_laureates_l2526_252674

theorem workshop_wolf_prize_laureates 
  (total_scientists : ℕ) 
  (both_wolf_and_nobel : ℕ) 
  (total_nobel : ℕ) 
  (h1 : total_scientists = 50)
  (h2 : both_wolf_and_nobel = 12)
  (h3 : total_nobel = 23)
  (h4 : ∃ (non_wolf_non_nobel : ℕ), 
        non_wolf_non_nobel + (non_wolf_non_nobel + 3) = total_scientists - both_wolf_and_nobel) :
  ∃ (wolf_laureates : ℕ), wolf_laureates = 31 ∧ 
    wolf_laureates + (total_scientists - wolf_laureates) = total_scientists :=
sorry

end workshop_wolf_prize_laureates_l2526_252674


namespace optimal_distribution_second_day_distribution_l2526_252670

/-- Represents a production line with its processing characteristics -/
structure ProductionLine where
  name : String
  process_time : ℝ → ℝ
  tonnage : ℝ

/-- The company with two production lines -/
structure Company where
  line_a : ProductionLine
  line_b : ProductionLine

/-- Defines the company with given production line characteristics -/
def our_company : Company :=
  { line_a := { name := "A", process_time := (λ a ↦ 4 * a + 1), tonnage := 0 },
    line_b := { name := "B", process_time := (λ b ↦ 2 * b + 3), tonnage := 0 } }

/-- Total raw materials allocated to both production lines -/
def total_raw_materials : ℝ := 5

/-- Theorem stating the optimal distribution of raw materials -/
theorem optimal_distribution (c : Company) (h : c = our_company) :
  ∃ (a b : ℝ),
    a + b = total_raw_materials ∧
    c.line_a.process_time a = c.line_b.process_time b ∧
    a = 2 ∧ b = 3 := by
  sorry

/-- Theorem stating the relationship between m and n for the second day -/
theorem second_day_distribution (m n : ℝ) (h : m + n = 6) :
  2 * m = n → m = 2 ∧ n = 4 := by
  sorry

end optimal_distribution_second_day_distribution_l2526_252670


namespace number_problem_l2526_252657

theorem number_problem : ∃ x : ℚ, (x / 6) * 12 = 10 ∧ x = 5 := by
  sorry

end number_problem_l2526_252657


namespace max_train_collection_l2526_252622

/-- The number of trains Max receives each year -/
def trains_per_year : ℕ := 3

/-- The number of years Max collects trains -/
def collection_years : ℕ := 5

/-- The factor by which Max's parents increase his collection -/
def parents_gift_factor : ℕ := 2

/-- The total number of trains Max has after the collection period and his parents' gift -/
def total_trains : ℕ := trains_per_year * collection_years * parents_gift_factor

theorem max_train_collection :
  total_trains = 30 := by sorry

end max_train_collection_l2526_252622


namespace jovana_shells_problem_l2526_252611

/-- Given that Jovana has an initial amount of shells and needs a total amount to fill her bucket,
    this function calculates the additional amount of shells needed. -/
def additional_shells_needed (initial_amount total_amount : ℕ) : ℕ :=
  total_amount - initial_amount

/-- Theorem stating that Jovana needs to add 12 more pounds of shells to fill her bucket. -/
theorem jovana_shells_problem :
  let initial_amount : ℕ := 5
  let total_amount : ℕ := 17
  additional_shells_needed initial_amount total_amount = 12 := by
  sorry

end jovana_shells_problem_l2526_252611


namespace oranges_per_box_l2526_252636

/-- Given 24 oranges distributed equally among 3 boxes, prove that each box contains 8 oranges. -/
theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (oranges_per_box : ℕ) 
  (h1 : total_oranges = 24) 
  (h2 : num_boxes = 3) 
  (h3 : oranges_per_box * num_boxes = total_oranges) : 
  oranges_per_box = 8 := by
  sorry

end oranges_per_box_l2526_252636


namespace point_distance_product_l2526_252626

theorem point_distance_product (y₁ y₂ : ℝ) : 
  ((-5 - 4)^2 + (y₁ - 5)^2 = 12^2) →
  ((-5 - 4)^2 + (y₂ - 5)^2 = 12^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -38 := by
sorry

end point_distance_product_l2526_252626


namespace folders_needed_l2526_252650

def initial_files : Real := 93.0
def additional_files : Real := 21.0
def files_per_folder : Real := 8.0

theorem folders_needed : 
  ∃ (n : ℕ), n = Int.ceil ((initial_files + additional_files) / files_per_folder) ∧ n = 15 := by
  sorry

end folders_needed_l2526_252650


namespace max_points_tournament_l2526_252620

-- Define the number of teams
def num_teams : ℕ := 8

-- Define the number of top teams with equal points
def num_top_teams : ℕ := 4

-- Define the points for win, draw, and loss
def win_points : ℕ := 3
def draw_points : ℕ := 1
def loss_points : ℕ := 0

-- Define the function to calculate the total number of games
def total_games (n : ℕ) : ℕ := n.choose 2 * 2

-- Define the function to calculate the maximum points for top teams
def max_points_top_team (n : ℕ) (k : ℕ) : ℕ :=
  (k - 1) * 3 + (n - k) * 3 * 2

-- Theorem statement
theorem max_points_tournament :
  max_points_top_team num_teams num_top_teams = 33 :=
sorry

end max_points_tournament_l2526_252620


namespace expression_one_equality_l2526_252635

theorem expression_one_equality : 
  4 * Real.sqrt 54 * 3 * Real.sqrt 2 / (-(3/2) * Real.sqrt (1/3)) = -144 := by
sorry

end expression_one_equality_l2526_252635


namespace peanut_ratio_l2526_252614

theorem peanut_ratio (initial : ℕ) (eaten_by_bonita : ℕ) (remaining : ℕ)
  (h1 : initial = 148)
  (h2 : eaten_by_bonita = 29)
  (h3 : remaining = 82) :
  (initial - remaining - eaten_by_bonita) / initial = 1 / 4 := by
sorry

end peanut_ratio_l2526_252614


namespace probability_of_selection_l2526_252681

theorem probability_of_selection (total_students : ℕ) (xiao_li_in_group : Prop) : 
  total_students = 5 → xiao_li_in_group → (1 : ℚ) / total_students = (1 : ℚ) / 5 := by
  sorry

end probability_of_selection_l2526_252681


namespace h_value_l2526_252649

-- Define polynomials f and h
variable (f h : ℝ → ℝ)

-- Define the conditions
axiom f_def : ∀ x, f x = x^4 - 2*x^3 + x - 1
axiom sum_eq : ∀ x, f x + h x = 3*x^2 + 5*x - 4

-- State the theorem
theorem h_value : ∀ x, h x = -x^4 + 2*x^3 + 3*x^2 + 4*x - 3 :=
sorry

end h_value_l2526_252649


namespace total_apples_is_100_l2526_252607

-- Define the types of apples
inductive AppleType
| Sweet
| Sour

-- Define the price function for apples
def applePrice : AppleType → ℚ
| AppleType.Sweet => 1/2
| AppleType.Sour => 1/10

-- Define the proportion of sweet apples
def sweetProportion : ℚ := 3/4

-- Define the total earnings
def totalEarnings : ℚ := 40

-- Theorem statement
theorem total_apples_is_100 :
  ∃ (n : ℕ), n = 100 ∧
  n * (sweetProportion * applePrice AppleType.Sweet +
       (1 - sweetProportion) * applePrice AppleType.Sour) = totalEarnings :=
by
  sorry


end total_apples_is_100_l2526_252607


namespace camp_cedar_counselors_l2526_252655

def camp_cedar (num_boys : ℕ) (num_girls : ℕ) (boy_ratio : ℕ) (girl_ratio : ℕ) : ℕ :=
  (num_boys + boy_ratio - 1) / boy_ratio + (num_girls + girl_ratio - 1) / girl_ratio

theorem camp_cedar_counselors :
  let num_boys : ℕ := 80
  let num_girls : ℕ := 6 * num_boys - 40
  let boy_ratio : ℕ := 5
  let girl_ratio : ℕ := 12
  camp_cedar num_boys num_girls boy_ratio girl_ratio = 53 := by
  sorry

#eval camp_cedar 80 (6 * 80 - 40) 5 12

end camp_cedar_counselors_l2526_252655


namespace airport_distance_proof_l2526_252688

/-- Calculates the remaining distance to a destination given the total distance,
    driving speed, and time driven. -/
def remaining_distance (total_distance speed time : ℝ) : ℝ :=
  total_distance - speed * time

/-- Theorem stating that given a total distance of 300 km, a driving speed of 60 km/hour,
    and a driving time of 2 hours, the remaining distance to the destination is 180 km. -/
theorem airport_distance_proof :
  remaining_distance 300 60 2 = 180 := by
  sorry

end airport_distance_proof_l2526_252688


namespace initial_girls_count_l2526_252604

theorem initial_girls_count (n : ℕ) : 
  n > 0 →
  (n : ℚ) / 2 - 2 = (2 * n : ℚ) / 5 →
  (n : ℚ) / 2 = 10 :=
by
  sorry

end initial_girls_count_l2526_252604

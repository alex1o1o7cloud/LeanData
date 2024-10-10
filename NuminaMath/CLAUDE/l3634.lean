import Mathlib

namespace bowling_ball_weight_l3634_363439

theorem bowling_ball_weight (canoe_weight : ℝ) (bowling_ball_weight : ℝ) : 
  canoe_weight = 36 →
  6 * bowling_ball_weight = 4 * canoe_weight →
  bowling_ball_weight = 24 := by
sorry

end bowling_ball_weight_l3634_363439


namespace max_value_of_x_l3634_363404

theorem max_value_of_x (x y z : ℝ) (sum_eq : x + y + z = 7) (prod_eq : x*y + x*z + y*z = 12) :
  x ≤ 1 ∧ ∃ (a b : ℝ), a + b + 1 = 7 ∧ a*b + a*1 + b*1 = 12 :=
sorry

end max_value_of_x_l3634_363404


namespace cubic_sequence_problem_l3634_363410

theorem cubic_sequence_problem (y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (eq1 : y₁ + 8*y₂ + 27*y₃ + 64*y₄ + 125*y₅ = 7)
  (eq2 : 8*y₁ + 27*y₂ + 64*y₃ + 125*y₄ + 216*y₅ = 100)
  (eq3 : 27*y₁ + 64*y₂ + 125*y₃ + 216*y₄ + 343*y₅ = 1000) :
  64*y₁ + 125*y₂ + 216*y₃ + 343*y₄ + 512*y₅ = -5999 := by
sorry

end cubic_sequence_problem_l3634_363410


namespace total_mail_delivered_l3634_363402

-- Define the number of junk mail pieces
def junk_mail : ℕ := 6

-- Define the number of magazines
def magazines : ℕ := 5

-- Theorem to prove
theorem total_mail_delivered : junk_mail + magazines = 11 := by
  sorry

end total_mail_delivered_l3634_363402


namespace reading_time_difference_example_l3634_363453

/-- The difference in reading time (in minutes) between two readers for a given book -/
def reading_time_difference (xavier_speed maya_speed : ℕ) (book_pages : ℕ) : ℕ :=
  ((book_pages / maya_speed - book_pages / xavier_speed) * 60)

/-- Theorem: Given Xavier's and Maya's reading speeds and the book length, 
    the difference in reading time is 180 minutes -/
theorem reading_time_difference_example : 
  reading_time_difference 120 60 360 = 180 := by
  sorry

end reading_time_difference_example_l3634_363453


namespace x_range_for_quadratic_inequality_l3634_363434

theorem x_range_for_quadratic_inequality :
  ∀ x : ℝ, (∀ a : ℝ, a ∈ Set.Icc (-1) 1 → x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔
    x ∈ Set.Ioi 3 ∪ Set.Iio 1 := by
  sorry

end x_range_for_quadratic_inequality_l3634_363434


namespace water_price_this_year_l3634_363442

-- Define the price of water last year
def price_last_year : ℝ := 1.6

-- Define the price increase rate
def price_increase_rate : ℝ := 0.2

-- Define Xiao Li's water bill in December last year
def december_bill : ℝ := 17

-- Define Xiao Li's water bill in January this year
def january_bill : ℝ := 30

-- Define the difference in water consumption between January and December
def consumption_difference : ℝ := 5

-- Theorem: The price of residential water this year is 1.92 yuan per cubic meter
theorem water_price_this_year :
  let price_this_year := price_last_year * (1 + price_increase_rate)
  price_this_year = 1.92 ∧
  january_bill / price_this_year - december_bill / price_last_year = consumption_difference :=
by sorry

end water_price_this_year_l3634_363442


namespace odd_function_negative_domain_l3634_363466

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_positive : ∀ x > 0, f x = -x + 1) :
  ∀ x < 0, f x = -x - 1 := by
sorry

end odd_function_negative_domain_l3634_363466


namespace rhombus_inscribed_circle_radius_l3634_363400

theorem rhombus_inscribed_circle_radius 
  (side_length : ℝ) 
  (acute_angle : ℝ) 
  (h : side_length = 8 ∧ acute_angle = 30 * π / 180) : 
  side_length * Real.sin (acute_angle) / 2 = 2 := by
  sorry

end rhombus_inscribed_circle_radius_l3634_363400


namespace koi_fish_count_l3634_363460

/-- Calculates the number of koi fish after 3 weeks given the initial conditions --/
def koi_fish_after_three_weeks (initial_total : ℕ) (koi_added_per_day : ℕ) (goldfish_added_per_day : ℕ) (days : ℕ) (final_goldfish : ℕ) : ℕ :=
  let total_added := (koi_added_per_day + goldfish_added_per_day) * days
  let final_total := initial_total + total_added
  final_total - final_goldfish

/-- Theorem stating that the number of koi fish after 3 weeks is 227 --/
theorem koi_fish_count : koi_fish_after_three_weeks 280 2 5 21 200 = 227 := by
  sorry

end koi_fish_count_l3634_363460


namespace max_value_trig_sum_l3634_363448

theorem max_value_trig_sum (a b φ : ℝ) :
  (∀ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) = Real.sqrt (a^2 + b^2)) :=
by sorry

end max_value_trig_sum_l3634_363448


namespace english_homework_time_l3634_363484

def total_time : ℕ := 180
def math_time : ℕ := 45
def science_time : ℕ := 50
def history_time : ℕ := 25
def project_time : ℕ := 30

theorem english_homework_time :
  total_time - (math_time + science_time + history_time + project_time) = 30 := by
sorry

end english_homework_time_l3634_363484


namespace negation_of_all_divisible_by_two_are_even_l3634_363480

theorem negation_of_all_divisible_by_two_are_even :
  (¬ ∀ n : ℕ, 2 ∣ n → Even n) ↔ (∃ n : ℕ, 2 ∣ n ∧ ¬Even n) := by sorry

end negation_of_all_divisible_by_two_are_even_l3634_363480


namespace spinner_probability_l3634_363468

/-- Given a spinner with three regions A, B, and C, where the probability of
    stopping on A is 1/2 and on B is 1/5, prove that the probability of
    stopping on C is 3/10. -/
theorem spinner_probability (p_A p_B p_C : ℚ) : 
  p_A = 1/2 → p_B = 1/5 → p_A + p_B + p_C = 1 → p_C = 3/10 := by
  sorry

end spinner_probability_l3634_363468


namespace books_from_second_shop_l3634_363429

theorem books_from_second_shop 
  (first_shop_books : ℕ) 
  (first_shop_cost : ℕ) 
  (second_shop_cost : ℕ) 
  (average_price : ℚ) : ℕ :=
  by
    have h1 : first_shop_books = 40 := by sorry
    have h2 : first_shop_cost = 600 := by sorry
    have h3 : second_shop_cost = 240 := by sorry
    have h4 : average_price = 14 := by sorry
    
    -- The number of books from the second shop
    let second_shop_books : ℕ := 20
    
    -- Prove that this satisfies the conditions
    sorry

#check books_from_second_shop

end books_from_second_shop_l3634_363429


namespace unique_pair_solution_l3634_363493

theorem unique_pair_solution : 
  ∃! (x y : ℕ), 
    x > 0 ∧ y > 0 ∧  -- Positive integers
    y ≥ x ∧          -- y ≥ x
    x + y ≤ 20 ∧     -- Sum constraint
    ¬(Nat.Prime (x * y)) ∧  -- Product is composite
    (∀ (a b : ℕ), a > 0 ∧ b > 0 ∧ b ≥ a ∧ a + b ≤ 20 ∧ a * b = x * y → a + b = x + y) ∧  -- Unique sum given product and constraints
    x = 2 ∧ y = 11 :=
by sorry

end unique_pair_solution_l3634_363493


namespace sin_x_squared_not_periodic_l3634_363471

theorem sin_x_squared_not_periodic : ¬ ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, Real.sin ((x + p)^2) = Real.sin (x^2) := by
  sorry

end sin_x_squared_not_periodic_l3634_363471


namespace cases_needed_l3634_363477

def boxes_sold : ℕ := 10
def boxes_per_case : ℕ := 2

theorem cases_needed : boxes_sold / boxes_per_case = 5 := by
  sorry

end cases_needed_l3634_363477


namespace calculation_comparison_l3634_363447

theorem calculation_comparison : 
  (3.04 / 0.25 > 1) ∧ (1.01 * 0.99 < 1) ∧ (0.15 / 0.25 < 1) := by
  sorry

end calculation_comparison_l3634_363447


namespace total_egg_collection_l3634_363414

/-- The number of dozen eggs collected by each person -/
structure EggCollection where
  benjamin : ℚ
  carla : ℚ
  trisha : ℚ
  david : ℚ
  emily : ℚ

/-- The conditions of the egg collection problem -/
def eggCollectionConditions (e : EggCollection) : Prop :=
  e.benjamin = 6 ∧
  e.carla = 3 * e.benjamin ∧
  e.trisha = e.benjamin - 4 ∧
  e.david = 2 * e.trisha ∧
  e.david = e.carla / 2 ∧
  e.emily = 3/4 * e.david ∧
  e.emily = e.trisha + e.trisha / 2

/-- The theorem stating that the total number of dozen eggs collected is 33 -/
theorem total_egg_collection (e : EggCollection) 
  (h : eggCollectionConditions e) : 
  e.benjamin + e.carla + e.trisha + e.david + e.emily = 33 := by
  sorry

end total_egg_collection_l3634_363414


namespace projection_correct_l3634_363465

/-- Given vectors u and t, prove that the projection of u onto t is correct. -/
theorem projection_correct (u t : ℝ × ℝ) : 
  u = (4, -3) → t = (-6, 8) → 
  let proj := ((u.1 * t.1 + u.2 * t.2) / (t.1 * t.1 + t.2 * t.2)) • t
  proj.1 = 288 / 100 ∧ proj.2 = -384 / 100 := by
  sorry

end projection_correct_l3634_363465


namespace sqrt_difference_equality_l3634_363475

theorem sqrt_difference_equality : Real.sqrt (49 + 81) - Real.sqrt (36 - 25) = Real.sqrt 130 - Real.sqrt 11 := by
  sorry

end sqrt_difference_equality_l3634_363475


namespace consecutive_digits_divisible_by_11_l3634_363486

/-- Given four consecutive digits x, x+1, x+2, x+3, the number formed by
    interchanging the first two digits of (1000x + 100(x+1) + 10(x+2) + (x+3))
    is divisible by 11 for any integer x. -/
theorem consecutive_digits_divisible_by_11 (x : ℤ) :
  ∃ k : ℤ, (1000 * (x + 1) + 100 * x + 10 * (x + 2) + (x + 3)) = 11 * k := by
  sorry

end consecutive_digits_divisible_by_11_l3634_363486


namespace tomato_plants_count_l3634_363450

/-- Represents the number of vegetables harvested from each surviving plant. -/
def vegetables_per_plant : ℕ := 7

/-- Represents the total number of vegetables harvested. -/
def total_vegetables : ℕ := 56

/-- Represents the number of eggplant plants. -/
def eggplant_plants : ℕ := 2

/-- Represents the initial number of pepper plants. -/
def initial_pepper_plants : ℕ := 4

/-- Represents the number of pepper plants that died. -/
def dead_pepper_plants : ℕ := 1

theorem tomato_plants_count (T : ℕ) : 
  (T / 2 + eggplant_plants + (initial_pepper_plants - dead_pepper_plants)) * vegetables_per_plant = total_vegetables → 
  T = 6 := by
sorry

end tomato_plants_count_l3634_363450


namespace function_two_zeros_implies_a_range_l3634_363438

/-- If the function y = x + a/x + 1 has two zeros, then a ∈ (-∞, 1/4) -/
theorem function_two_zeros_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ + a / x₁ + 1 = 0 ∧ x₂ + a / x₂ + 1 = 0) →
  a < 1/4 :=
by sorry

end function_two_zeros_implies_a_range_l3634_363438


namespace museum_visit_permutations_l3634_363488

theorem museum_visit_permutations : Nat.factorial 6 = 720 := by
  sorry

end museum_visit_permutations_l3634_363488


namespace power_product_equals_sixteen_l3634_363446

theorem power_product_equals_sixteen (m n : ℤ) (h : 2*m + 3*n - 4 = 0) : 
  (4:ℝ)^m * (8:ℝ)^n = 16 := by
sorry

end power_product_equals_sixteen_l3634_363446


namespace min_value_abc_min_value_exists_l3634_363456

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 1/x + 1/y + 1/z = 9 → a^2 * b^3 * c ≤ x^2 * y^3 * z :=
by sorry

theorem min_value_exists (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) :
  a^2 * b^3 * c = 1/108 :=
by sorry

end min_value_abc_min_value_exists_l3634_363456


namespace n_cube_minus_n_l3634_363426

theorem n_cube_minus_n (n : ℕ) (h : ∃ k : ℕ, 33 * 20 * n = k) : n^3 - n = 388944 := by
  sorry

end n_cube_minus_n_l3634_363426


namespace money_lending_problem_l3634_363415

theorem money_lending_problem (total : ℝ) (rate_A rate_B : ℝ) (time : ℝ) (interest_diff : ℝ) :
  total = 10000 ∧ 
  rate_A = 15 / 100 ∧ 
  rate_B = 18 / 100 ∧ 
  time = 2 ∧ 
  interest_diff = 360 →
  ∃ (amount_A amount_B : ℝ),
    amount_A + amount_B = total ∧
    amount_A * rate_A * time = amount_B * rate_B * time + interest_diff ∧
    amount_B = 4000 := by
  sorry

end money_lending_problem_l3634_363415


namespace burger_cost_proof_l3634_363483

/-- The cost of a burger at McDonald's -/
def burger_cost : ℝ := 5

/-- The cost of one pack of fries -/
def fries_cost : ℝ := 2

/-- The cost of a salad -/
def salad_cost : ℝ := 3 * fries_cost

/-- The total cost of the meal -/
def total_cost : ℝ := 15

theorem burger_cost_proof :
  burger_cost + 2 * fries_cost + salad_cost = total_cost :=
by sorry

end burger_cost_proof_l3634_363483


namespace function_composition_result_l3634_363421

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 3
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem function_composition_result (a b : ℝ) :
  (∀ x, h a b x = x + 9) → a - b = -10 := by
  sorry

end function_composition_result_l3634_363421


namespace product_of_mixed_numbers_l3634_363412

theorem product_of_mixed_numbers :
  let a : Rat := 2 + 1/6
  let b : Rat := 3 + 2/9
  a * b = 377/54 := by
  sorry

end product_of_mixed_numbers_l3634_363412


namespace least_common_multiple_first_ten_l3634_363427

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 := by
sorry

end least_common_multiple_first_ten_l3634_363427


namespace pens_cost_after_discount_and_tax_l3634_363489

/-- The cost of one pen in terms of the cost of one pencil -/
def pen_cost (pencil_cost : ℚ) : ℚ := 5 * pencil_cost

/-- The total cost of pens and pencils -/
def total_cost (pencil_cost : ℚ) : ℚ := 3 * pen_cost pencil_cost + 5 * pencil_cost

/-- The cost of one dozen pens -/
def dozen_pens_cost (pencil_cost : ℚ) : ℚ := 12 * pen_cost pencil_cost

/-- The discount rate applied to one dozen pens -/
def discount_rate : ℚ := 1 / 10

/-- The tax rate applied after the discount -/
def tax_rate : ℚ := 18 / 100

/-- The final cost of one dozen pens after discount and tax -/
def final_cost (pencil_cost : ℚ) : ℚ :=
  let discounted_cost := dozen_pens_cost pencil_cost * (1 - discount_rate)
  discounted_cost * (1 + tax_rate)

theorem pens_cost_after_discount_and_tax :
  ∃ (pencil_cost : ℚ),
    total_cost pencil_cost = 260 ∧
    final_cost pencil_cost = 828.36 := by
  sorry


end pens_cost_after_discount_and_tax_l3634_363489


namespace peters_candies_l3634_363417

theorem peters_candies : ∃ (initial : ℚ), 
  initial > 0 ∧ 
  (1/4 * initial - 13/2 : ℚ) = 6 ∧
  initial = 50 := by
  sorry

end peters_candies_l3634_363417


namespace xyz_value_l3634_363473

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 168) (h2 : y * (z + x) = 180) (h3 : z * (x + y) = 192) :
  x * y * z = 842 := by
  sorry

end xyz_value_l3634_363473


namespace negation_equivalence_l3634_363458

theorem negation_equivalence : 
  (¬ ∃ (x : ℝ), x > 0 ∧ Real.sqrt x ≤ x + 1) ↔ 
  (∀ (x : ℝ), x > 0 → Real.sqrt x > x + 1) := by
sorry

end negation_equivalence_l3634_363458


namespace panthers_score_l3634_363492

theorem panthers_score (total_score cougars_margin : ℕ) 
  (h1 : total_score = 48)
  (h2 : cougars_margin = 20) :
  ∃ (panthers_score cougars_score : ℕ),
    panthers_score + cougars_score = total_score ∧
    cougars_score = panthers_score + cougars_margin ∧
    panthers_score = 14 :=
by sorry

end panthers_score_l3634_363492


namespace fraction_zero_implies_x_equals_three_l3634_363445

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (|x| - 3) / (x + 3) = 0 → x = 3 := by
  sorry

end fraction_zero_implies_x_equals_three_l3634_363445


namespace video_game_expenditure_l3634_363469

/-- The cost of the basketball game -/
def basketball_cost : ℚ := 5.20

/-- The cost of the racing game -/
def racing_cost : ℚ := 4.23

/-- The total cost of video games -/
def total_cost : ℚ := basketball_cost + racing_cost

theorem video_game_expenditure : total_cost = 9.43 := by
  sorry

end video_game_expenditure_l3634_363469


namespace hypotenuse_length_l3634_363479

-- Define a right triangle with side lengths a, b, and c
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : c^2 = a^2 + b^2

-- Define the theorem
theorem hypotenuse_length (t : RightTriangle) 
  (h : Real.sqrt ((t.a - 3)^2 + (t.b - 2)^2) = 0) :
  t.c = 3 ∨ t.c = Real.sqrt 13 := by
  sorry

end hypotenuse_length_l3634_363479


namespace both_correct_probability_l3634_363431

-- Define the probabilities
def prob_first : ℝ := 0.75
def prob_second : ℝ := 0.55
def prob_neither : ℝ := 0.20

-- Theorem statement
theorem both_correct_probability : 
  prob_first + prob_second - (1 - prob_neither) = 0.5 := by
  sorry

end both_correct_probability_l3634_363431


namespace faulty_key_is_seven_or_nine_l3634_363440

/-- Represents a digit key on a keypad -/
inductive Digit : Type
| zero | one | two | three | four | five | six | seven | eight | nine

/-- Represents whether a key press was registered or not -/
inductive KeyPress
| registered
| notRegistered

/-- Represents a sequence of ten attempted key presses -/
def AttemptedSequence := Vector Digit 10

/-- Represents the actual registered sequence after pressing keys -/
def RegisteredSequence := Vector Digit 7

/-- Checks if a digit appears at least five times in a sequence -/
def appearsAtLeastFiveTimes (d : Digit) (s : AttemptedSequence) : Prop := sorry

/-- Checks if the registration pattern of a digit matches the faulty key pattern -/
def matchesFaultyPattern (d : Digit) (s : AttemptedSequence) (r : RegisteredSequence) : Prop := sorry

/-- The main theorem stating that the faulty key must be either 7 or 9 -/
theorem faulty_key_is_seven_or_nine
  (attempted : AttemptedSequence)
  (registered : RegisteredSequence)
  (h1 : ∃ (d : Digit), appearsAtLeastFiveTimes d attempted)
  (h2 : ∀ (d : Digit), appearsAtLeastFiveTimes d attempted → matchesFaultyPattern d attempted registered) :
  ∃ (d : Digit), d = Digit.seven ∨ d = Digit.nine :=
sorry

end faulty_key_is_seven_or_nine_l3634_363440


namespace unique_prime_satisfying_condition_l3634_363478

theorem unique_prime_satisfying_condition : 
  ∀ p : ℕ, Prime p → (Prime (p^3 + p^2 + 11*p + 2) ↔ p = 3) := by sorry

end unique_prime_satisfying_condition_l3634_363478


namespace frustum_height_calc_l3634_363435

/-- Represents a pyramid cut by a plane parallel to its base -/
structure CutPyramid where
  height : ℝ              -- Total height of the pyramid
  cut_height : ℝ          -- Height of the cut part
  area_ratio : ℝ          -- Ratio of upper to lower base areas

/-- The height of the frustum in a cut pyramid -/
def frustum_height (p : CutPyramid) : ℝ := p.height - p.cut_height

/-- Theorem stating the height of the frustum given specific conditions -/
theorem frustum_height_calc (p : CutPyramid) 
  (h1 : p.area_ratio = 1 / 4)
  (h2 : p.cut_height = 3) :
  frustum_height p = 3 := by
  sorry

#check frustum_height_calc

end frustum_height_calc_l3634_363435


namespace infinite_set_A_l3634_363491

/-- Given a function f: ℝ → ℝ satisfying the inequality f²(x) ≤ 2x² f(x/2) for all x,
    and a non-empty set A = {a ∈ ℝ | f(a) > a²}, prove that A is infinite. -/
theorem infinite_set_A (f : ℝ → ℝ) 
    (h1 : ∀ x : ℝ, f x ^ 2 ≤ 2 * x ^ 2 * f (x / 2))
    (A : Set ℝ)
    (h2 : A = {a : ℝ | f a > a ^ 2})
    (h3 : Set.Nonempty A) :
  Set.Infinite A :=
sorry

end infinite_set_A_l3634_363491


namespace two_real_roots_l3634_363409

-- Define the quadratic function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem two_real_roots (a b c : ℝ) 
  (h : ∀ x ∈ Set.Icc (-1) 1, |f a b c x| < 1) :
  ∃ x y : ℝ, x ≠ y ∧ f a b c x = 2 * x^2 - 1 ∧ f a b c y = 2 * y^2 - 1 :=
by sorry

end two_real_roots_l3634_363409


namespace taxi_fare_calculation_l3634_363476

/-- Represents the taxi fare structure -/
structure TaxiFare where
  startingFee : ℝ
  ratePerMile : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.startingFee + tf.ratePerMile * distance

theorem taxi_fare_calculation (tf : TaxiFare) 
  (h1 : tf.startingFee = 20)
  (h2 : totalFare tf 80 = 160)
  : totalFare tf 120 = 230 := by
  sorry

end taxi_fare_calculation_l3634_363476


namespace smallest_n_factor_smallest_n_is_75_l3634_363464

theorem smallest_n_factor (n : ℕ+) : 
  (5^2 ∣ n * (2^5) * (6^2) * (7^3)) ∧ 
  (3^3 ∣ n * (2^5) * (6^2) * (7^3)) →
  n ≥ 75 :=
by sorry

theorem smallest_n_is_75 : 
  ∃ (n : ℕ+), n = 75 ∧ 
  (5^2 ∣ n * (2^5) * (6^2) * (7^3)) ∧ 
  (3^3 ∣ n * (2^5) * (6^2) * (7^3)) ∧
  ∀ (m : ℕ+), m < 75 → 
    ¬((5^2 ∣ m * (2^5) * (6^2) * (7^3)) ∧ 
      (3^3 ∣ m * (2^5) * (6^2) * (7^3))) :=
by sorry

end smallest_n_factor_smallest_n_is_75_l3634_363464


namespace infinite_series_sum_l3634_363459

open Real

noncomputable def seriesTerms (k : ℕ) : ℝ :=
  (7^k) / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem infinite_series_sum : 
  ∑' k, seriesTerms k = 7 := by sorry

end infinite_series_sum_l3634_363459


namespace plate_cup_cost_l3634_363420

/-- Given that 100 plates and 200 cups cost $7.50, prove that 20 plates and 40 cups cost $1.50 -/
theorem plate_cup_cost (plate_rate cup_rate : ℚ) : 
  100 * plate_rate + 200 * cup_rate = (7.5 : ℚ) → 
  20 * plate_rate + 40 * cup_rate = (1.5 : ℚ) := by
  sorry


end plate_cup_cost_l3634_363420


namespace negative_abs_negative_five_l3634_363467

theorem negative_abs_negative_five : -|-5| = -5 := by
  sorry

end negative_abs_negative_five_l3634_363467


namespace quadratic_one_root_l3634_363457

theorem quadratic_one_root (a : ℝ) : 
  (∃! x : ℝ, a * x^2 - 2 * x + 1 = 0) ↔ (a = 0 ∨ a = 1) := by
  sorry

end quadratic_one_root_l3634_363457


namespace area_original_figure_l3634_363461

/-- Given an isosceles trapezoid representing the isometric drawing of a horizontally placed figure,
    with a bottom angle of 60°, legs and top base of length 1,
    the area of the original plane figure is 3√6/2. -/
theorem area_original_figure (bottom_angle : ℝ) (leg_length : ℝ) (top_base : ℝ) : 
  bottom_angle = π / 3 →
  leg_length = 1 →
  top_base = 1 →
  ∃ (area : ℝ), area = (3 * Real.sqrt 6) / 2 := by
  sorry

end area_original_figure_l3634_363461


namespace square_area_error_l3634_363497

theorem square_area_error (S : ℝ) (h : S > 0) :
  let measured_side := 1.05 * S
  let actual_area := S^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 10.25 := by
  sorry

end square_area_error_l3634_363497


namespace existence_of_fractions_l3634_363474

theorem existence_of_fractions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a < b) :
  ∃ (p q r s : ℕ+), 
    (a < (p : ℝ) / q ∧ (p : ℝ) / q < (r : ℝ) / s ∧ (r : ℝ) / s < b) ∧
    (p : ℝ)^2 + (q : ℝ)^2 = (r : ℝ)^2 + (s : ℝ)^2 :=
by sorry

end existence_of_fractions_l3634_363474


namespace event_guests_l3634_363443

theorem event_guests (men : ℕ) (women : ℕ) (children : ℕ) : 
  men = 40 →
  women = men / 2 →
  children + 10 = 30 →
  men + women + children = 80 :=
by
  sorry

end event_guests_l3634_363443


namespace smallest_number_divisible_l3634_363495

theorem smallest_number_divisible (n : ℕ) : n = 1013 ↔ 
  (∀ m : ℕ, m < n → 
    ¬(∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
      m - 5 = 12 * k₁ ∧
      m - 5 = 16 * k₂ ∧
      m - 5 = 18 * k₃ ∧
      m - 5 = 21 * k₄ ∧
      m - 5 = 28 * k₅)) ∧
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    n - 5 = 12 * k₁ ∧
    n - 5 = 16 * k₂ ∧
    n - 5 = 18 * k₃ ∧
    n - 5 = 21 * k₄ ∧
    n - 5 = 28 * k₅) :=
by sorry


end smallest_number_divisible_l3634_363495


namespace fraction_simplification_l3634_363406

theorem fraction_simplification : (20 : ℚ) / 19 * 15 / 28 * 76 / 45 = 95 / 21 := by
  sorry

end fraction_simplification_l3634_363406


namespace book_price_percentage_l3634_363463

/-- Given the original price and current price of a book, prove that the current price is 80% of the original price. -/
theorem book_price_percentage (original_price current_price : ℝ) 
  (h1 : original_price = 25)
  (h2 : current_price = 20) :
  current_price / original_price = 0.8 := by
  sorry

end book_price_percentage_l3634_363463


namespace no_real_roots_for_ff_l3634_363432

/-- A quadratic polynomial function -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The property that f(x) = x has no real roots -/
def NoRealRootsForFX (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≠ x

theorem no_real_roots_for_ff (a b c : ℝ) :
  let f := QuadraticPolynomial a b c
  NoRealRootsForFX f → NoRealRootsForFX (f ∘ f) := by
  sorry

#check no_real_roots_for_ff

end no_real_roots_for_ff_l3634_363432


namespace arithmetic_mean_of_fractions_l3634_363455

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((3 / 8 : ℚ) + (5 / 9 : ℚ)) = 67 / 144 := by
  sorry

end arithmetic_mean_of_fractions_l3634_363455


namespace isaac_pen_purchase_l3634_363401

theorem isaac_pen_purchase : ∃ (pens : ℕ), 
  pens + (12 + 5 * pens) = 108 ∧ pens = 16 := by
  sorry

end isaac_pen_purchase_l3634_363401


namespace sequence_fourth_term_l3634_363485

theorem sequence_fourth_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n^3) : a 4 = 37 := by
  sorry

end sequence_fourth_term_l3634_363485


namespace instantaneous_velocity_zero_l3634_363451

/-- The motion law of an object -/
def S (t : ℝ) : ℝ := t^3 - 6*t^2 + 5

/-- The instantaneous velocity of the object -/
def V (t : ℝ) : ℝ := 3*t^2 - 12*t

theorem instantaneous_velocity_zero (t : ℝ) (h : t > 0) :
  V t = 0 → t = 4 := by sorry

end instantaneous_velocity_zero_l3634_363451


namespace system_solution_existence_l3634_363418

theorem system_solution_existence (a : ℝ) : 
  (∃ (b x y : ℝ), y = x^2 + a ∧ x^2 + y^2 + 2*b^2 = 2*b*(x - y) + 1) ↔ 
  a ≤ Real.sqrt 2 + 1/4 := by
sorry

end system_solution_existence_l3634_363418


namespace z_in_second_quadrant_l3634_363498

def z : ℂ := Complex.I * (Complex.I + 2)

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end z_in_second_quadrant_l3634_363498


namespace perpendicular_lines_slope_l3634_363482

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, a * x + 2 * y + 2 = 0 → 3 * x - y - 2 = 0 → 
    (a * x + 2 * y + 2 = 0 ∧ 3 * x - y - 2 = 0) → 
    ((-a/2) * 3 = -1)) → 
  a = 2/3 := by
sorry

end perpendicular_lines_slope_l3634_363482


namespace black_area_proof_l3634_363441

theorem black_area_proof (white_area black_area : ℝ) : 
  white_area + black_area = 9^2 + 5^2 →
  white_area + 2 * black_area = 11^2 + 7^2 →
  black_area = 64 := by
  sorry

end black_area_proof_l3634_363441


namespace insurance_agents_count_l3634_363433

/-- The number of claims Jan can handle -/
def jan_claims : ℕ := 20

/-- The number of claims John can handle -/
def john_claims : ℕ := jan_claims + jan_claims * 30 / 100

/-- The number of claims Missy can handle -/
def missy_claims : ℕ := john_claims + 15

/-- The total number of agents -/
def num_agents : ℕ := 3

theorem insurance_agents_count :
  missy_claims = 41 → num_agents = 3 := by
  sorry

end insurance_agents_count_l3634_363433


namespace min_value_expression_l3634_363487

theorem min_value_expression (a b c d : ℝ) (h1 : b > c) (h2 : c > d) (h3 : d > a) (h4 : b ≠ 0) :
  (a + b)^2 + (b - c)^2 + (c - d)^2 + (d - a)^2 ≥ b^2 := by
  sorry

end min_value_expression_l3634_363487


namespace yearly_savings_multiple_l3634_363422

theorem yearly_savings_multiple (monthly_salary : ℝ) (h : monthly_salary > 0) :
  let monthly_spending := 0.75 * monthly_salary
  let monthly_savings := monthly_salary - monthly_spending
  let yearly_savings := 12 * monthly_savings
  yearly_savings = 4 * monthly_spending :=
by sorry

end yearly_savings_multiple_l3634_363422


namespace largest_digit_divisible_by_six_l3634_363408

theorem largest_digit_divisible_by_six : 
  ∃ (M : ℕ), M ≤ 9 ∧ (45670 + M) % 6 = 0 ∧ 
  ∀ (N : ℕ), N ≤ 9 ∧ (45670 + N) % 6 = 0 → N ≤ M :=
by sorry

end largest_digit_divisible_by_six_l3634_363408


namespace faucet_filling_time_l3634_363419

/-- Given that five faucets fill a 150-gallon tub in 9 minutes,
    prove that ten faucets will fill a 75-gallon tub in 135 seconds. -/
theorem faucet_filling_time 
  (initial_faucets : ℕ) 
  (initial_volume : ℝ) 
  (initial_time : ℝ) 
  (target_faucets : ℕ) 
  (target_volume : ℝ) 
  (h1 : initial_faucets = 5) 
  (h2 : initial_volume = 150) 
  (h3 : initial_time = 9) 
  (h4 : target_faucets = 10) 
  (h5 : target_volume = 75) : 
  (target_volume / target_faucets) * (initial_time / (initial_volume / initial_faucets)) * 60 = 135 := by
  sorry

#check faucet_filling_time

end faucet_filling_time_l3634_363419


namespace complete_square_k_value_l3634_363407

theorem complete_square_k_value (x : ℝ) : 
  ∃ (p k : ℝ), (x^2 - 6*x + 5 = 0) ↔ ((x - p)^2 = k) ∧ k = 4 := by
sorry

end complete_square_k_value_l3634_363407


namespace factorial_ratio_equals_sixty_sevenths_l3634_363490

theorem factorial_ratio_equals_sixty_sevenths : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end factorial_ratio_equals_sixty_sevenths_l3634_363490


namespace zero_point_in_interval_l3634_363444

/-- The function f(x) = 2bx - 3b + 1 has a zero point in (-1, 1) iff b ∈ (1/5, 1) -/
theorem zero_point_in_interval (b : ℝ) :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, 2 * b * x - 3 * b + 1 = 0) ↔ b ∈ Set.Ioo (1/5 : ℝ) 1 := by
  sorry


end zero_point_in_interval_l3634_363444


namespace two_numbers_difference_l3634_363481

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 20)
  (square_diff : x^2 - y^2 = 200)
  (diff_eq : x - y = 10) : x - y = 10 := by
  sorry

end two_numbers_difference_l3634_363481


namespace min_sum_given_product_l3634_363403

theorem min_sum_given_product (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 8) :
  ∃ (min : ℝ), min = 6 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x * y * z = 8 → x + y + z ≥ min :=
by
  sorry

end min_sum_given_product_l3634_363403


namespace special_sequence_property_l3634_363462

/-- A sequence of positive real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  (∀ m n : ℕ, m > 0 → n > 0 → (m + n : ℝ) * a (m + n) ≤ a m + a n) ∧
  (∀ i : ℕ, i > 0 → a i > 0)

/-- The main theorem to be proved -/
theorem special_sequence_property (a : ℕ → ℝ) (h : SpecialSequence a) : 
  1 / a 200 > 4 * 10^7 := by sorry

end special_sequence_property_l3634_363462


namespace concentric_circles_ratio_l3634_363496

theorem concentric_circles_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 * (2 * Real.pi * r₁) = 48 / 360 * (2 * Real.pi * r₂)) →
  (r₁ / r₂ = 4 / 5 ∧ (r₁^2 / r₂^2 = 16 / 25)) := by
  sorry

end concentric_circles_ratio_l3634_363496


namespace tangent_slope_circle_tangent_slope_specific_circle_l3634_363449

theorem tangent_slope_circle (center : ℝ × ℝ) (tangent_point : ℝ × ℝ) : ℝ :=
  let center_x : ℝ := center.1
  let center_y : ℝ := center.2
  let tangent_x : ℝ := tangent_point.1
  let tangent_y : ℝ := tangent_point.2
  let radius_slope : ℝ := (tangent_y - center_y) / (tangent_x - center_x)
  let tangent_slope : ℝ := -1 / radius_slope
  tangent_slope

theorem tangent_slope_specific_circle : 
  tangent_slope_circle (2, 3) (7, 8) = -1 := by
  sorry

end tangent_slope_circle_tangent_slope_specific_circle_l3634_363449


namespace bus_probability_l3634_363428

theorem bus_probability (p3 p6 : ℝ) (h1 : p3 = 0.20) (h2 : p6 = 0.60) :
  p3 + p6 = 0.80 := by
  sorry

end bus_probability_l3634_363428


namespace exists_self_power_congruence_l3634_363413

theorem exists_self_power_congruence : ∃ N : ℕ, 
  (10^2000 ≤ N) ∧ (N < 10^2001) ∧ (N ≡ N^2001 [ZMOD 10^2001]) := by
  sorry

end exists_self_power_congruence_l3634_363413


namespace worker_count_proof_l3634_363411

theorem worker_count_proof : ∃ (x y : ℕ), 
  y = (15 * x) / 19 ∧ 
  (4 * y) / 7 < 1000 ∧ 
  (3 * x) / 5 > 1000 ∧ 
  x = 1995 ∧ 
  y = 1575 := by
sorry

end worker_count_proof_l3634_363411


namespace paul_reading_time_l3634_363452

/-- The number of hours Paul spent reading after nine weeks -/
def reading_hours (books_per_week : ℕ) (pages_per_book : ℕ) (pages_per_hour : ℕ) (weeks : ℕ) : ℕ :=
  books_per_week * pages_per_book * weeks / pages_per_hour

/-- Theorem stating that Paul spent 540 hours reading after nine weeks -/
theorem paul_reading_time : reading_hours 10 300 50 9 = 540 := by
  sorry

end paul_reading_time_l3634_363452


namespace prism_volume_sum_l3634_363470

theorem prism_volume_sum (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  Nat.lcm a b = 72 →
  Nat.lcm a c = 24 →
  Nat.lcm b c = 18 →
  (∃ (a_min b_min c_min a_max b_max c_max : ℕ),
    (∀ a' b' c' : ℕ, 
      Nat.lcm a' b' = 72 → Nat.lcm a' c' = 24 → Nat.lcm b' c' = 18 →
      a' * b' * c' ≥ a_min * b_min * c_min) ∧
    (∀ a' b' c' : ℕ, 
      Nat.lcm a' b' = 72 → Nat.lcm a' c' = 24 → Nat.lcm b' c' = 18 →
      a' * b' * c' ≤ a_max * b_max * c_max) ∧
    a_min * b_min * c_min + a_max * b_max * c_max = 3024) := by
  sorry

#check prism_volume_sum

end prism_volume_sum_l3634_363470


namespace factorization_proof_l3634_363437

theorem factorization_proof (y : ℝ) : 81 * y^19 + 162 * y^38 = 81 * y^19 * (1 + 2 * y^19) := by
  sorry

end factorization_proof_l3634_363437


namespace y_axis_symmetry_of_P_l3634_363405

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The y-axis symmetry operation on a point -/
def yAxisSymmetry (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

/-- The theorem stating that the y-axis symmetry of P(0, -2, 3) is (0, -2, -3) -/
theorem y_axis_symmetry_of_P :
  let P : Point3D := { x := 0, y := -2, z := 3 }
  yAxisSymmetry P = { x := 0, y := -2, z := -3 } := by
  sorry

end y_axis_symmetry_of_P_l3634_363405


namespace vector_sum_magnitude_l3634_363499

/-- Given two plane vectors a and b, prove that |a + 2b| = √7 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  (a.fst = 1 ∧ a.snd = 0) →  -- a = (1,0)
  ‖b‖ = 1 →  -- |b| = 1
  Real.cos (Real.pi / 3) = (a.fst * b.fst + a.snd * b.snd) / (‖a‖ * ‖b‖) →  -- angle between a and b is 60°
  ‖a + 2 • b‖ = Real.sqrt 7 := by
sorry

end vector_sum_magnitude_l3634_363499


namespace product_quantity_relationship_l3634_363430

/-- The initial budget in yuan -/
def initial_budget : ℝ := 1500

/-- The price increase of product A in yuan -/
def price_increase_A : ℝ := 1.5

/-- The price increase of product B in yuan -/
def price_increase_B : ℝ := 1

/-- The reduction in quantity of product A in the first scenario -/
def quantity_reduction_A1 : ℝ := 10

/-- The budget excess in the first scenario -/
def budget_excess : ℝ := 29

/-- The reduction in quantity of product A in the second scenario -/
def quantity_reduction_A2 : ℝ := 5

/-- The total cost in the second scenario -/
def total_cost_scenario2 : ℝ := 1563.5

theorem product_quantity_relationship (x y a b : ℝ) :
  (a * x + b * y = initial_budget) →
  ((a + price_increase_A) * (x - quantity_reduction_A1) + (b + price_increase_B) * y = initial_budget + budget_excess) →
  ((a + 1) * (x - quantity_reduction_A2) + (b + 1) * y = total_cost_scenario2) →
  (2 * x + y > 205) →
  (2 * x + y < 210) →
  (x + 2 * y = 186) := by
sorry

end product_quantity_relationship_l3634_363430


namespace binomial_prob_half_l3634_363472

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: If X ~ B(n, p) with E(X) = 6 and D(X) = 3, then p = 1/2 -/
theorem binomial_prob_half (X : BinomialRV) 
  (h_exp : expectation X = 6)
  (h_var : variance X = 3) : 
  X.p = 1/2 := by
  sorry

end binomial_prob_half_l3634_363472


namespace tank_capacity_l3634_363436

theorem tank_capacity (fill_time_A fill_time_B drain_rate_C combined_fill_time : ℝ) 
  (h1 : fill_time_A = 12)
  (h2 : fill_time_B = 20)
  (h3 : drain_rate_C = 45)
  (h4 : combined_fill_time = 15) :
  ∃ V : ℝ, V = 675 ∧ 
    (V / fill_time_A + V / fill_time_B - drain_rate_C = V / combined_fill_time) :=
by sorry

end tank_capacity_l3634_363436


namespace function_min_value_l3634_363416

/-- Given a function f(x) = (1/3)x³ - x + m with a maximum value of 1,
    prove that its minimum value is -1/3 -/
theorem function_min_value 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ x, f x = (1/3) * x^3 - x + m) 
  (h2 : ∃ x, f x = 1 ∧ ∀ y, f y ≤ 1) : 
  ∃ x, f x = -1/3 ∧ ∀ y, f y ≥ -1/3 :=
sorry

end function_min_value_l3634_363416


namespace inscribed_circle_rectangle_area_l3634_363424

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (length width : ℝ),
    r = 7 →
    length = 3 * width →
    width = 2 * r →
    length * width = 588 :=
by
  sorry

end inscribed_circle_rectangle_area_l3634_363424


namespace problem_1_l3634_363425

theorem problem_1 : (2/3 - 1/12 - 1/15) * (-60) = -31 := by
  sorry

end problem_1_l3634_363425


namespace max_triangle_area_l3634_363454

/-- The parabola function y = x^2 - 4x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The area of triangle ABC given p -/
def triangle_area (p : ℝ) : ℝ := 2 * |((p - 1) * (p - 3))|

theorem max_triangle_area :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 4 ∧
  f 0 = 3 ∧ f 4 = 3 ∧ 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → triangle_area x ≤ 2) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ triangle_area x = 2) :=
by sorry

end max_triangle_area_l3634_363454


namespace number_divided_by_ratio_l3634_363494

theorem number_divided_by_ratio (x : ℝ) : 
  0.55 * x = 4.235 → x / 0.55 = 14 := by
  sorry

end number_divided_by_ratio_l3634_363494


namespace quadratic_solution_implies_sum_l3634_363423

theorem quadratic_solution_implies_sum (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0 → 2 * a + 4 * b = -2 := by
  sorry

end quadratic_solution_implies_sum_l3634_363423

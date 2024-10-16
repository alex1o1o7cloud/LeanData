import Mathlib

namespace NUMINAMATH_CALUDE_largest_non_representable_largest_non_representable_proof_l2244_224465

def is_representable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 5 * a + 8 * b + 12 * c

theorem largest_non_representable : ℕ :=
  19

theorem largest_non_representable_proof :
  (¬ is_representable largest_non_representable) ∧
  (∀ m : ℕ, m > largest_non_representable → is_representable m) :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_largest_non_representable_proof_l2244_224465


namespace NUMINAMATH_CALUDE_unique_valid_quintuple_l2244_224404

/-- A quintuple of nonnegative real numbers satisfying the given conditions -/
structure ValidQuintuple where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  nonneg_a : 0 ≤ a
  nonneg_b : 0 ≤ b
  nonneg_c : 0 ≤ c
  nonneg_d : 0 ≤ d
  nonneg_e : 0 ≤ e
  condition1 : a^2 + b^2 + c^3 + d^3 + e^3 = 5
  condition2 : (a + b + c + d + e) * (a^3 + b^3 + c^2 + d^2 + e^2) = 25

/-- There exists exactly one valid quintuple -/
theorem unique_valid_quintuple : ∃! q : ValidQuintuple, True :=
  sorry

end NUMINAMATH_CALUDE_unique_valid_quintuple_l2244_224404


namespace NUMINAMATH_CALUDE_simplify_fraction_l2244_224490

theorem simplify_fraction : (130 : ℚ) / 16900 * 65 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2244_224490


namespace NUMINAMATH_CALUDE_min_120_degree_turns_l2244_224433

/-- A triangular graph representing a city --/
structure TriangularCity where
  /-- The number of triangular blocks in the city --/
  blocks : Nat
  /-- The number of intersections (squares) in the city --/
  intersections : Nat
  /-- The path taken by the tourist --/
  tourist_path : List Nat
  /-- Ensures the number of blocks is 16 --/
  blocks_count : blocks = 16
  /-- Ensures the number of intersections is 15 --/
  intersections_count : intersections = 15
  /-- Ensures the tourist visits each intersection exactly once --/
  path_visits_all_once : tourist_path.length = intersections ∧ tourist_path.Nodup

/-- The number of 120° turns in a given path --/
def count_120_degree_turns (path : List Nat) : Nat :=
  sorry

/-- Theorem stating that a tourist in a triangular city must make at least 4 turns of 120° --/
theorem min_120_degree_turns (city : TriangularCity) :
  count_120_degree_turns city.tourist_path ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_120_degree_turns_l2244_224433


namespace NUMINAMATH_CALUDE_marys_birthday_money_l2244_224493

theorem marys_birthday_money (M : ℚ) : 
  (3/4 : ℚ) * M - (1/5 : ℚ) * ((3/4 : ℚ) * M) = 60 → M = 100 := by
  sorry

end NUMINAMATH_CALUDE_marys_birthday_money_l2244_224493


namespace NUMINAMATH_CALUDE_evaluate_power_l2244_224407

theorem evaluate_power : (64 : ℝ) ^ (3/4) = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_power_l2244_224407


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_l2244_224429

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if a circle with diameter equal to the distance between the foci
    intersects one of the hyperbola's asymptotes at point (4, 3),
    then a = 4 and b = 3 -/
theorem hyperbola_circle_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c^2 = 16 + 9 ∧ 3 = (b / a) * 4) → a = 4 ∧ b = 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_l2244_224429


namespace NUMINAMATH_CALUDE_point_A_satisfies_condition_l2244_224485

/-- The line on which point P moves -/
def line_P (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The line that always passes through point A -/
def line_A (a b x y : ℝ) : Prop := 3 * a * x + 4 * b * y = 12

/-- Point A -/
def point_A : ℝ × ℝ := (1, 1)

theorem point_A_satisfies_condition :
  ∀ a b : ℝ, line_P a b → line_A a b (point_A.1) (point_A.2) :=
sorry

end NUMINAMATH_CALUDE_point_A_satisfies_condition_l2244_224485


namespace NUMINAMATH_CALUDE_card_statements_l2244_224432

/-- Represents the number of true statements on the card -/
def TrueStatements : Nat → Prop
  | 0 => True
  | 1 => False
  | 2 => False
  | 3 => False
  | 4 => False
  | 5 => False
  | _ => False

/-- The five statements on the card -/
def Statement : Nat → Prop
  | 1 => TrueStatements 1
  | 2 => TrueStatements 2
  | 3 => TrueStatements 3
  | 4 => TrueStatements 4
  | 5 => TrueStatements 5
  | _ => False

/-- Theorem stating that the number of true statements is 0 -/
theorem card_statements :
  (∀ n : Nat, Statement n ↔ TrueStatements n) →
  TrueStatements 0 := by
  sorry

end NUMINAMATH_CALUDE_card_statements_l2244_224432


namespace NUMINAMATH_CALUDE_rays_grocery_bill_l2244_224450

def calculate_total_bill (hamburger_price : ℚ) (cracker_price : ℚ) (vegetable_price : ℚ) 
  (vegetable_quantity : ℕ) (cheese_price : ℚ) (chicken_price : ℚ) (cereal_price : ℚ) 
  (rewards_discount : ℚ) (meat_cheese_discount : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let discounted_hamburger := hamburger_price * (1 - rewards_discount)
  let discounted_crackers := cracker_price * (1 - rewards_discount)
  let discounted_vegetables := vegetable_price * (1 - rewards_discount) * vegetable_quantity
  let discounted_cheese := cheese_price * (1 - meat_cheese_discount)
  let discounted_chicken := chicken_price * (1 - meat_cheese_discount)
  let subtotal := discounted_hamburger + discounted_crackers + discounted_vegetables + 
                  discounted_cheese + discounted_chicken + cereal_price
  let total := subtotal * (1 + sales_tax_rate)
  total

theorem rays_grocery_bill : 
  calculate_total_bill 5 (7/2) 2 4 (7/2) (13/2) 4 (1/10) (1/20) (7/100) = (3035/100) := by
  sorry

end NUMINAMATH_CALUDE_rays_grocery_bill_l2244_224450


namespace NUMINAMATH_CALUDE_quadratic_one_root_l2244_224456

theorem quadratic_one_root (m : ℝ) : m > 0 ∧ 
  (∃! x : ℝ, x^2 + 6*m*x + 3*m = 0) ↔ m = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l2244_224456


namespace NUMINAMATH_CALUDE_painters_work_days_theorem_l2244_224403

/-- The number of work-days required for a given number of painters to complete a job,
    assuming the product of painters and work-days is constant. -/
def work_days (painters : ℕ) (total_work : ℚ) : ℚ :=
  total_work / painters

theorem painters_work_days_theorem (total_work : ℚ) :
  let five_painters_days : ℚ := 3/2
  let four_painters_days : ℚ := work_days 4 (5 * five_painters_days)
  four_painters_days = 15/8 := by sorry

end NUMINAMATH_CALUDE_painters_work_days_theorem_l2244_224403


namespace NUMINAMATH_CALUDE_trailer_homes_count_l2244_224422

/-- Represents the number of new trailer homes added -/
def new_homes : ℕ := 17

/-- The initial number of trailer homes -/
def initial_homes : ℕ := 25

/-- The initial average age of trailer homes (in years) -/
def initial_avg_age : ℕ := 15

/-- The time elapsed since the initial state (in years) -/
def time_elapsed : ℕ := 3

/-- The current average age of all trailer homes (in years) -/
def current_avg_age : ℕ := 12

theorem trailer_homes_count :
  (initial_homes * (initial_avg_age + time_elapsed) + new_homes * time_elapsed) / 
  (initial_homes + new_homes) = current_avg_age := by sorry

end NUMINAMATH_CALUDE_trailer_homes_count_l2244_224422


namespace NUMINAMATH_CALUDE_library_repacking_l2244_224413

theorem library_repacking (initial_packages : ℕ) (pamphlets_per_initial_package : ℕ) (pamphlets_per_new_package : ℕ) : 
  initial_packages = 1450 →
  pamphlets_per_initial_package = 42 →
  pamphlets_per_new_package = 45 →
  (initial_packages * pamphlets_per_initial_package) % pamphlets_per_new_package = 15 :=
by
  sorry

#check library_repacking

end NUMINAMATH_CALUDE_library_repacking_l2244_224413


namespace NUMINAMATH_CALUDE_f_composition_eq_log_range_l2244_224452

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then (1/2) * x - 1/2 else Real.log x

theorem f_composition_eq_log_range (a : ℝ) :
  f (f a) = Real.log (f a) → a ∈ Set.Ici (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_f_composition_eq_log_range_l2244_224452


namespace NUMINAMATH_CALUDE_expression_evaluation_l2244_224426

theorem expression_evaluation : (4^2 - 4) + (5^2 - 5) - (7^3 - 7) + (3^2 - 3) = -298 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2244_224426


namespace NUMINAMATH_CALUDE_bounded_sequence_characterization_l2244_224498

def sequence_rule (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 2) = (a (n + 1) + a n) / (Nat.gcd (a (n + 1)) (a n))

def is_bounded (a : ℕ → ℕ) : Prop :=
  ∃ M, ∀ n, a n ≤ M

theorem bounded_sequence_characterization (a : ℕ → ℕ) :
  (∀ n, a n > 0) →
  sequence_rule a →
  is_bounded a ↔ a 1 = 2 ∧ a 2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_bounded_sequence_characterization_l2244_224498


namespace NUMINAMATH_CALUDE_career_preference_degrees_l2244_224447

/-- Represents the ratio of male to female students in a class -/
structure GenderRatio where
  male : ℕ
  female : ℕ

/-- Represents the number of students preferring a career -/
structure CareerPreference where
  male : ℕ
  female : ℕ

/-- Calculates the degrees in a circle graph for a career preference -/
def degreesForPreference (ratio : GenderRatio) (pref : CareerPreference) : ℚ :=
  360 * (pref.male + pref.female : ℚ) / (ratio.male + ratio.female : ℚ)

theorem career_preference_degrees 
  (ratio : GenderRatio) 
  (pref : CareerPreference) : 
  ratio.male = 2 ∧ ratio.female = 3 ∧ pref.male = 1 ∧ pref.female = 1 → 
  degreesForPreference ratio pref = 144 := by
  sorry

end NUMINAMATH_CALUDE_career_preference_degrees_l2244_224447


namespace NUMINAMATH_CALUDE_max_voters_is_five_l2244_224451

/-- Represents a movie rating system where:
    - Scores are integers from 0 to 10
    - The rating is the sum of scores divided by the number of voters
    - At moment T, the rating is an integer
    - After moment T, each new vote decreases the rating by one unit -/
structure MovieRating where
  scores : List ℤ
  rating_at_T : ℤ

/-- The maximum number of viewers who could have voted after moment T
    while maintaining the property that each new vote decreases the rating by one unit -/
def max_voters_after_T (mr : MovieRating) : ℕ :=
  sorry

/-- All scores are between 0 and 10 -/
axiom scores_range (mr : MovieRating) : ∀ s ∈ mr.scores, 0 ≤ s ∧ s ≤ 10

/-- The rating at moment T is the sum of scores divided by the number of voters -/
axiom rating_calculation (mr : MovieRating) :
  mr.rating_at_T = (mr.scores.sum / mr.scores.length : ℤ)

/-- After moment T, each new vote decreases the rating by exactly one unit -/
axiom rating_decrease (mr : MovieRating) (new_score : ℤ) :
  let new_rating := ((mr.scores.sum + new_score) / (mr.scores.length + 1) : ℤ)
  new_rating = mr.rating_at_T - 1

/-- The maximum number of viewers who could have voted after moment T is 5 -/
theorem max_voters_is_five (mr : MovieRating) :
  max_voters_after_T mr = 5 :=
sorry

end NUMINAMATH_CALUDE_max_voters_is_five_l2244_224451


namespace NUMINAMATH_CALUDE_function_equality_l2244_224474

-- Define the function type
def FunctionType := ℝ → ℝ → ℝ → ℝ

-- State the theorem
theorem function_equality (f : FunctionType) 
  (h : ∀ x y z : ℝ, f x y z = 2 * f z x y) : 
  ∀ x y z : ℝ, f x y z = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2244_224474


namespace NUMINAMATH_CALUDE_circle_radius_is_six_l2244_224418

/-- A regular hexagon with side length 6 -/
structure RegularHexagon where
  side_length : ℝ
  is_regular : side_length = 6

/-- A circle passing through two vertices of a hexagon and tangent to an extended side -/
structure CircleOnHexagon (h : RegularHexagon) where
  center : ℝ × ℝ
  passes_through_A : True  -- Simplified condition
  passes_through_E : True  -- Simplified condition
  tangent_to_CD_extension : True  -- Simplified condition

/-- The radius of a circle on a regular hexagon with the given properties is 6 -/
theorem circle_radius_is_six (h : RegularHexagon) (c : CircleOnHexagon h) :
  let r := Real.sqrt ((c.center.1 - 3)^2 + (c.center.2 - 3*Real.sqrt 3)^2)
  r = 6 := by sorry

end NUMINAMATH_CALUDE_circle_radius_is_six_l2244_224418


namespace NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l2244_224473

/-- Given two plane vectors a and b that are perpendicular, prove that their difference has a magnitude of 2 -/
theorem perpendicular_vectors_difference_magnitude
  (x : ℝ)
  (a : ℝ × ℝ := (4^x, 2^x))
  (b : ℝ × ℝ := (1, (2^x - 2)/2^x))
  (h : a.1 * b.1 + a.2 * b.2 = 0) : 
  ‖(a.1 - b.1, a.2 - b.2)‖ = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l2244_224473


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l2244_224496

theorem price_reduction_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 250)
  (h2 : new_price = 200) : 
  (original_price - new_price) / original_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l2244_224496


namespace NUMINAMATH_CALUDE_mobile_purchase_price_l2244_224438

def grinder_price : ℝ := 15000
def grinder_loss_percent : ℝ := 0.04
def mobile_profit_percent : ℝ := 0.10
def total_profit : ℝ := 200

theorem mobile_purchase_price (mobile_price : ℝ) : 
  (grinder_price * (1 - grinder_loss_percent) + mobile_price * (1 + mobile_profit_percent)) - 
  (grinder_price + mobile_price) = total_profit → 
  mobile_price = 8000 := by
sorry

end NUMINAMATH_CALUDE_mobile_purchase_price_l2244_224438


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l2244_224457

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The function y = (x+1)(x-a) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_a_equals_one :
  ∃ a : ℝ, IsEven (f a) → a = 1 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l2244_224457


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2244_224444

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2244_224444


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l2244_224435

theorem regular_polygon_with_150_degree_angles (n : ℕ) : 
  (n ≥ 3) →  -- Ensuring the polygon has at least 3 sides
  (∀ angle : ℝ, angle = 150 → 180 * (n - 2) = n * angle) →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l2244_224435


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l2244_224412

def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_in_second_quadrant :
  let z : ℂ := -1 + 3*Complex.I
  second_quadrant z :=
by
  sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l2244_224412


namespace NUMINAMATH_CALUDE_monopolist_optimal_quantity_l2244_224411

/-- Represents the demand function for a monopolist's product -/
def demand (P : ℝ) : ℝ := 10 - P

/-- Represents the revenue function for the monopolist -/
def revenue (Q : ℝ) : ℝ := Q * (10 - Q)

/-- Represents the profit function for the monopolist -/
def profit (Q : ℝ) : ℝ := revenue Q

/-- The maximum quantity of goods the monopolist can sell -/
def max_quantity : ℝ := 10

/-- Theorem: The monopolist maximizes profit by selling 5 units -/
theorem monopolist_optimal_quantity :
  ∃ (Q : ℝ), Q = 5 ∧ 
  Q ≤ max_quantity ∧
  ∀ (Q' : ℝ), Q' ≤ max_quantity → profit Q' ≤ profit Q :=
sorry

end NUMINAMATH_CALUDE_monopolist_optimal_quantity_l2244_224411


namespace NUMINAMATH_CALUDE_squares_in_figure_50_l2244_224436

/-- The function representing the number of squares in the nth figure -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- Theorem stating that the 50th figure has 7651 squares -/
theorem squares_in_figure_50 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 50 = 7651 := by
  sorry

#eval f 50  -- This will evaluate f(50) and should output 7651

end NUMINAMATH_CALUDE_squares_in_figure_50_l2244_224436


namespace NUMINAMATH_CALUDE_prime_order_existence_l2244_224472

theorem prime_order_existence (p : ℕ) (hp : Prime p) :
  ∃ k : ℤ, (∀ m : ℕ, m < p - 1 → k ^ m % p ≠ 1) ∧
            k ^ (p - 1) % p = 1 ∧
            (∀ m : ℕ, m < p * (p - 1) → k ^ m % (p ^ 2) ≠ 1) ∧
            k ^ (p * (p - 1)) % (p ^ 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_order_existence_l2244_224472


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2244_224414

def I : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem intersection_of_M_and_N : M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2244_224414


namespace NUMINAMATH_CALUDE_number_of_products_l2244_224425

/-- Given fixed cost, marginal cost, and total cost, prove the number of products. -/
theorem number_of_products (fixed_cost marginal_cost total_cost : ℚ) (n : ℕ) : 
  fixed_cost = 12000 →
  marginal_cost = 200 →
  total_cost = 16000 →
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_products_l2244_224425


namespace NUMINAMATH_CALUDE_x_range_for_sqrt_equality_l2244_224410

theorem x_range_for_sqrt_equality (x : ℝ) : 
  (Real.sqrt (x / (1 - x)) = Real.sqrt x / Real.sqrt (1 - x)) → 
  (0 ≤ x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_sqrt_equality_l2244_224410


namespace NUMINAMATH_CALUDE_sandy_grew_six_carrots_l2244_224409

/-- The number of carrots grown by Sandy -/
def sandy_carrots : ℕ := sorry

/-- The number of carrots grown by Sam -/
def sam_carrots : ℕ := 3

/-- The total number of carrots grown by Sandy and Sam -/
def total_carrots : ℕ := 9

/-- Theorem stating that Sandy grew 6 carrots -/
theorem sandy_grew_six_carrots : sandy_carrots = 6 := by sorry

end NUMINAMATH_CALUDE_sandy_grew_six_carrots_l2244_224409


namespace NUMINAMATH_CALUDE_box_removal_proof_l2244_224469

theorem box_removal_proof (total_boxes : Nat) (boxes_10lb boxes_20lb boxes_30lb boxes_40lb : Nat)
  (initial_avg_weight : Nat) (target_avg_weight : Nat) 
  (h1 : total_boxes = 30)
  (h2 : boxes_10lb = 10)
  (h3 : boxes_20lb = 10)
  (h4 : boxes_30lb = 5)
  (h5 : boxes_40lb = 5)
  (h6 : initial_avg_weight = 20)
  (h7 : target_avg_weight = 17) :
  let total_weight := boxes_10lb * 10 + boxes_20lb * 20 + boxes_30lb * 30 + boxes_40lb * 40
  let remaining_boxes := total_boxes - 6
  let remaining_weight := total_weight - (5 * 20 + 1 * 40)
  remaining_weight / remaining_boxes = target_avg_weight :=
by sorry

end NUMINAMATH_CALUDE_box_removal_proof_l2244_224469


namespace NUMINAMATH_CALUDE_middle_group_frequency_l2244_224441

/-- Represents a frequency distribution histogram -/
structure FrequencyHistogram where
  num_rectangles : ℕ
  sample_size : ℕ
  middle_area : ℝ
  other_areas : ℝ

/-- Theorem: The frequency of the middle group in a specific histogram -/
theorem middle_group_frequency (h : FrequencyHistogram) 
  (h_num_rectangles : h.num_rectangles = 11)
  (h_area_equality : h.middle_area = h.other_areas)
  (h_sample_size : h.sample_size = 160) :
  (h.middle_area / (h.middle_area + h.other_areas)) * h.sample_size = 80 := by
  sorry

#check middle_group_frequency

end NUMINAMATH_CALUDE_middle_group_frequency_l2244_224441


namespace NUMINAMATH_CALUDE_modulo_residue_sum_of_cubes_l2244_224468

theorem modulo_residue_sum_of_cubes (m : ℕ) (h : m = 17) :
  (512^3 + (6*104)^3 + (8*289)^3 + (5*68)^3) % m = 9 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_sum_of_cubes_l2244_224468


namespace NUMINAMATH_CALUDE_paper_strips_length_l2244_224467

/-- The total length of overlapping paper strips -/
def total_length (n : ℕ) (sheet_length : ℝ) (overlap : ℝ) : ℝ :=
  sheet_length + (n - 1) * (sheet_length - overlap)

/-- Theorem: The total length of 30 sheets of 25 cm paper strips overlapped by 6 cm is 576 cm -/
theorem paper_strips_length :
  total_length 30 25 6 = 576 := by
  sorry

end NUMINAMATH_CALUDE_paper_strips_length_l2244_224467


namespace NUMINAMATH_CALUDE_expression_simplification_l2244_224434

theorem expression_simplification (a b : ℝ) 
  (h : (a - 2)^2 + Real.sqrt (b + 1) = 0) :
  (a^2 - 2*a*b + b^2) / (a^2 - b^2) / ((a^2 - a*b) / a) - 2 / (a + b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2244_224434


namespace NUMINAMATH_CALUDE_triangle_center_distance_inequality_l2244_224445

/-- Given a triangle with circumradius R, inradius r, and distance d between
    its circumcenter and centroid, prove that d^2 ≤ R(R - 2r) -/
theorem triangle_center_distance_inequality 
  (R r d : ℝ) 
  (h_R : R > 0) 
  (h_r : r > 0) 
  (h_d : d ≥ 0) 
  (h_circumradius : R = circumradius_of_triangle) 
  (h_inradius : r = inradius_of_triangle) 
  (h_distance : d = distance_between_circumcenter_and_centroid) : 
  d^2 ≤ R * (R - 2*r) := by
sorry

end NUMINAMATH_CALUDE_triangle_center_distance_inequality_l2244_224445


namespace NUMINAMATH_CALUDE_janes_bagels_l2244_224497

theorem janes_bagels (b m : ℕ) : 
  b + m = 5 →
  (75 * b + 50 * m) % 100 = 0 →
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_janes_bagels_l2244_224497


namespace NUMINAMATH_CALUDE_renovation_profit_threshold_l2244_224424

/-- Annual profit without renovation (in millions of yuan) -/
def a (n : ℕ) : ℚ := 500 - 20 * n

/-- Annual profit with renovation (in millions of yuan) -/
def b (n : ℕ) : ℚ := 1000 - 1000 / (2^n)

/-- Cumulative profit without renovation (in millions of yuan) -/
def A (n : ℕ) : ℚ := 500 * n - 10 * n * (n + 1)

/-- Cumulative profit with renovation (in millions of yuan) -/
def B (n : ℕ) : ℚ := 1000 * n - 2600 + 2000 / (2^n)

/-- The minimum number of years for cumulative profit with renovation to exceed that without renovation -/
theorem renovation_profit_threshold : 
  ∀ n : ℕ, n ≥ 5 ↔ B n > A n :=
by sorry

end NUMINAMATH_CALUDE_renovation_profit_threshold_l2244_224424


namespace NUMINAMATH_CALUDE_prob_two_or_fewer_white_eq_23_28_l2244_224471

/-- The number of white balls in the bag -/
def white_balls : Nat := 5

/-- The number of red balls in the bag -/
def red_balls : Nat := 3

/-- The total number of balls in the bag -/
def total_balls : Nat := white_balls + red_balls

/-- The probability of drawing 2 or fewer white balls before a red ball -/
def prob_two_or_fewer_white : Rat :=
  (red_balls : Rat) / total_balls +
  (white_balls * red_balls : Rat) / (total_balls * (total_balls - 1)) +
  (white_balls * (white_balls - 1) * red_balls : Rat) / (total_balls * (total_balls - 1) * (total_balls - 2))

theorem prob_two_or_fewer_white_eq_23_28 : prob_two_or_fewer_white = 23 / 28 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_or_fewer_white_eq_23_28_l2244_224471


namespace NUMINAMATH_CALUDE_cans_per_carton_l2244_224406

theorem cans_per_carton (total_cartons : ℕ) (loaded_cartons : ℕ) (remaining_cans : ℕ) :
  total_cartons = 50 →
  loaded_cartons = 40 →
  remaining_cans = 200 →
  (total_cartons - loaded_cartons) * (remaining_cans / (total_cartons - loaded_cartons)) = remaining_cans :=
by sorry

end NUMINAMATH_CALUDE_cans_per_carton_l2244_224406


namespace NUMINAMATH_CALUDE_parabola_equation_l2244_224415

/-- A parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop
  vertex_origin : equation 0 0
  focus_x_axis : ∃ (f : ℝ), equation f 0 ∧ f ≠ 0

/-- The line y = 2x + 1 -/
def line (x y : ℝ) : Prop := y = 2 * x + 1

/-- The chord created by intersecting the parabola with the line -/
def chord (p : Parabola) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  p.equation x₁ y₁ ∧ p.equation x₂ y₂ ∧ line x₁ y₁ ∧ line x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

theorem parabola_equation (p : Parabola) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), chord p x₁ y₁ x₂ y₂ ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = 15) →
  p.equation = λ x y => y^2 = 12 * x :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2244_224415


namespace NUMINAMATH_CALUDE_orange_savings_l2244_224408

theorem orange_savings (liam_oranges : ℕ) (liam_price : ℚ) (claire_oranges : ℕ) (claire_price : ℚ) :
  liam_oranges = 40 →
  liam_price = 5/2 →
  claire_oranges = 30 →
  claire_price = 6/5 →
  (liam_oranges / 2 * liam_price + claire_oranges * claire_price : ℚ) = 86 := by
  sorry

end NUMINAMATH_CALUDE_orange_savings_l2244_224408


namespace NUMINAMATH_CALUDE_property_implies_linear_l2244_224477

/-- A function f: ℚ → ℚ satisfies the given property if
    f(x) + f(t) = f(y) + f(z) for all rational x < y < z < t
    that form an arithmetic progression -/
def SatisfiesProperty (f : ℚ → ℚ) : Prop :=
  ∀ (x y z t : ℚ) (d : ℚ), 0 < d → x < y ∧ y < z ∧ z < t →
  y = x + d ∧ z = y + d ∧ t = z + d →
  f x + f t = f y + f z

/-- A function f: ℚ → ℚ is linear if there exist rational m and b
    such that f(x) = mx + b for all rational x -/
def IsLinear (f : ℚ → ℚ) : Prop :=
  ∃ (m b : ℚ), ∀ (x : ℚ), f x = m * x + b

theorem property_implies_linear (f : ℚ → ℚ) :
  SatisfiesProperty f → IsLinear f := by
  sorry

end NUMINAMATH_CALUDE_property_implies_linear_l2244_224477


namespace NUMINAMATH_CALUDE_a3_greater_b3_l2244_224446

/-- Two sequences satisfying the given conditions -/
def sequences (a b : ℕ+ → ℝ) : Prop :=
  (∀ n, a n + b n = 700) ∧
  (∀ n, a (n + 1) = (7/10) * a n + (2/5) * b n) ∧
  (a 6 = 400)

/-- Theorem stating that a_3 > b_3 for sequences satisfying the given conditions -/
theorem a3_greater_b3 (a b : ℕ+ → ℝ) (h : sequences a b) : a 3 > b 3 := by
  sorry

end NUMINAMATH_CALUDE_a3_greater_b3_l2244_224446


namespace NUMINAMATH_CALUDE_expression_equality_l2244_224421

theorem expression_equality : 4 + (-8) / (-4) - (-1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2244_224421


namespace NUMINAMATH_CALUDE_monthly_payment_difference_l2244_224491

/-- The cost of the house in dollars -/
def house_cost : ℕ := 480000

/-- The cost of the trailer in dollars -/
def trailer_cost : ℕ := 120000

/-- The loan term in years -/
def loan_term : ℕ := 20

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- Calculates the monthly payment for a given cost over the loan term -/
def monthly_payment (cost : ℕ) : ℚ :=
  cost / (loan_term * months_per_year)

/-- The statement to be proved -/
theorem monthly_payment_difference :
  monthly_payment house_cost - monthly_payment trailer_cost = 1500 := by
  sorry

end NUMINAMATH_CALUDE_monthly_payment_difference_l2244_224491


namespace NUMINAMATH_CALUDE_jake_weight_proof_l2244_224443

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 196

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 290 - jake_weight

theorem jake_weight_proof :
  (jake_weight - 8 = 2 * sister_weight) ∧
  (jake_weight + sister_weight = 290) :=
by sorry

end NUMINAMATH_CALUDE_jake_weight_proof_l2244_224443


namespace NUMINAMATH_CALUDE_truth_telling_probability_l2244_224463

theorem truth_telling_probability (prob_A prob_B : ℝ) 
  (h_A : prob_A = 0.85) 
  (h_B : prob_B = 0.60) : 
  prob_A * prob_B = 0.51 := by
  sorry

end NUMINAMATH_CALUDE_truth_telling_probability_l2244_224463


namespace NUMINAMATH_CALUDE_parabola_parameter_range_l2244_224478

theorem parabola_parameter_range (a m n : ℝ) : 
  a ≠ 0 → 
  n = a * m^2 - 4 * a^2 * m - 3 →
  0 ≤ m → m ≤ 4 → 
  n ≤ -3 →
  (a ≥ 1 ∨ a < 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_parameter_range_l2244_224478


namespace NUMINAMATH_CALUDE_journey_fraction_by_foot_l2244_224492

/-- Given a journey with specific conditions, proves the fraction traveled by foot -/
theorem journey_fraction_by_foot :
  let total_distance : ℝ := 30.000000000000007
  let bus_fraction : ℝ := 3/5
  let car_distance : ℝ := 2
  let foot_distance : ℝ := total_distance - bus_fraction * total_distance - car_distance
  foot_distance / total_distance = 1/3 := by
sorry

end NUMINAMATH_CALUDE_journey_fraction_by_foot_l2244_224492


namespace NUMINAMATH_CALUDE_mass_percentage_cl_l2244_224464

/-- Given a compound where the mass percentage of Cl is 92.11%,
    prove that the mass percentage of Cl in the compound is 92.11%. -/
theorem mass_percentage_cl (compound_mass_percentage : ℝ) 
  (h : compound_mass_percentage = 92.11) : 
  compound_mass_percentage = 92.11 := by
  sorry

end NUMINAMATH_CALUDE_mass_percentage_cl_l2244_224464


namespace NUMINAMATH_CALUDE_work_time_problem_l2244_224448

/-- The work time problem for Mr. Willson -/
theorem work_time_problem (total_time tuesday wednesday thursday friday : ℚ) :
  total_time = 4 ∧
  tuesday = 1/2 ∧
  wednesday = 2/3 ∧
  thursday = 5/6 ∧
  friday = 75/60 →
  total_time - (tuesday + wednesday + thursday + friday) = 3/4 := by
sorry

end NUMINAMATH_CALUDE_work_time_problem_l2244_224448


namespace NUMINAMATH_CALUDE_max_games_buyable_l2244_224459

def total_earnings : ℝ := 180
def blade_percentage : ℝ := 0.35
def game_cost : ℝ := 12.50
def tax_rate : ℝ := 0.05

def remaining_money : ℝ := total_earnings * (1 - blade_percentage)
def game_cost_with_tax : ℝ := game_cost * (1 + tax_rate)

theorem max_games_buyable : 
  ⌊remaining_money / game_cost_with_tax⌋ = 8 :=
sorry

end NUMINAMATH_CALUDE_max_games_buyable_l2244_224459


namespace NUMINAMATH_CALUDE_concentration_a_is_45_percent_l2244_224476

/-- The concentration of spirit in vessel a -/
def concentration_a : ℝ := 45

/-- The concentration of spirit in vessel b -/
def concentration_b : ℝ := 30

/-- The concentration of spirit in vessel c -/
def concentration_c : ℝ := 10

/-- The volume taken from vessel a -/
def volume_a : ℝ := 4

/-- The volume taken from vessel b -/
def volume_b : ℝ := 5

/-- The volume taken from vessel c -/
def volume_c : ℝ := 6

/-- The concentration of spirit in the resultant solution -/
def concentration_result : ℝ := 26

/-- Theorem stating that the concentration of spirit in vessel a is 45% -/
theorem concentration_a_is_45_percent :
  (volume_a * concentration_a / 100 + 
   volume_b * concentration_b / 100 + 
   volume_c * concentration_c / 100) / 
  (volume_a + volume_b + volume_c) * 100 = concentration_result :=
by sorry

end NUMINAMATH_CALUDE_concentration_a_is_45_percent_l2244_224476


namespace NUMINAMATH_CALUDE_baxter_peanut_purchase_l2244_224482

/-- Represents the purchase of peanuts with various discounts and taxes applied. -/
structure PeanutPurchase where
  basePrice : ℝ  -- Base price per pound in dollars
  minPurchase : ℕ  -- Minimum purchase in pounds
  bulkDiscountThreshold : ℕ  -- Threshold for bulk discount in pounds
  bulkDiscountRate : ℝ  -- Bulk discount rate
  earlyBirdDiscountRate : ℝ  -- Early bird discount rate
  taxRate : ℝ  -- Sales tax rate
  totalSpent : ℝ  -- Total amount spent including taxes and discounts

/-- Calculates the number of pounds purchased given the total spent and purchase conditions. -/
def calculatePoundsPurchased (purchase : PeanutPurchase) : ℕ :=
  sorry

/-- Theorem stating that given the specific purchase conditions, 
    Baxter bought 28 pounds over the minimum. -/
theorem baxter_peanut_purchase :
  let purchase : PeanutPurchase := {
    basePrice := 3,
    minPurchase := 15,
    bulkDiscountThreshold := 25,
    bulkDiscountRate := 0.1,
    earlyBirdDiscountRate := 0.05,
    taxRate := 0.08,
    totalSpent := 119.88
  }
  calculatePoundsPurchased purchase - purchase.minPurchase = 28 := by
  sorry

end NUMINAMATH_CALUDE_baxter_peanut_purchase_l2244_224482


namespace NUMINAMATH_CALUDE_not_square_of_integer_l2244_224419

theorem not_square_of_integer (n : ℕ+) : ¬ ∃ m : ℤ, m^2 = 2*(n.val^2 + 1) - n.val := by
  sorry

end NUMINAMATH_CALUDE_not_square_of_integer_l2244_224419


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2244_224460

theorem simplify_and_evaluate_expression :
  let x := Real.cos (30 * π / 180)
  (x - (2 * x - 1) / x) / (x / (x - 1)) = Real.sqrt 3 / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2244_224460


namespace NUMINAMATH_CALUDE_consumption_increase_l2244_224481

theorem consumption_increase (original_tax original_consumption : ℝ) 
  (h_tax_positive : original_tax > 0) 
  (h_consumption_positive : original_consumption > 0) : 
  ∃ (increase_percentage : ℝ),
    (original_tax * 0.8 * (original_consumption * (1 + increase_percentage / 100)) = 
     original_tax * original_consumption * 0.96) ∧
    increase_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_consumption_increase_l2244_224481


namespace NUMINAMATH_CALUDE_remainder_theorem_l2244_224458

theorem remainder_theorem (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2244_224458


namespace NUMINAMATH_CALUDE_experiment_comparison_l2244_224428

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  total : Nat
  red : Nat
  black : Nat

/-- Represents the result of an experiment -/
structure ExperimentResult where
  expectation : ℚ
  variance : ℚ

/-- Calculates the result of drawing with replacement -/
def drawWithReplacement (bag : BagContents) (draws : Nat) : ExperimentResult :=
  sorry

/-- Calculates the result of drawing without replacement -/
def drawWithoutReplacement (bag : BagContents) (draws : Nat) : ExperimentResult :=
  sorry

theorem experiment_comparison (bag : BagContents) (draws : Nat) :
  let withReplacement := drawWithReplacement bag draws
  let withoutReplacement := drawWithoutReplacement bag draws
  (bag.total = 5 ∧ bag.red = 2 ∧ bag.black = 3 ∧ draws = 2) →
  (withReplacement.expectation = withoutReplacement.expectation ∧
   withReplacement.variance > withoutReplacement.variance) :=
by sorry

end NUMINAMATH_CALUDE_experiment_comparison_l2244_224428


namespace NUMINAMATH_CALUDE_operations_result_in_30_l2244_224495

def lenya_multiply (x : ℚ) : ℚ := x * 7
def gleb_add (x : ℚ) : ℚ := x + 3
def sasha_divide (x : ℚ) : ℚ := x / 4
def andrey_subtract (x : ℚ) : ℚ := x - 5

theorem operations_result_in_30 :
  andrey_subtract (lenya_multiply (gleb_add (sasha_divide 8))) = 30 :=
by sorry

end NUMINAMATH_CALUDE_operations_result_in_30_l2244_224495


namespace NUMINAMATH_CALUDE_sqrt_six_minus_sqrt_two_squared_l2244_224402

theorem sqrt_six_minus_sqrt_two_squared (x : ℝ) : x = Real.sqrt 6 - Real.sqrt 2 → 2 * x^2 + 4 * Real.sqrt 2 * x = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_minus_sqrt_two_squared_l2244_224402


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_sine_l2244_224427

/-- For a hyperbola with eccentricity √10 and transverse axis along the y-axis,
    the sine of the slope angle of its asymptote is √10/10. -/
theorem hyperbola_asymptote_slope_sine (e : ℝ) (h : e = Real.sqrt 10) :
  ∃ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 ∧ Real.sin θ = Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_sine_l2244_224427


namespace NUMINAMATH_CALUDE_min_value_problem_l2244_224479

theorem min_value_problem (x y : ℝ) 
  (h1 : (x - 3)^3 + 2014 * (x - 3) = 1)
  (h2 : (2 * y - 3)^3 + 2014 * (2 * y - 3) = -1) :
  ∀ z : ℝ, z = x^2 + 4 * y^2 + 4 * x → z ≥ 28 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l2244_224479


namespace NUMINAMATH_CALUDE_complex_sum_exponential_form_l2244_224475

theorem complex_sum_exponential_form :
  10 * Complex.exp (2 * π * I / 11) + 10 * Complex.exp (15 * π * I / 22) =
  10 * Real.sqrt 2 * Complex.exp (19 * π * I / 44) := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_exponential_form_l2244_224475


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2244_224420

theorem unique_positive_solution :
  ∃! x : ℚ, x > 0 ∧ 3 * x^2 + 7 * x - 20 = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2244_224420


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_real_range_l2244_224437

theorem sqrt_x_minus_2_real_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) → x ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_real_range_l2244_224437


namespace NUMINAMATH_CALUDE_books_sold_l2244_224455

/-- Proves the number of books Adam sold given initial count, books bought, and final count -/
theorem books_sold (initial : ℕ) (bought : ℕ) (final : ℕ) : 
  initial = 33 → bought = 23 → final = 45 → initial - (initial - final + bought) = 11 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l2244_224455


namespace NUMINAMATH_CALUDE_number_greater_than_half_l2244_224489

theorem number_greater_than_half : ∃ x : ℝ, x = 1/2 + 0.3 ∧ x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_number_greater_than_half_l2244_224489


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2244_224484

theorem sum_of_coefficients (d : ℝ) (a b c : ℤ) (h : d ≠ 0) :
  (8 : ℝ) * d + 9 + 10 * d^2 + 4 * d + 3 = (a : ℝ) * d + b + (c : ℝ) * d^2 →
  a + b + c = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2244_224484


namespace NUMINAMATH_CALUDE_hannah_mugs_problem_l2244_224488

/-- The number of mugs Hannah has of a color other than blue, red, or yellow -/
def other_color_mugs (total : ℕ) (blue red yellow : ℕ) : ℕ :=
  total - (blue + red + yellow)

theorem hannah_mugs_problem :
  ∀ (total blue red yellow : ℕ),
  total = 40 →
  blue = 3 * red →
  yellow = 12 →
  red = yellow / 2 →
  other_color_mugs total blue red yellow = 4 := by
sorry

end NUMINAMATH_CALUDE_hannah_mugs_problem_l2244_224488


namespace NUMINAMATH_CALUDE_comparison_theorem_l2244_224470

theorem comparison_theorem (a b c : ℝ) 
  (ha : a = Real.log 1.01)
  (hb : b = 1 / 101)
  (hc : c = Real.sin 0.01) :
  a > b ∧ c > a := by sorry

end NUMINAMATH_CALUDE_comparison_theorem_l2244_224470


namespace NUMINAMATH_CALUDE_virus_spread_l2244_224401

theorem virus_spread (x : ℝ) : 
  x > 0 ∧ (1 + x)^2 = 81 → x = 8 ∧ (1 + x)^3 > 700 := by
  sorry

end NUMINAMATH_CALUDE_virus_spread_l2244_224401


namespace NUMINAMATH_CALUDE_sin_30_degrees_l2244_224405

theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l2244_224405


namespace NUMINAMATH_CALUDE_roots_relation_l2244_224430

-- Define the polynomials f and g
def f (x : ℝ) := x^3 + 2*x^2 + 3*x + 4
def g (x b c d : ℝ) := x^3 + b*x^2 + c*x + d

-- State the theorem
theorem roots_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  (∀ r : ℝ, f r = 0 → g (r^2) b c d = 0) →
  b = -2 ∧ c = 1 ∧ d = -12 :=
sorry

end NUMINAMATH_CALUDE_roots_relation_l2244_224430


namespace NUMINAMATH_CALUDE_remainder_sum_mod_seven_l2244_224442

theorem remainder_sum_mod_seven : (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_seven_l2244_224442


namespace NUMINAMATH_CALUDE_angel_letters_count_l2244_224416

theorem angel_letters_count :
  ∀ (large_envelopes small_letters letters_per_large : ℕ),
    large_envelopes = 30 →
    letters_per_large = 2 →
    small_letters = 20 →
    large_envelopes * letters_per_large + small_letters = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_angel_letters_count_l2244_224416


namespace NUMINAMATH_CALUDE_fuel_station_problem_l2244_224423

/-- Fuel station problem -/
theorem fuel_station_problem 
  (service_cost : ℝ) 
  (fuel_cost_per_liter : ℝ) 
  (total_cost : ℝ) 
  (minivan_tank : ℝ) 
  (truck_tank_ratio : ℝ) 
  (num_trucks : ℕ) 
  (h1 : service_cost = 2.30)
  (h2 : fuel_cost_per_liter = 0.70)
  (h3 : total_cost = 396)
  (h4 : minivan_tank = 65)
  (h5 : truck_tank_ratio = 2.20)
  (h6 : num_trucks = 2) :
  ∃ (num_minivans : ℕ), 
    (num_minivans : ℝ) * (service_cost + minivan_tank * fuel_cost_per_liter) + 
    (num_trucks : ℝ) * (service_cost + truck_tank_ratio * minivan_tank * fuel_cost_per_liter) = 
    total_cost ∧ num_minivans = 4 := by
  sorry

end NUMINAMATH_CALUDE_fuel_station_problem_l2244_224423


namespace NUMINAMATH_CALUDE_rectangle_area_l2244_224439

theorem rectangle_area (x : ℝ) (h : x > 0) : 
  ∃ (width length : ℝ), 
    width > 0 ∧ 
    length = 2 * width ∧ 
    x^2 = width^2 + length^2 ∧ 
    width * length = (2/5) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2244_224439


namespace NUMINAMATH_CALUDE_train_length_proof_l2244_224480

/-- Given two trains running in opposite directions with the same speed,
    prove that their length is 120 meters. -/
theorem train_length_proof (speed : ℝ) (crossing_time : ℝ) :
  speed = 36 → crossing_time = 12 → 
  ∃ (train_length : ℝ), train_length = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l2244_224480


namespace NUMINAMATH_CALUDE_product_divisible_by_nine_l2244_224486

theorem product_divisible_by_nine : ∃ k : ℤ, 12345 * 54321 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_nine_l2244_224486


namespace NUMINAMATH_CALUDE_min_sum_squares_l2244_224449

def S : Finset Int := {-11, -8, -6, -1, 1, 5, 7, 12}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (∀ a b c d e f g h : Int, 
    a ∈ S → b ∈ S → c ∈ S → d ∈ S → e ∈ S → f ∈ S → g ∈ S → h ∈ S →
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f → a ≠ g → a ≠ h →
    b ≠ c → b ≠ d → b ≠ e → b ≠ f → b ≠ g → b ≠ h →
    c ≠ d → c ≠ e → c ≠ f → c ≠ g → c ≠ h →
    d ≠ e → d ≠ f → d ≠ g → d ≠ h →
    e ≠ f → e ≠ g → e ≠ h →
    f ≠ g → f ≠ h →
    g ≠ h →
    (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 1) ∧
  (p + q + r + s)^2 + (t + u + v + w)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2244_224449


namespace NUMINAMATH_CALUDE_custom_mult_example_l2244_224487

-- Define the custom operation *
def custom_mult (a b : Int) : Int := a * b

-- Theorem statement
theorem custom_mult_example : custom_mult 2 (-3) = -6 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_example_l2244_224487


namespace NUMINAMATH_CALUDE_f_two_roots_l2244_224417

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 5

-- State the theorem
theorem f_two_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f x = a ∧ f y = a) ↔ (3 ≤ a ∧ a ≤ 7) :=
sorry

end NUMINAMATH_CALUDE_f_two_roots_l2244_224417


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l2244_224499

/-- Given a triangle ABC, prove that (sin A + sin B + sin C) / (sin A * sin B * sin C) ≥ 4,
    with equality if and only if the triangle is equilateral -/
theorem triangle_sine_inequality (A B C : ℝ) (h_triangle : A + B + C = π) (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) :
  (Real.sin A + Real.sin B + Real.sin C) / (Real.sin A * Real.sin B * Real.sin C) ≥ 4 ∧
  ((Real.sin A + Real.sin B + Real.sin C) / (Real.sin A * Real.sin B * Real.sin C) = 4 ↔ A = B ∧ B = C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l2244_224499


namespace NUMINAMATH_CALUDE_circle_tangent_to_directrix_l2244_224466

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the right focus of the hyperbola
def right_focus : ℝ × ℝ := (2, 0)

-- Define the right directrix of the hyperbola
def right_directrix (x : ℝ) : Prop := x = 1

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Theorem statement
theorem circle_tangent_to_directrix :
  ∀ x y : ℝ,
  hyperbola x y →
  circle_equation x y ↔
    (∃ (cx cy : ℝ), (cx, cy) = right_focus ∧
      ∀ (dx : ℝ), right_directrix dx →
        (x - cx)^2 + (y - cy)^2 = (x - dx)^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_directrix_l2244_224466


namespace NUMINAMATH_CALUDE_candy_cost_l2244_224431

def amount_given : ℚ := 1
def change_received : ℚ := 0.46

theorem candy_cost : amount_given - change_received = 0.54 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_l2244_224431


namespace NUMINAMATH_CALUDE_divisibility_property_l2244_224454

theorem divisibility_property (p : ℕ) (h_odd : Odd p) (h_gt_one : p > 1) :
  ∃ k : ℤ, (p - 1) ^ ((p - 1) / 2) - 1 = (p - 2) * k :=
sorry

end NUMINAMATH_CALUDE_divisibility_property_l2244_224454


namespace NUMINAMATH_CALUDE_smallest_factorizable_b_l2244_224494

/-- A polynomial of degree 2 with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Represents a factorization of a quadratic polynomial into two linear factors -/
structure Factorization where
  p : ℤ
  q : ℤ

/-- Checks if a factorization is valid for a given quadratic polynomial -/
def isValidFactorization (poly : QuadraticPolynomial) (fac : Factorization) : Prop :=
  poly.a = 1 ∧ poly.b = fac.p + fac.q ∧ poly.c = fac.p * fac.q

/-- Theorem stating that 259 is the smallest positive integer b for which
    x^2 + bx + 2008 can be factored into a product of two polynomials
    with integer coefficients -/
theorem smallest_factorizable_b :
  ∀ b : ℤ, b > 0 →
  (∃ fac : Factorization, isValidFactorization ⟨1, b, 2008⟩ fac) →
  b ≥ 259 :=
sorry

end NUMINAMATH_CALUDE_smallest_factorizable_b_l2244_224494


namespace NUMINAMATH_CALUDE_initial_investment_interest_rate_l2244_224462

/-- Proves that the interest rate of the initial investment is 5% given the problem conditions --/
theorem initial_investment_interest_rate
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 2000)
  (h2 : additional_investment = 1000)
  (h3 : additional_rate = 0.08)
  (h4 : total_rate = 0.06)
  (h5 : ∃ r : ℝ, r * initial_investment + additional_rate * additional_investment = 
        total_rate * (initial_investment + additional_investment)) :
  ∃ r : ℝ, r = 0.05 :=
sorry

end NUMINAMATH_CALUDE_initial_investment_interest_rate_l2244_224462


namespace NUMINAMATH_CALUDE_product_of_sums_and_differences_l2244_224440

theorem product_of_sums_and_differences (P Q R S : ℝ) : 
  P = Real.sqrt 2011 + Real.sqrt 2010 →
  Q = -Real.sqrt 2011 - Real.sqrt 2010 →
  R = Real.sqrt 2011 - Real.sqrt 2010 →
  S = Real.sqrt 2010 - Real.sqrt 2011 →
  P * Q * R * S = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_and_differences_l2244_224440


namespace NUMINAMATH_CALUDE_sqrt_neg_x_squared_meaningful_l2244_224400

theorem sqrt_neg_x_squared_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = -x^2) ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_sqrt_neg_x_squared_meaningful_l2244_224400


namespace NUMINAMATH_CALUDE_boat_upstream_speed_l2244_224483

/-- Proves that the upstream speed is approximately 29.82 miles per hour given the conditions of the boat problem -/
theorem boat_upstream_speed (distance : ℝ) (downstream_time : ℝ) (time_difference : ℝ) :
  distance = 90 ∧ 
  downstream_time = 2.5191640969412834 ∧ 
  time_difference = 0.5 →
  ∃ upstream_speed : ℝ, 
    distance = upstream_speed * (downstream_time + time_difference) ∧
    abs (upstream_speed - 29.82) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_speed_l2244_224483


namespace NUMINAMATH_CALUDE_circle_op_difference_l2244_224461

/-- The custom operation ⊙ for three natural numbers -/
def circle_op (a b c : ℕ) : ℕ :=
  (a * b) * 100 + (b * c)

/-- Theorem stating the result of the calculation -/
theorem circle_op_difference : circle_op 5 7 4 - circle_op 7 4 5 = 708 := by
  sorry

end NUMINAMATH_CALUDE_circle_op_difference_l2244_224461


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l2244_224453

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l2244_224453

import Mathlib

namespace NUMINAMATH_CALUDE_second_day_sales_l3605_360526

def ice_cream_sales (x : ℕ) : List ℕ := [100, x, 109, 96, 103, 96, 105]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

theorem second_day_sales :
  ∃ (x : ℕ), mean (ice_cream_sales x) = 100.1 ∧ x = 92 := by sorry

end NUMINAMATH_CALUDE_second_day_sales_l3605_360526


namespace NUMINAMATH_CALUDE_total_cost_is_18_l3605_360594

-- Define the cost of a single soda
def soda_cost : ℝ := 1

-- Define the cost of a single soup
def soup_cost : ℝ := 3 * soda_cost

-- Define the cost of a sandwich
def sandwich_cost : ℝ := 3 * soup_cost

-- Define the total cost
def total_cost : ℝ := 3 * soda_cost + 2 * soup_cost + sandwich_cost

-- Theorem statement
theorem total_cost_is_18 : total_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_18_l3605_360594


namespace NUMINAMATH_CALUDE_equation_solution_l3605_360567

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 - Real.sqrt (16 + 8*x)) + Real.sqrt (5 - Real.sqrt (5 + x)) = 3 + Real.sqrt 5 :=
by
  use 4
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3605_360567


namespace NUMINAMATH_CALUDE_factorization_equality_l3605_360590

theorem factorization_equality (x : ℝ) : x * (x - 3) - x + 3 = (x - 1) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3605_360590


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_count_l3605_360540

theorem systematic_sampling_interval_count
  (total_papers : Nat)
  (selected_papers : Nat)
  (interval_start : Nat)
  (interval_end : Nat)
  (h1 : total_papers = 1000)
  (h2 : selected_papers = 50)
  (h3 : interval_start = 850)
  (h4 : interval_end = 949)
  (h5 : interval_start ≤ interval_end)
  (h6 : interval_end ≤ total_papers) :
  let sample_interval := total_papers / selected_papers
  let interval_size := interval_end - interval_start + 1
  interval_size / sample_interval = 5 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_count_l3605_360540


namespace NUMINAMATH_CALUDE_history_paper_pages_l3605_360596

/-- The number of days Stacy has to complete her history paper -/
def days_to_complete : ℕ := 3

/-- The number of pages Stacy must write per day to finish on time -/
def pages_per_day : ℕ := 11

/-- The total number of pages in Stacy's history paper -/
def total_pages : ℕ := days_to_complete * pages_per_day

theorem history_paper_pages : total_pages = 33 := by
  sorry

end NUMINAMATH_CALUDE_history_paper_pages_l3605_360596


namespace NUMINAMATH_CALUDE_sophomore_sample_size_l3605_360587

/-- Calculates the number of sophomores in a stratified sample -/
def sophomores_in_sample (total_students : ℕ) (total_sophomores : ℕ) (sample_size : ℕ) : ℕ :=
  (total_sophomores * sample_size) / total_students

theorem sophomore_sample_size :
  let total_students : ℕ := 4500
  let total_sophomores : ℕ := 1500
  let sample_size : ℕ := 600
  sophomores_in_sample total_students total_sophomores sample_size = 200 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_sample_size_l3605_360587


namespace NUMINAMATH_CALUDE_lollipop_count_l3605_360542

theorem lollipop_count (total_cost : ℝ) (single_cost : ℝ) (h1 : total_cost = 90) (h2 : single_cost = 0.75) :
  total_cost / single_cost = 120 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_count_l3605_360542


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3605_360585

/-- Given that a² and √b vary inversely, prove that b = 16 when a + b = 20 -/
theorem inverse_variation_problem (a b : ℝ) (k : ℝ) : 
  (∀ (a b : ℝ), a^2 * (b^(1/2)) = k) →  -- a² and √b vary inversely
  (4^2 * 16^(1/2) = k) →                -- a = 4 when b = 16
  (a + b = 20) →                        -- condition for the question
  (b = 16) :=                           -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3605_360585


namespace NUMINAMATH_CALUDE_printing_speed_proof_l3605_360577

/-- Mike's initial printing speed in pamphlets per hour -/
def initial_speed : ℕ := 600

/-- Total number of pamphlets printed -/
def total_pamphlets : ℕ := 9400

/-- Mike's initial printing time in hours -/
def mike_initial_time : ℕ := 9

/-- Mike's reduced speed printing time in hours -/
def mike_reduced_time : ℕ := 2

/-- Leo's printing time in hours -/
def leo_time : ℕ := 3

theorem printing_speed_proof :
  initial_speed * mike_initial_time + 
  (initial_speed / 3) * mike_reduced_time + 
  (2 * initial_speed) * leo_time = total_pamphlets :=
by sorry

end NUMINAMATH_CALUDE_printing_speed_proof_l3605_360577


namespace NUMINAMATH_CALUDE_dodge_to_hyundai_ratio_l3605_360552

/-- Given a car dealership with the following conditions:
  - Total number of vehicles is 400
  - Number of Kia vehicles is 100
  - Number of Hyundai vehicles is half the number of Dodge vehicles
Prove that the ratio of Dodge to Hyundai vehicles is 2:1 -/
theorem dodge_to_hyundai_ratio 
  (total : ℕ) 
  (kia : ℕ) 
  (dodge : ℕ) 
  (hyundai : ℕ) 
  (h1 : total = 400)
  (h2 : kia = 100)
  (h3 : hyundai = dodge / 2)
  (h4 : total = dodge + hyundai + kia) :
  dodge / hyundai = 2 := by
  sorry

end NUMINAMATH_CALUDE_dodge_to_hyundai_ratio_l3605_360552


namespace NUMINAMATH_CALUDE_max_piles_660_max_piles_optimal_l3605_360536

/-- The maximum number of piles that can be created from a given number of stones,
    where any two piles differ by strictly less than 2 times. -/
def maxPiles (totalStones : ℕ) : ℕ :=
  30  -- The actual implementation is not provided, just the result

theorem max_piles_660 :
  maxPiles 660 = 30 := by sorry

/-- A function to check if two pile sizes are similar (differ by strictly less than 2 times) -/
def areSimilarSizes (a b : ℕ) : Prop :=
  a < 2 * b ∧ b < 2 * a

/-- A function to represent a valid distribution of stones into piles -/
def isValidDistribution (piles : List ℕ) (totalStones : ℕ) : Prop :=
  piles.sum = totalStones ∧
  ∀ (a b : ℕ), a ∈ piles → b ∈ piles → areSimilarSizes a b

theorem max_piles_optimal (piles : List ℕ) :
  isValidDistribution piles 660 →
  piles.length ≤ 30 := by sorry

end NUMINAMATH_CALUDE_max_piles_660_max_piles_optimal_l3605_360536


namespace NUMINAMATH_CALUDE_min_value_S_range_of_c_l3605_360570

-- Define the constraint
def constraint (a b c : ℝ) : Prop := a + b + c = 1

-- Define the function S
def S (a b c : ℝ) : ℝ := 2 * a^2 + 3 * b^2 + c^2

-- Theorem 1: Minimum value of S
theorem min_value_S (a b c : ℝ) (h : constraint a b c) :
  S a b c ≥ 6/11 := by sorry

-- Theorem 2: Range of c
theorem range_of_c (a b c : ℝ) (h1 : constraint a b c) (h2 : S a b c = 1) :
  1/11 ≤ c ∧ c ≤ 1 := by sorry

end NUMINAMATH_CALUDE_min_value_S_range_of_c_l3605_360570


namespace NUMINAMATH_CALUDE_election_winning_probability_l3605_360514

/-- Represents the number of voters in the election -/
def total_voters : ℕ := 2019

/-- Represents the number of initial votes for the leading candidate -/
def initial_leading_votes : ℕ := 2

/-- Represents the number of initial votes for the trailing candidate -/
def initial_trailing_votes : ℕ := 1

/-- Represents the number of undecided voters -/
def undecided_voters : ℕ := total_voters - initial_leading_votes - initial_trailing_votes

/-- Calculates the probability of a candidate winning given their initial vote advantage -/
def winning_probability (initial_advantage : ℕ) : ℚ :=
  (1513 : ℚ) / 2017

/-- Theorem stating the probability of the leading candidate winning the election -/
theorem election_winning_probability :
  winning_probability (initial_leading_votes - initial_trailing_votes) = 1513 / 2017 := by
  sorry

end NUMINAMATH_CALUDE_election_winning_probability_l3605_360514


namespace NUMINAMATH_CALUDE_planar_graph_inequality_l3605_360551

/-- A planar graph is a graph that can be embedded in the plane without edge crossings. -/
structure PlanarGraph where
  /-- The number of edges in the planar graph -/
  E : ℕ
  /-- The number of faces in the planar graph -/
  F : ℕ

/-- For any planar graph, twice the number of edges is greater than or equal to
    three times the number of faces. -/
theorem planar_graph_inequality (G : PlanarGraph) : 2 * G.E ≥ 3 * G.F := by
  sorry

end NUMINAMATH_CALUDE_planar_graph_inequality_l3605_360551


namespace NUMINAMATH_CALUDE_strongest_correlation_l3605_360571

-- Define the type for a pair of observations
structure Observation where
  n : ℕ
  r : ℝ

-- Define the four given observations
def obs1 : Observation := ⟨10, 0.9533⟩
def obs2 : Observation := ⟨15, 0.3012⟩
def obs3 : Observation := ⟨17, 0.9991⟩
def obs4 : Observation := ⟨3, 0.9950⟩

-- Define a function to check if an observation indicates strong linear correlation
def isStrongCorrelation (obs : Observation) : Prop :=
  abs obs.r > 0.95

-- Theorem stating that obs1 and obs3 have the strongest linear correlation
theorem strongest_correlation :
  isStrongCorrelation obs1 ∧ isStrongCorrelation obs3 ∧
  ¬isStrongCorrelation obs2 ∧ ¬isStrongCorrelation obs4 :=
sorry

end NUMINAMATH_CALUDE_strongest_correlation_l3605_360571


namespace NUMINAMATH_CALUDE_range_of_a_l3605_360593

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1/2 then (1/2)^(x - 1/2) else Real.log x / Real.log a

theorem range_of_a (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) →
  a ≥ Real.sqrt 2 / 2 ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3605_360593


namespace NUMINAMATH_CALUDE_intercollegiate_competition_l3605_360548

theorem intercollegiate_competition (day1 day2 day3 day1_and_2 day2_and_3 only_day1 : ℕ)
  (h1 : day1 = 175)
  (h2 : day2 = 210)
  (h3 : day3 = 150)
  (h4 : day1_and_2 = 80)
  (h5 : day2_and_3 = 70)
  (h6 : only_day1 = 45)
  : ∃ all_days : ℕ,
    day1 = only_day1 + day1_and_2 + all_days ∧
    day2 = day1_and_2 + day2_and_3 + all_days ∧
    day3 = day2_and_3 + all_days ∧
    all_days = 50 := by
  sorry

end NUMINAMATH_CALUDE_intercollegiate_competition_l3605_360548


namespace NUMINAMATH_CALUDE_number_line_percentage_l3605_360520

theorem number_line_percentage : 
  let start : ℝ := -55
  let end_point : ℝ := 55
  let target : ℝ := 5.5
  let total_distance := end_point - start
  let target_distance := target - start
  (target_distance / total_distance) * 100 = 55 := by
sorry

end NUMINAMATH_CALUDE_number_line_percentage_l3605_360520


namespace NUMINAMATH_CALUDE_exam_average_l3605_360546

theorem exam_average (group1_count : ℕ) (group1_avg : ℚ) 
                      (group2_count : ℕ) (group2_avg : ℚ) : 
  group1_count = 15 →
  group1_avg = 70 / 100 →
  group2_count = 10 →
  group2_avg = 90 / 100 →
  (group1_count * group1_avg + group2_count * group2_avg) / (group1_count + group2_count) = 78 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l3605_360546


namespace NUMINAMATH_CALUDE_xy_sum_problem_l3605_360559

theorem xy_sum_problem (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x < y) (h4 : x + y + x * y = 119) :
  x + y ∈ ({20, 21, 24, 27} : Set ℕ) := by
sorry

end NUMINAMATH_CALUDE_xy_sum_problem_l3605_360559


namespace NUMINAMATH_CALUDE_total_matches_l3605_360513

def dozen : ℕ := 12

def boxes : ℕ := 5 * dozen

def matches_per_box : ℕ := 20

theorem total_matches : boxes * matches_per_box = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_matches_l3605_360513


namespace NUMINAMATH_CALUDE_a_less_than_two_thirds_l3605_360516

-- Define a decreasing function
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem a_less_than_two_thirds
  (f : ℝ → ℝ) (a : ℝ)
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f (1 - a) < f (2 * a - 1)) :
  a < 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_two_thirds_l3605_360516


namespace NUMINAMATH_CALUDE_tangent_line_sum_range_l3605_360563

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem tangent_line_sum_range (x₀ : ℝ) (h : x₀ > 0) :
  let k := 1 / x₀
  let b := Real.log x₀ - 1
  ∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, x > 0 ∧ k + b = y :=
sorry

end NUMINAMATH_CALUDE_tangent_line_sum_range_l3605_360563


namespace NUMINAMATH_CALUDE_dice_roll_probability_l3605_360550

def probability_first_die : ℚ := 3 / 8
def probability_second_die : ℚ := 3 / 8

theorem dice_roll_probability : 
  probability_first_die * probability_second_die = 9 / 64 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l3605_360550


namespace NUMINAMATH_CALUDE_abs_neg_2023_l3605_360598

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l3605_360598


namespace NUMINAMATH_CALUDE_orange_juice_consumption_l3605_360579

theorem orange_juice_consumption (initial_amount : ℚ) (alex_fraction : ℚ) (pat_fraction : ℚ) :
  initial_amount = 3/4 →
  alex_fraction = 1/2 →
  pat_fraction = 1/3 →
  pat_fraction * (initial_amount - alex_fraction * initial_amount) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_consumption_l3605_360579


namespace NUMINAMATH_CALUDE_selling_price_fraction_l3605_360568

theorem selling_price_fraction (cost_price : ℝ) (original_selling_price : ℝ) : 
  original_selling_price = cost_price * (1 + 0.275) →
  ∃ (f : ℝ), f * original_selling_price = cost_price * (1 - 0.15) ∧ f = 17 / 25 :=
by
  sorry

end NUMINAMATH_CALUDE_selling_price_fraction_l3605_360568


namespace NUMINAMATH_CALUDE_power_of_ten_problem_l3605_360537

theorem power_of_ten_problem (a b : ℝ) 
  (h1 : (40 : ℝ) ^ a = 5) 
  (h2 : (40 : ℝ) ^ b = 8) : 
  (10 : ℝ) ^ ((1 - a - b) / (2 * (1 - b))) = 1 := by
sorry

end NUMINAMATH_CALUDE_power_of_ten_problem_l3605_360537


namespace NUMINAMATH_CALUDE_color_natural_numbers_l3605_360530

theorem color_natural_numbers :
  ∃ (f : ℕ → Fin 2009),
    (∀ c : Fin 2009, Set.Infinite {n : ℕ | f n = c}) ∧
    (∀ x y z : ℕ, f x ≠ f y → f y ≠ f z → f x ≠ f z → x * y ≠ z) :=
by sorry

end NUMINAMATH_CALUDE_color_natural_numbers_l3605_360530


namespace NUMINAMATH_CALUDE_largest_number_l3605_360523

def A : ℕ := 27

def B (A : ℕ) : ℕ := A + 7

def C (B : ℕ) : ℕ := B - 9

def D (C : ℕ) : ℕ := 2 * C

theorem largest_number (A B C D : ℕ) (hA : A = 27) (hB : B = A + 7) (hC : C = B - 9) (hD : D = 2 * C) :
  D = max A (max B (max C D)) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3605_360523


namespace NUMINAMATH_CALUDE_jake_weight_proof_l3605_360501

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 196

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 290 - jake_weight

theorem jake_weight_proof :
  (jake_weight - 8 = 2 * sister_weight) ∧
  (jake_weight + sister_weight = 290) →
  jake_weight = 196 := by
  sorry

end NUMINAMATH_CALUDE_jake_weight_proof_l3605_360501


namespace NUMINAMATH_CALUDE_slope_range_for_given_inclination_l3605_360572

theorem slope_range_for_given_inclination (α : Real) (h : α ∈ Set.Icc (π / 4) (3 * π / 4)) :
  let k := Real.tan α
  k ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_slope_range_for_given_inclination_l3605_360572


namespace NUMINAMATH_CALUDE_subtract_twice_l3605_360507

theorem subtract_twice (a : ℝ) : a - 2*a = -a := by sorry

end NUMINAMATH_CALUDE_subtract_twice_l3605_360507


namespace NUMINAMATH_CALUDE_lcm_gcf_product_24_150_l3605_360574

theorem lcm_gcf_product_24_150 : Nat.lcm 24 150 * Nat.gcd 24 150 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_product_24_150_l3605_360574


namespace NUMINAMATH_CALUDE_solution_characterization_l3605_360573

/-- The set of ordered pairs (m, n) that satisfy the given condition -/
def SolutionSet : Set (ℕ × ℕ) :=
  {(2, 2), (2, 1), (3, 1), (1, 2), (1, 3), (5, 2), (5, 3), (2, 5), (3, 5)}

/-- Predicate to check if a pair (m, n) satisfies the condition -/
def SatisfiesCondition (p : ℕ × ℕ) : Prop :=
  let m := p.1
  let n := p.2
  m > 0 ∧ n > 0 ∧ ∃ k : ℤ, (n^3 + 1 : ℤ) = k * (m * n - 1)

theorem solution_characterization :
  ∀ p : ℕ × ℕ, p ∈ SolutionSet ↔ SatisfiesCondition p :=
sorry

end NUMINAMATH_CALUDE_solution_characterization_l3605_360573


namespace NUMINAMATH_CALUDE_womens_average_age_l3605_360515

/-- Represents the problem of finding the average age of two women -/
theorem womens_average_age 
  (n : ℕ) 
  (initial_total_age : ℝ) 
  (age_increase : ℝ) 
  (man1_age man2_age : ℝ) :
  n = 10 →
  age_increase = 6 →
  man1_age = 18 →
  man2_age = 22 →
  (initial_total_age / n + age_increase) * n = initial_total_age - man1_age - man2_age + 2 * ((initial_total_age / n + age_increase) * n - initial_total_age + man1_age + man2_age) / 2 →
  ((initial_total_age / n + age_increase) * n - initial_total_age + man1_age + man2_age) / 2 = 50 :=
by sorry

end NUMINAMATH_CALUDE_womens_average_age_l3605_360515


namespace NUMINAMATH_CALUDE_pages_of_maps_skipped_l3605_360575

theorem pages_of_maps_skipped (total_pages read_pages pages_left : ℕ) 
  (h1 : total_pages = 372)
  (h2 : read_pages = 125)
  (h3 : pages_left = 231) :
  total_pages - (read_pages + pages_left) = 16 := by
  sorry

end NUMINAMATH_CALUDE_pages_of_maps_skipped_l3605_360575


namespace NUMINAMATH_CALUDE_gross_revenue_increase_l3605_360543

theorem gross_revenue_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_decrease_percentage : ℝ)
  (quantity_increase_percentage : ℝ)
  (h1 : price_decrease_percentage = 20)
  (h2 : quantity_increase_percentage = 70)
  : let new_price := original_price * (1 - price_decrease_percentage / 100)
    let new_quantity := original_quantity * (1 + quantity_increase_percentage / 100)
    let original_revenue := original_price * original_quantity
    let new_revenue := new_price * new_quantity
    (new_revenue - original_revenue) / original_revenue * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gross_revenue_increase_l3605_360543


namespace NUMINAMATH_CALUDE_max_red_points_in_grid_l3605_360564

/-- 
Given a rectangular grid of m × n points where m and n are integers greater than 7,
this theorem states that the maximum number of points that can be colored red
such that no right-angled triangle with sides parallel to the rectangle's sides
has all three vertices colored red is m + n - 2.
-/
theorem max_red_points_in_grid (m n : ℕ) (hm : m > 7) (hn : n > 7) :
  (∃ (k : ℕ), k = m + n - 2 ∧
    ∀ (S : Finset (ℕ × ℕ)), S.card = k →
      (∀ (a b c : ℕ × ℕ), a ∈ S → b ∈ S → c ∈ S →
        (a.1 = b.1 ∧ a.2 = c.2 ∧ b.2 = c.1) → false) →
    (∀ (T : Finset (ℕ × ℕ)), T.card > k →
      ∃ (a b c : ℕ × ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧
        a.1 = b.1 ∧ a.2 = c.2 ∧ b.2 = c.1)) :=
by sorry

end NUMINAMATH_CALUDE_max_red_points_in_grid_l3605_360564


namespace NUMINAMATH_CALUDE_white_squares_in_20th_row_l3605_360581

/-- Represents the number of squares in the nth row of the modified stair-step figure -/
def squares_in_row (n : ℕ) : ℕ := 3 * n

/-- Represents the number of white squares in the nth row of the modified stair-step figure -/
def white_squares_in_row (n : ℕ) : ℕ := squares_in_row n / 2

theorem white_squares_in_20th_row :
  white_squares_in_row 20 = 30 := by
  sorry

#eval white_squares_in_row 20

end NUMINAMATH_CALUDE_white_squares_in_20th_row_l3605_360581


namespace NUMINAMATH_CALUDE_triangle_tangent_product_l3605_360549

theorem triangle_tangent_product (A B C : Real) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  A + B + C = π →
  (Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2 = (Real.sin B)^2 →
  (Real.tan A) * (Real.tan C) = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_product_l3605_360549


namespace NUMINAMATH_CALUDE_grandfather_gift_is_100_l3605_360534

/-- The amount of money Amy's grandfather gave her --/
def grandfather_gift : ℕ := sorry

/-- The number of dolls Amy bought --/
def dolls_bought : ℕ := 3

/-- The cost of each doll in dollars --/
def doll_cost : ℕ := 1

/-- The amount of money Amy has left after buying the dolls --/
def money_left : ℕ := 97

/-- Theorem stating that the grandfather's gift is $100 --/
theorem grandfather_gift_is_100 :
  grandfather_gift = dolls_bought * doll_cost + money_left :=
by sorry

end NUMINAMATH_CALUDE_grandfather_gift_is_100_l3605_360534


namespace NUMINAMATH_CALUDE_find_m_l3605_360547

noncomputable def f (x m c : ℝ) : ℝ :=
  if x < m then c / Real.sqrt x else c / Real.sqrt m

theorem find_m : ∃ m : ℝ, 
  (∃ c : ℝ, f 4 m c = 30 ∧ f m m c = 15) → m = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3605_360547


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_condition_l3605_360522

theorem not_sufficient_nor_necessary_condition (x y : ℝ) : 
  ¬(∀ x y : ℝ, x > y → x^2 > y^2) ∧ ¬(∀ x y : ℝ, x^2 > y^2 → x > y) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_condition_l3605_360522


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l3605_360599

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- If the points (2, 3), (7, k), and (15, 4) are collinear, then k = 44/13. -/
theorem collinear_points_k_value :
  collinear 2 3 7 k 15 4 → k = 44 / 13 :=
by sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l3605_360599


namespace NUMINAMATH_CALUDE_g_positive_f_local_min_iff_l3605_360554

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - (1/3) * a * x^3 - (1/2) * x^2

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a * x^2 - x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (f' a x) / x

-- Theorem 1: When a > 0, g(a) > 0
theorem g_positive (a : ℝ) (h : a > 0) : g a a > 0 := by sorry

-- Theorem 2: f(x) has a local minimum if and only if a ∈ (0, +∞)
theorem f_local_min_iff (a : ℝ) :
  (∃ x : ℝ, IsLocalMin (f a) x) ↔ a > 0 := by sorry

end NUMINAMATH_CALUDE_g_positive_f_local_min_iff_l3605_360554


namespace NUMINAMATH_CALUDE_A_divisible_by_1980_l3605_360583

def A : ℕ := sorry  -- Definition of A as the concatenated number

-- Theorem statement
theorem A_divisible_by_1980 : 1980 ∣ A :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_A_divisible_by_1980_l3605_360583


namespace NUMINAMATH_CALUDE_jake_peaches_l3605_360597

/-- 
Given that Steven has 13 peaches and Jake has six fewer peaches than Steven,
prove that Jake has 7 peaches.
-/
theorem jake_peaches (steven_peaches : ℕ) (jake_peaches : ℕ) 
  (h1 : steven_peaches = 13)
  (h2 : jake_peaches = steven_peaches - 6) :
  jake_peaches = 7 := by
  sorry

end NUMINAMATH_CALUDE_jake_peaches_l3605_360597


namespace NUMINAMATH_CALUDE_diamond_seven_three_l3605_360586

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ :=
  sorry

-- Axioms for the diamond operation
axiom diamond_zero (x : ℝ) : diamond x 0 = x
axiom diamond_comm (x y : ℝ) : diamond x y = diamond y x
axiom diamond_rec (x y : ℝ) : diamond (x + 2) y = diamond x y + y + 2

-- Theorem to prove
theorem diamond_seven_three : diamond 7 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_diamond_seven_three_l3605_360586


namespace NUMINAMATH_CALUDE_geometric_progression_equality_l3605_360533

theorem geometric_progression_equality (x y z : ℝ) :
  (∃ r : ℝ, y = x * r ∧ z = y * r) ↔ (x^2 + y^2) * (y^2 + z^2) = (x*y + y*z)^2 :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_equality_l3605_360533


namespace NUMINAMATH_CALUDE_uranus_appearance_time_l3605_360558

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def add_minutes (t : Time) (m : ℕ) : Time :=
  sorry

/-- Calculates the difference in minutes between two times -/
def minutes_difference (t1 t2 : Time) : ℕ :=
  sorry

theorem uranus_appearance_time 
  (mars_disappearance : Time)
  (jupiter_delay : ℕ)
  (uranus_delay : ℕ)
  (h_mars : mars_disappearance = ⟨0, 10, sorry, sorry⟩)  -- 12:10 AM
  (h_jupiter : jupiter_delay = 2 * 60 + 41)  -- 2 hours and 41 minutes
  (h_uranus : uranus_delay = 3 * 60 + 16)  -- 3 hours and 16 minutes
  : 
  let jupiter_appearance := add_minutes mars_disappearance jupiter_delay
  let uranus_appearance := add_minutes jupiter_appearance uranus_delay
  minutes_difference ⟨6, 0, sorry, sorry⟩ uranus_appearance = 7 :=
sorry

end NUMINAMATH_CALUDE_uranus_appearance_time_l3605_360558


namespace NUMINAMATH_CALUDE_triangle_problem_l3605_360511

open Real

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) → -- Angles are in (0, π)
  (a > 0) ∧ (b > 0) ∧ (c > 0) → -- Sides are positive
  (sin A / sin C = a / c) ∧ (sin B / sin C = b / c) → -- Law of sines
  (cos C + c / b * cos B = 2) → -- Given equation
  (C = π / 3) → -- Given angle C
  (c = 2 * Real.sqrt 3) → -- Given side c
  -- Conclusions to prove
  (sin A / sin B = 2) ∧ 
  (1 / 2 * a * b * sin C = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3605_360511


namespace NUMINAMATH_CALUDE_twenty_four_game_l3605_360528

theorem twenty_four_game (a b c d : ℤ) (e f g h : ℕ) : 
  (a = 3 ∧ b = 4 ∧ c = -6 ∧ d = 10) →
  (e = 3 ∧ f = 2 ∧ g = 6 ∧ h = 7) →
  ∃ (expr1 expr2 : ℤ → ℤ → ℤ → ℤ → ℤ),
    expr1 a b c d = 24 ∧
    expr2 e f g h = 24 :=
by sorry

end NUMINAMATH_CALUDE_twenty_four_game_l3605_360528


namespace NUMINAMATH_CALUDE_family_gathering_handshakes_l3605_360519

/-- The number of handshakes at a family gathering -/
def total_handshakes (twin_sets quadruplet_sets : ℕ) : ℕ :=
  let twins := twin_sets * 2
  let quadruplets := quadruplet_sets * 4
  let twin_handshakes := twins * (twins - 2) / 2
  let quadruplet_handshakes := quadruplets * (quadruplets - 4) / 2
  let cross_handshakes := twins * (quadruplets / 3) + quadruplets * (twins / 4)
  twin_handshakes + quadruplet_handshakes + cross_handshakes

/-- Theorem stating the number of handshakes at the family gathering -/
theorem family_gathering_handshakes :
  total_handshakes 12 8 = 1168 := by
  sorry

end NUMINAMATH_CALUDE_family_gathering_handshakes_l3605_360519


namespace NUMINAMATH_CALUDE_sin_135_degrees_l3605_360561

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l3605_360561


namespace NUMINAMATH_CALUDE_ellipse_condition_l3605_360562

def is_ellipse (m : ℝ) : Prop :=
  m + 3 > 0 ∧ m - 1 > 0

theorem ellipse_condition (m : ℝ) :
  (m > -3 → is_ellipse m) ∧ ¬(is_ellipse m → m > -3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3605_360562


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l3605_360504

/-- Given a rectangular plot with one side of 10 meters, where fence poles are placed 5 meters apart
    and 24 poles are needed in total, the length of the longer side is 40 meters. -/
theorem rectangle_longer_side (width : ℝ) (length : ℝ) (poles : ℕ) :
  width = 10 →
  poles = 24 →
  (2 * width + 2 * length) / 5 = poles →
  length = 40 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l3605_360504


namespace NUMINAMATH_CALUDE_men_entered_room_l3605_360512

theorem men_entered_room (initial_men : ℕ) (initial_women : ℕ) (men_entered : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  2 * (initial_women - 3) = 24 →
  initial_men + men_entered = 14 →
  men_entered = 2 := by
  sorry

end NUMINAMATH_CALUDE_men_entered_room_l3605_360512


namespace NUMINAMATH_CALUDE_packing_theorem_l3605_360578

/-- Represents the types of boxes that can be packed. -/
inductive BoxType
  | Large
  | Medium
  | Small

/-- Represents the types of packing tape. -/
inductive TapeType
  | A
  | B

/-- Calculates the amount of tape needed for a given box type. -/
def tapeNeeded (b : BoxType) : ℕ :=
  match b with
  | BoxType.Large => 5
  | BoxType.Medium => 3
  | BoxType.Small => 2

/-- Calculates the total tape used for packing a list of boxes. -/
def totalTapeUsed (boxes : List (BoxType × ℕ)) : ℕ :=
  boxes.foldl (fun acc (b, n) => acc + n * tapeNeeded b) 0

/-- Represents the packing scenario for Debbie and Mike. -/
structure PackingScenario where
  debbieBoxes : List (BoxType × ℕ)
  mikeBoxes : List (BoxType × ℕ)
  tapeARollLength : ℕ
  tapeBRollLength : ℕ

/-- Calculates the remaining tape for Debbie and Mike. -/
def remainingTape (scenario : PackingScenario) : TapeType → ℕ
  | TapeType.A => scenario.tapeARollLength - totalTapeUsed scenario.debbieBoxes
  | TapeType.B => 
      let usedTapeB := totalTapeUsed scenario.mikeBoxes
      scenario.tapeBRollLength - (usedTapeB % scenario.tapeBRollLength)

/-- The main theorem stating the remaining tape for Debbie and Mike. -/
theorem packing_theorem (scenario : PackingScenario) 
    (h1 : scenario.debbieBoxes = [(BoxType.Large, 2), (BoxType.Medium, 8), (BoxType.Small, 5)])
    (h2 : scenario.mikeBoxes = [(BoxType.Large, 3), (BoxType.Medium, 6), (BoxType.Small, 10)])
    (h3 : scenario.tapeARollLength = 50)
    (h4 : scenario.tapeBRollLength = 40) :
    remainingTape scenario TapeType.A = 6 ∧ remainingTape scenario TapeType.B = 27 := by
  sorry

end NUMINAMATH_CALUDE_packing_theorem_l3605_360578


namespace NUMINAMATH_CALUDE_min_trucks_required_l3605_360500

/-- Represents the total weight of boxes in tons -/
def total_weight : ℝ := 10

/-- Represents the maximum weight of a single box in tons -/
def max_box_weight : ℝ := 1

/-- Represents the capacity of each truck in tons -/
def truck_capacity : ℝ := 3

/-- Calculates the minimum number of trucks required -/
def min_trucks : ℕ := 5

theorem min_trucks_required :
  ∀ (weights : List ℝ),
    weights.sum = total_weight →
    (∀ w ∈ weights, w ≤ max_box_weight) →
    (∀ n : ℕ, n < min_trucks → 
      ∃ partition : List (List ℝ),
        partition.length = n ∧
        partition.join.sum = total_weight ∧
        (∀ part ∈ partition, part.sum > truck_capacity)) →
    ∃ partition : List (List ℝ),
      partition.length = min_trucks ∧
      partition.join.sum = total_weight ∧
      (∀ part ∈ partition, part.sum ≤ truck_capacity) :=
by sorry

#check min_trucks_required

end NUMINAMATH_CALUDE_min_trucks_required_l3605_360500


namespace NUMINAMATH_CALUDE_two_books_from_shelves_l3605_360591

/-- The number of ways to choose two books of different subjects -/
def choose_two_books (chinese : ℕ) (math : ℕ) (english : ℕ) : ℕ :=
  chinese * math + chinese * english + math * english

/-- Theorem stating that choosing two books of different subjects from the given shelves results in 242 ways -/
theorem two_books_from_shelves :
  choose_two_books 10 9 8 = 242 := by
  sorry

end NUMINAMATH_CALUDE_two_books_from_shelves_l3605_360591


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3605_360503

theorem complex_equation_solution (z : ℂ) 
  (h1 : Complex.abs (1 - z) + z = 10 - 3*I) :
  ∃ (m n : ℝ), 
    z = 5 - 3*I ∧ 
    z^2 + m*z + n = 1 - 3*I ∧ 
    m = -9 ∧ 
    n = 30 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3605_360503


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3605_360517

theorem polynomial_expansion (t : ℝ) : 
  (2 * t^3 - 3 * t + 2) * (-3 * t^2 + 3 * t - 5) = 
  -6 * t^5 + 6 * t^4 - t^3 + 3 * t^2 + 21 * t - 10 := by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3605_360517


namespace NUMINAMATH_CALUDE_total_tips_proof_l3605_360555

/-- Calculates the total tips earned over 3 days for a food truck --/
def total_tips (tips_per_customer : ℚ) (friday_customers : ℕ) (sunday_customers : ℕ) : ℚ :=
  let saturday_customers := 3 * friday_customers
  tips_per_customer * (friday_customers + saturday_customers + sunday_customers)

/-- Proves that the total tips earned over 3 days is $296.00 --/
theorem total_tips_proof :
  total_tips 2 28 36 = 296 := by
  sorry

end NUMINAMATH_CALUDE_total_tips_proof_l3605_360555


namespace NUMINAMATH_CALUDE_perfect_square_equation_l3605_360502

theorem perfect_square_equation (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_equation_l3605_360502


namespace NUMINAMATH_CALUDE_no_prime_solution_l3605_360553

/-- Converts a number from base p to decimal --/
def to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * p^i) 0

/-- The equation that p must satisfy --/
def equation (p : Nat) : Prop :=
  to_decimal [9, 0, 0, 1] p + to_decimal [7, 0, 3] p + 
  to_decimal [5, 1, 1] p + to_decimal [6, 2, 1] p + 
  to_decimal [7] p = 
  to_decimal [3, 4, 1] p + to_decimal [4, 7, 2] p + 
  to_decimal [1, 6, 3] p

theorem no_prime_solution : ¬∃ p : Nat, Nat.Prime p ∧ equation p := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l3605_360553


namespace NUMINAMATH_CALUDE_sarah_and_bob_walking_l3605_360545

/-- Sarah's walking rate in miles per minute -/
def sarah_rate : ℚ := 1 / 18

/-- Time Sarah walks in minutes -/
def sarah_time : ℚ := 15

/-- Distance Sarah walks in miles -/
def sarah_distance : ℚ := sarah_rate * sarah_time

/-- Bob's walking rate in miles per minute -/
def bob_rate : ℚ := 2 * sarah_rate

/-- Time Bob takes to walk Sarah's distance in minutes -/
def bob_time : ℚ := sarah_distance / bob_rate

theorem sarah_and_bob_walking :
  sarah_distance = 5 / 6 ∧ bob_time = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sarah_and_bob_walking_l3605_360545


namespace NUMINAMATH_CALUDE_tickets_difference_l3605_360521

theorem tickets_difference (initial_tickets : ℕ) (toys_tickets : ℕ) (clothes_tickets : ℕ)
  (h1 : initial_tickets = 13)
  (h2 : toys_tickets = 8)
  (h3 : clothes_tickets = 18) :
  clothes_tickets - toys_tickets = 10 := by
  sorry

end NUMINAMATH_CALUDE_tickets_difference_l3605_360521


namespace NUMINAMATH_CALUDE_regular_implies_all_equal_regular_implies_rotational_symmetry_rotational_symmetry_implies_regular_regular_implies_topologically_regular_l3605_360556

-- Define a structure for a polyhedron
structure Polyhedron where
  vertices : Set Point
  edges : Set (Point × Point)
  faces : Set (Set Point)

-- Define properties of a regular polyhedron
def is_regular (P : Polyhedron) : Prop := sorry

-- Define equality of geometric elements
def all_elements_equal (P : Polyhedron) : Prop := sorry

-- Define rotational symmetry property
def has_rotational_symmetry (P : Polyhedron) : Prop := sorry

-- Define topological regularity
def is_topologically_regular (P : Polyhedron) : Prop := sorry

-- Theorem 1
theorem regular_implies_all_equal (P : Polyhedron) :
  is_regular P → all_elements_equal P := by sorry

-- Theorem 2
theorem regular_implies_rotational_symmetry (P : Polyhedron) :
  is_regular P → has_rotational_symmetry P := by sorry

-- Theorem 3
theorem rotational_symmetry_implies_regular (P : Polyhedron) :
  has_rotational_symmetry P → is_regular P := by sorry

-- Theorem 4
theorem regular_implies_topologically_regular (P : Polyhedron) :
  is_regular P → is_topologically_regular P := by sorry

end NUMINAMATH_CALUDE_regular_implies_all_equal_regular_implies_rotational_symmetry_rotational_symmetry_implies_regular_regular_implies_topologically_regular_l3605_360556


namespace NUMINAMATH_CALUDE_remaining_pictures_l3605_360527

/-- The number of pictures Haley took at the zoo -/
def zoo_pictures : ℕ := 50

/-- The number of pictures Haley took at the museum -/
def museum_pictures : ℕ := 8

/-- The number of pictures Haley deleted -/
def deleted_pictures : ℕ := 38

/-- Theorem: The number of pictures Haley still has from her vacation is 20 -/
theorem remaining_pictures : 
  zoo_pictures + museum_pictures - deleted_pictures = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pictures_l3605_360527


namespace NUMINAMATH_CALUDE_min_tiles_needed_l3605_360588

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- The dimensions of the tile -/
def tileDimensions : Dimensions := ⟨2, 5⟩

/-- The dimensions of the floor in feet -/
def floorDimensionsFeet : Dimensions := ⟨3, 4⟩

/-- The dimensions of the floor in inches -/
def floorDimensionsInches : Dimensions :=
  ⟨feetToInches floorDimensionsFeet.length, feetToInches floorDimensionsFeet.width⟩

/-- Calculates the number of tiles needed to cover the floor -/
def tilesNeeded : ℕ :=
  (area floorDimensionsInches + area tileDimensions - 1) / area tileDimensions

theorem min_tiles_needed :
  tilesNeeded = 173 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_needed_l3605_360588


namespace NUMINAMATH_CALUDE_sqrt_four_ninths_l3605_360529

theorem sqrt_four_ninths : Real.sqrt (4 / 9) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_ninths_l3605_360529


namespace NUMINAMATH_CALUDE_intersection_when_a_zero_union_equals_A_iff_l3605_360584

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x < a + 5}

-- Theorem 1: When a = 0, A ∩ B = {x | -1 < x < 5}
theorem intersection_when_a_zero :
  A ∩ B 0 = {x : ℝ | -1 < x ∧ x < 5} := by sorry

-- Theorem 2: A ∪ B = A if and only if a ∈ (0, 1] ∪ [6, +∞)
theorem union_equals_A_iff (a : ℝ) :
  A ∪ B a = A ↔ a ∈ Set.Ioo 0 1 ∪ Set.Ici 6 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_zero_union_equals_A_iff_l3605_360584


namespace NUMINAMATH_CALUDE_existence_of_decreasing_lcm_sequence_l3605_360576

theorem existence_of_decreasing_lcm_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j, i < j → a i < a j) ∧ 
    (∀ i : Fin 99, Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_decreasing_lcm_sequence_l3605_360576


namespace NUMINAMATH_CALUDE_odd_prime_power_equality_l3605_360580

theorem odd_prime_power_equality (p m : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) :
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ (x^p + y^p : ℚ) / 2 = ((x + y : ℚ) / 2)^m) → m = p := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_power_equality_l3605_360580


namespace NUMINAMATH_CALUDE_fencing_length_l3605_360557

theorem fencing_length (area : ℝ) (uncovered_side : ℝ) : 
  area = 680 → uncovered_side = 8 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    uncovered_side + 2 * width = 178 := by
  sorry

end NUMINAMATH_CALUDE_fencing_length_l3605_360557


namespace NUMINAMATH_CALUDE_five_students_neither_subject_l3605_360569

/-- Represents a school club with students taking various subjects. -/
structure SchoolClub where
  total : Nat
  math : Nat
  physics : Nat
  mathAndPhysics : Nat
  onlyChemistry : Nat

/-- Calculates the number of students taking neither mathematics, physics, nor chemistry. -/
def studentsNeitherSubject (club : SchoolClub) : Nat :=
  club.total - (club.math + club.physics - club.mathAndPhysics + club.onlyChemistry)

/-- Theorem stating that in the given school club, 5 students take neither mathematics, physics, nor chemistry. -/
theorem five_students_neither_subject (club : SchoolClub) 
  (h1 : club.total = 80)
  (h2 : club.math = 50)
  (h3 : club.physics = 40)
  (h4 : club.mathAndPhysics = 25)
  (h5 : club.onlyChemistry = 10) :
  studentsNeitherSubject club = 5 := by
  sorry

#eval studentsNeitherSubject { total := 80, math := 50, physics := 40, mathAndPhysics := 25, onlyChemistry := 10 }

end NUMINAMATH_CALUDE_five_students_neither_subject_l3605_360569


namespace NUMINAMATH_CALUDE_course_selection_theorem_l3605_360532

/-- The number of ways for students to select courses --/
def selectCourses (numCourses numStudents coursesPerStudent : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of selection methods --/
theorem course_selection_theorem :
  selectCourses 4 3 2 = 114 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l3605_360532


namespace NUMINAMATH_CALUDE_parabola_equation_l3605_360582

/-- Given a parabola y^2 = 2px where p > 0, if a line with slope 1 passing through
    the focus intersects the parabola at points A and B such that |AB| = 8,
    then the equation of the parabola is y^2 = 4x -/
theorem parabola_equation (p : ℝ) (A B : ℝ × ℝ) (h1 : p > 0) :
  (∀ x y, y^2 = 2*p*x → (∃ t, y = t ∧ x = t + p/2)) →  -- Line passing through focus
  (A.2^2 = 2*p*A.1 ∧ B.2^2 = 2*p*B.1) →                -- A and B on parabola
  ‖A - B‖ = 8 →                                        -- |AB| = 8
  ∀ x y, y^2 = 2*p*x ↔ y^2 = 4*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3605_360582


namespace NUMINAMATH_CALUDE_f_monotonic_decreasing_l3605_360539

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + (a - 1) * x^2 + 48 * (a - 2) * x + b

theorem f_monotonic_decreasing (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  ∀ x ∈ Set.Icc (-4 : ℝ) 4, ∀ y ∈ Set.Icc (-4 : ℝ) 4, x ≤ y → f a b x ≥ f a b y :=
by sorry

end NUMINAMATH_CALUDE_f_monotonic_decreasing_l3605_360539


namespace NUMINAMATH_CALUDE_marble_selection_probability_l3605_360510

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 3

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 3

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles

/-- The number of marbles to be selected -/
def selected_marbles : ℕ := 4

/-- The probability of selecting exactly 1 red, 2 blue, and 1 green marble -/
def probability : ℚ := 3 / 14

theorem marble_selection_probability : 
  (Nat.choose red_marbles 1 * Nat.choose blue_marbles 2 * Nat.choose green_marbles 1 : ℚ) / 
  (Nat.choose total_marbles selected_marbles) = probability := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_probability_l3605_360510


namespace NUMINAMATH_CALUDE_probability_at_least_one_male_doctor_l3605_360531

/-- The probability of selecting at least one male doctor when choosing 3 doctors from 4 female and 3 male doctors. -/
theorem probability_at_least_one_male_doctor : 
  let total_doctors : ℕ := 7
  let female_doctors : ℕ := 4
  let male_doctors : ℕ := 3
  let doctors_to_select : ℕ := 3
  let total_combinations := Nat.choose total_doctors doctors_to_select
  let favorable_outcomes := 
    Nat.choose male_doctors 1 * Nat.choose female_doctors 2 +
    Nat.choose male_doctors 2 * Nat.choose female_doctors 1 +
    Nat.choose male_doctors 3
  (favorable_outcomes : ℚ) / total_combinations = 31 / 35 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_male_doctor_l3605_360531


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l3605_360524

/-- A geometric figure composed of five identical squares arranged in a 'T' shape -/
structure TShape where
  /-- The total area of the figure in square centimeters -/
  total_area : ℝ
  /-- The figure is composed of five identical squares -/
  num_squares : ℕ
  /-- Assumption that the total area is 125 cm² -/
  area_assumption : total_area = 125
  /-- Assumption that the number of squares is 5 -/
  squares_assumption : num_squares = 5

/-- The perimeter of the 'T' shaped figure -/
def perimeter (t : TShape) : ℝ :=
  sorry

/-- Theorem stating that the perimeter of the 'T' shaped figure is 35 cm -/
theorem t_shape_perimeter (t : TShape) : perimeter t = 35 :=
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l3605_360524


namespace NUMINAMATH_CALUDE_min_value_xyz_plus_2sum_l3605_360505

theorem min_value_xyz_plus_2sum (x y z : ℝ) 
  (hx : |x| ≥ 2) (hy : |y| ≥ 2) (hz : |z| ≥ 2) : 
  |x * y * z + 2 * (x + y + z)| ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xyz_plus_2sum_l3605_360505


namespace NUMINAMATH_CALUDE_find_A_value_l3605_360525

theorem find_A_value : ∃ (A B : ℕ), 
  A < 10 ∧ B < 10 ∧ 
  10 * A + 8 + 30 + B = 99 ∧
  A = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_A_value_l3605_360525


namespace NUMINAMATH_CALUDE_minimum_teacher_time_l3605_360595

def student_time (explanation_time : ℕ) (completion_time : ℕ) : ℕ :=
  explanation_time + completion_time

theorem minimum_teacher_time 
  (student_A : ℕ) 
  (student_B : ℕ) 
  (student_C : ℕ) 
  (explanation_time : ℕ) 
  (h1 : student_A = student_time explanation_time 13)
  (h2 : student_B = student_time explanation_time 10)
  (h3 : student_C = student_time explanation_time 16)
  (h4 : explanation_time = 3) :
  3 * explanation_time + 2 * student_B + student_A + student_C = 90 :=
sorry

end NUMINAMATH_CALUDE_minimum_teacher_time_l3605_360595


namespace NUMINAMATH_CALUDE_root_product_theorem_l3605_360541

theorem root_product_theorem (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a^2 + 1/b)^2 - p*(a^2 + 1/b) + r = 0) →
  ((b^2 + 1/a)^2 - p*(b^2 + 1/a) + r = 0) →
  r = 46/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3605_360541


namespace NUMINAMATH_CALUDE_unique_positive_cyclic_shift_l3605_360509

def CyclicShift (a : List ℤ) : List (List ℤ) :=
  List.range a.length |>.map (λ i => a.rotate i)

def PositivePartialSums (a : List ℤ) : Prop :=
  List.scanl (· + ·) 0 a |>.tail |>.all (λ x => x > 0)

theorem unique_positive_cyclic_shift
  (a : List ℤ)
  (h_sum : a.sum = 1) :
  ∃! shift, shift ∈ CyclicShift a ∧ PositivePartialSums shift :=
sorry

end NUMINAMATH_CALUDE_unique_positive_cyclic_shift_l3605_360509


namespace NUMINAMATH_CALUDE_min_removals_correct_l3605_360535

/-- Represents a triangular grid constructed with toothpicks -/
structure ToothpickGrid where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ

/-- The number of horizontal toothpicks in the grid -/
def horizontal_toothpicks (grid : ToothpickGrid) : ℕ :=
  grid.upward_triangles

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_removals (grid : ToothpickGrid) : ℕ :=
  horizontal_toothpicks grid

theorem min_removals_correct (grid : ToothpickGrid) 
  (h1 : grid.total_toothpicks = 50)
  (h2 : grid.upward_triangles = 15)
  (h3 : grid.downward_triangles = 10) :
  min_removals grid = 15 := by
  sorry

#eval min_removals { total_toothpicks := 50, upward_triangles := 15, downward_triangles := 10 }

end NUMINAMATH_CALUDE_min_removals_correct_l3605_360535


namespace NUMINAMATH_CALUDE_paper_used_calculation_l3605_360518

-- Define the variables
def total_paper : ℕ := 900
def remaining_paper : ℕ := 744

-- Define the theorem
theorem paper_used_calculation : total_paper - remaining_paper = 156 := by
  sorry

end NUMINAMATH_CALUDE_paper_used_calculation_l3605_360518


namespace NUMINAMATH_CALUDE_polygon_distance_inequality_l3605_360589

/-- A polygon in a plane -/
structure Polygon where
  vertices : List (Real × Real)
  is_closed : vertices.length > 2

/-- Calculate the perimeter of a polygon -/
def perimeter (F : Polygon) : Real :=
  sorry

/-- Calculate the sum of distances from a point to the vertices of a polygon -/
def sum_distances_to_vertices (X : Real × Real) (F : Polygon) : Real :=
  sorry

/-- Calculate the sum of distances from a point to the sidelines of a polygon -/
def sum_distances_to_sidelines (X : Real × Real) (F : Polygon) : Real :=
  sorry

/-- The main theorem -/
theorem polygon_distance_inequality (X : Real × Real) (F : Polygon) :
  let p := perimeter F
  let d := sum_distances_to_vertices X F
  let h := sum_distances_to_sidelines X F
  d^2 - h^2 ≥ p^2 / 4 := by
    sorry

end NUMINAMATH_CALUDE_polygon_distance_inequality_l3605_360589


namespace NUMINAMATH_CALUDE_work_completion_time_l3605_360508

/-- Given:
  * A can finish a work in x days
  * B can finish the same work in x/2 days
  * A and B together can finish half the work in 1 day
Prove that x = 6 -/
theorem work_completion_time (x : ℝ) 
  (hx : x > 0) 
  (hA : (1 : ℝ) / x = 1 / x) 
  (hB : (1 : ℝ) / (x/2) = 2 / x) 
  (hAB : (1 : ℝ) / x + 2 / x = 1 / 2) : 
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3605_360508


namespace NUMINAMATH_CALUDE_geometric_sequence_shift_l3605_360592

theorem geometric_sequence_shift (a : ℕ → ℝ) (q c : ℝ) : 
  (∀ n, a (n + 1) = q * a n) →  -- {a_n} is geometric with ratio q
  q ≠ 1 →                       -- q is not equal to 1
  (∃ r, ∀ n, (a (n + 1) + c) = r * (a n + c)) →  -- {a_n + c} is geometric
  c = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_shift_l3605_360592


namespace NUMINAMATH_CALUDE_probability_sum_less_than_five_l3605_360565

def dice_outcomes : ℕ := 6 * 6

def favorable_outcomes : ℕ := 6

theorem probability_sum_less_than_five (p : ℚ) : 
  p = favorable_outcomes / dice_outcomes → p = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_less_than_five_l3605_360565


namespace NUMINAMATH_CALUDE_probability_shaded_is_one_fourth_l3605_360538

/-- Represents a shape in the configuration -/
inductive Shape
| Square
| Triangle

/-- The configuration of shapes -/
def Configuration := List Shape

/-- A configuration is valid if it contains exactly 4 shapes -/
def ValidConfiguration (config : Configuration) : Prop :=
  config.length = 4

/-- The number of shaded shapes in the configuration -/
def NumShaded (config : Configuration) : Nat := 1

/-- The probability of selecting a shaded shape -/
def ProbabilityShaded (config : Configuration) : ℚ :=
  (NumShaded config : ℚ) / (config.length : ℚ)

theorem probability_shaded_is_one_fourth
  (config : Configuration)
  (h_valid : ValidConfiguration config) :
  ProbabilityShaded config = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_shaded_is_one_fourth_l3605_360538


namespace NUMINAMATH_CALUDE_min_distance_vectors_l3605_360566

def a (t : ℝ) : Fin 3 → ℝ := ![2, t, t]
def b (t : ℝ) : Fin 3 → ℝ := ![1 - t, 2 * t - 1, 0]

theorem min_distance_vectors (t : ℝ) :
  ∃ (min : ℝ), min = Real.sqrt 2 ∧ ∀ (s : ℝ), ‖b s - a s‖ ≥ min := by sorry

end NUMINAMATH_CALUDE_min_distance_vectors_l3605_360566


namespace NUMINAMATH_CALUDE_relationship_abc_l3605_360544

theorem relationship_abc (a b c : Real) 
  (ha : a = 3^(0.3 : Real))
  (hb : b = Real.log 3 / Real.log π)
  (hc : c = Real.log 2 / Real.log 0.3) :
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l3605_360544


namespace NUMINAMATH_CALUDE_house_painting_cans_l3605_360560

/-- Calculates the number of paint cans needed for a house painting job -/
def paint_cans_needed (num_bedrooms : ℕ) (paint_per_room : ℕ) 
  (color_can_size : ℕ) (white_can_size : ℕ) : ℕ :=
  let num_other_rooms := 2 * num_bedrooms
  let total_color_paint := num_bedrooms * paint_per_room
  let total_white_paint := num_other_rooms * paint_per_room
  let color_cans := (total_color_paint + color_can_size - 1) / color_can_size
  let white_cans := (total_white_paint + white_can_size - 1) / white_can_size
  color_cans + white_cans

/-- Theorem stating that the number of paint cans needed for the given conditions is 10 -/
theorem house_painting_cans : paint_cans_needed 3 2 1 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_house_painting_cans_l3605_360560


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l3605_360506

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_diff : |a 2 - a 3| = 14)
  (h_product : a 1 * a 2 * a 3 = 343)
  (h_geometric : geometric_sequence a) :
  ∃ q : ℝ, q = 3 ∧ ∀ n : ℕ, a n = 7 * q^(n - 2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l3605_360506

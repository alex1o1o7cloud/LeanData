import Mathlib

namespace park_rose_bushes_l1035_103526

/-- The number of rose bushes in a park after planting new ones -/
def total_rose_bushes (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: The park will have 6 rose bushes after planting -/
theorem park_rose_bushes : total_rose_bushes 2 4 = 6 := by
  sorry

end park_rose_bushes_l1035_103526


namespace notebook_packages_l1035_103550

theorem notebook_packages (L : ℕ) : L > 4 →
  (∃ a b : ℕ, a > 0 ∧ a * L + 4 * b = 69) →
  L = 23 := by sorry

end notebook_packages_l1035_103550


namespace circle_equation_m_range_l1035_103559

theorem circle_equation_m_range (m : ℝ) :
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y + m = 0 ↔ (x - 1)^2 + (y - 2)^2 = r^2) →
  m < 5 :=
by sorry

end circle_equation_m_range_l1035_103559


namespace smallest_b_value_l1035_103510

theorem smallest_b_value (a b : ℝ) (h1 : 1 < a) (h2 : a < b)
  (h3 : b ≥ a + 1)
  (h4 : 1 / b + 1 / a ≤ 1) :
  b ≥ (3 + Real.sqrt 5) / 2 := by
  sorry

end smallest_b_value_l1035_103510


namespace price_per_drawing_l1035_103542

-- Define the variables
def saturday_sales : ℕ := 24
def sunday_sales : ℕ := 16
def total_revenue : ℕ := 800

-- Define the theorem
theorem price_per_drawing : 
  ∃ (price : ℚ), price * (saturday_sales + sunday_sales) = total_revenue ∧ price = 20 := by
  sorry

end price_per_drawing_l1035_103542


namespace sqrt_product_equality_l1035_103596

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l1035_103596


namespace train_length_calculation_l1035_103537

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (time_to_cross : ℝ) (bridge_length : ℝ) :
  train_speed = 36 * (1000 / 3600) →
  time_to_cross = 82.49340052795776 →
  bridge_length = 660 →
  train_speed * time_to_cross - bridge_length = 164.9340052795776 := by
  sorry

#check train_length_calculation

end train_length_calculation_l1035_103537


namespace atMostOneHead_exactlyTwoHeads_mutuallyExclusive_l1035_103530

/-- Represents the outcome of tossing a coin -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the result of tossing two coins simultaneously -/
def TwoCoinsResult := (CoinOutcome × CoinOutcome)

/-- The set of all possible outcomes when tossing two coins -/
def sampleSpace : Set TwoCoinsResult := {(CoinOutcome.Heads, CoinOutcome.Heads),
                                         (CoinOutcome.Heads, CoinOutcome.Tails),
                                         (CoinOutcome.Tails, CoinOutcome.Heads),
                                         (CoinOutcome.Tails, CoinOutcome.Tails)}

/-- The event of getting at most 1 head -/
def atMostOneHead : Set TwoCoinsResult := {(CoinOutcome.Heads, CoinOutcome.Tails),
                                           (CoinOutcome.Tails, CoinOutcome.Heads),
                                           (CoinOutcome.Tails, CoinOutcome.Tails)}

/-- The event of getting exactly 2 heads -/
def exactlyTwoHeads : Set TwoCoinsResult := {(CoinOutcome.Heads, CoinOutcome.Heads)}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set TwoCoinsResult) : Prop := A ∩ B = ∅

theorem atMostOneHead_exactlyTwoHeads_mutuallyExclusive :
  mutuallyExclusive atMostOneHead exactlyTwoHeads := by
  sorry

end atMostOneHead_exactlyTwoHeads_mutuallyExclusive_l1035_103530


namespace logarithm_equation_solution_l1035_103548

theorem logarithm_equation_solution :
  ∃ (A B C : ℕ+), 
    (Nat.gcd A.val (Nat.gcd B.val C.val) = 1) ∧
    (A.val : ℝ) * (Real.log 5 / Real.log 300) + (B.val : ℝ) * (Real.log (2 * A.val) / Real.log 300) = C.val ∧
    A.val + B.val + C.val = 4 := by
  sorry

#check logarithm_equation_solution

end logarithm_equation_solution_l1035_103548


namespace book_pages_theorem_l1035_103509

/-- Calculates the number of pages with text in a book with given specifications. -/
def pages_with_text (total_pages : ℕ) (image_pages : ℕ) (intro_pages : ℕ) : ℕ :=
  let remaining_pages := total_pages - image_pages - intro_pages
  remaining_pages / 2

/-- Theorem stating that a book with 98 pages, half images, 11 intro pages, 
    and remaining pages split equally between blank and text, has 19 pages of text. -/
theorem book_pages_theorem : 
  pages_with_text 98 (98 / 2) 11 = 19 := by
sorry

#eval pages_with_text 98 (98 / 2) 11

end book_pages_theorem_l1035_103509


namespace greatest_distance_between_circle_centers_l1035_103553

theorem greatest_distance_between_circle_centers 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ) 
  (circle_diameter : ℝ) 
  (h1 : rectangle_width = 16) 
  (h2 : rectangle_height = 20) 
  (h3 : circle_diameter = 8) :
  ∃ (d : ℝ), d = 4 * Real.sqrt 13 ∧ 
  ∀ (d' : ℝ), d' ≤ d ∧ 
  ∃ (x1 y1 x2 y2 : ℝ),
    0 ≤ x1 ∧ x1 ≤ rectangle_width ∧
    0 ≤ y1 ∧ y1 ≤ rectangle_height ∧
    0 ≤ x2 ∧ x2 ≤ rectangle_width ∧
    0 ≤ y2 ∧ y2 ≤ rectangle_height ∧
    (x1 - circle_diameter / 2 ≥ 0) ∧ (x1 + circle_diameter / 2 ≤ rectangle_width) ∧
    (y1 - circle_diameter / 2 ≥ 0) ∧ (y1 + circle_diameter / 2 ≤ rectangle_height) ∧
    (x2 - circle_diameter / 2 ≥ 0) ∧ (x2 + circle_diameter / 2 ≤ rectangle_width) ∧
    (y2 - circle_diameter / 2 ≥ 0) ∧ (y2 + circle_diameter / 2 ≤ rectangle_height) ∧
    d' = Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) := by
  sorry


end greatest_distance_between_circle_centers_l1035_103553


namespace largest_divisor_of_cube_difference_l1035_103532

theorem largest_divisor_of_cube_difference (n : ℤ) (h : 5 ∣ n) :
  (∃ (m : ℤ), m ∣ (n^3 - n) ∧ ∀ (k : ℤ), k ∣ (n^3 - n) → k ≤ m) → 
  (∃ (m : ℤ), m ∣ (n^3 - n) ∧ ∀ (k : ℤ), k ∣ (n^3 - n) → k ≤ m) ∧ m = 10 :=
sorry

end largest_divisor_of_cube_difference_l1035_103532


namespace tony_fish_count_l1035_103565

/-- The number of fish Tony's parents buy each year -/
def fish_bought_yearly : ℕ := 2

/-- The number of years that pass -/
def years : ℕ := 5

/-- The number of fish Tony starts with -/
def initial_fish : ℕ := 2

/-- The number of fish that die each year -/
def fish_lost_yearly : ℕ := 1

/-- The number of fish Tony has after 5 years -/
def final_fish : ℕ := 7

theorem tony_fish_count :
  initial_fish + years * (fish_bought_yearly - fish_lost_yearly) = final_fish :=
by sorry

end tony_fish_count_l1035_103565


namespace remaining_money_l1035_103505

def money_spent_on_books : ℝ := 76.8
def money_spent_on_apples : ℝ := 12
def total_money_brought : ℝ := 100

theorem remaining_money :
  total_money_brought - money_spent_on_books - money_spent_on_apples = 11.2 := by
  sorry

end remaining_money_l1035_103505


namespace sqrt_nine_factorial_over_126_l1035_103563

theorem sqrt_nine_factorial_over_126 : 
  Real.sqrt (Nat.factorial 9 / 126) = 8 * Real.sqrt 5 := by
  sorry

end sqrt_nine_factorial_over_126_l1035_103563


namespace gcd_9125_4277_l1035_103552

theorem gcd_9125_4277 : Nat.gcd 9125 4277 = 1 := by
  sorry

end gcd_9125_4277_l1035_103552


namespace intersection_M_N_l1035_103570

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0, 2} := by sorry

end intersection_M_N_l1035_103570


namespace no_real_solutions_for_sqrt_equation_l1035_103512

theorem no_real_solutions_for_sqrt_equation :
  ¬∃ x : ℝ, Real.sqrt (4 + 2*x) + Real.sqrt (6 + 3*x) + Real.sqrt (8 + 4*x) = 9 + 3*x/2 :=
by sorry

end no_real_solutions_for_sqrt_equation_l1035_103512


namespace race_time_proof_l1035_103567

/-- The time A takes to complete the race -/
def race_time_A : ℝ := 390

/-- The distance of the race in meters -/
def race_distance : ℝ := 1000

/-- The difference in distance between A and B at the finish -/
def distance_diff_AB : ℝ := 25

/-- The time difference between A and B -/
def time_diff_AB : ℝ := 10

/-- The difference in distance between A and C at the finish -/
def distance_diff_AC : ℝ := 40

/-- The time difference between A and C -/
def time_diff_AC : ℝ := 8

/-- The difference in distance between B and C at the finish -/
def distance_diff_BC : ℝ := 15

/-- The time difference between B and C -/
def time_diff_BC : ℝ := 2

theorem race_time_proof :
  let v_a := race_distance / race_time_A
  let v_b := (race_distance - distance_diff_AB) / race_time_A
  let v_c := (race_distance - distance_diff_AC) / race_time_A
  (v_b * (race_time_A + time_diff_AB) = race_distance) ∧
  (v_c * (race_time_A + time_diff_AC) = race_distance) ∧
  (v_c * (race_time_A + time_diff_AB + time_diff_BC) = race_distance) →
  race_time_A = 390 := by
sorry

end race_time_proof_l1035_103567


namespace max_k_value_l1035_103539

noncomputable def f (x : ℝ) := x + x * Real.log x

theorem max_k_value (k : ℤ) :
  (∀ x > 2, k * (x - 2) < f x) → k ≤ 4 :=
by sorry

end max_k_value_l1035_103539


namespace implication_q_not_p_l1035_103506

theorem implication_q_not_p (x : ℝ) : x^2 - x - 2 > 0 → x ≥ -1 := by
  sorry

end implication_q_not_p_l1035_103506


namespace function_condition_l1035_103588

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

theorem function_condition (a : ℝ) : f a (f a 0) = 3 * a → a = 4 := by
  sorry

end function_condition_l1035_103588


namespace corner_sum_equality_l1035_103519

/-- A matrix satisfying the given condition for any 2x2 sub-matrix -/
def SpecialMatrix (n : ℕ) := Matrix (Fin n) (Fin n) ℝ

/-- The condition that must hold for any 2x2 sub-matrix -/
def satisfies_condition (A : SpecialMatrix 2000) : Prop :=
  ∀ i j, i.val < 1999 → j.val < 1999 →
    A i j + A (Fin.succ i) (Fin.succ j) = A i (Fin.succ j) + A (Fin.succ i) j

/-- The theorem to be proved -/
theorem corner_sum_equality (A : SpecialMatrix 2000) (h : satisfies_condition A) :
  A 0 0 + A 1999 1999 = A 0 1999 + A 1999 0 := by
  sorry

end corner_sum_equality_l1035_103519


namespace sandy_balloons_l1035_103511

/-- Given the total number of blue balloons and the number of balloons Alyssa and Sally have,
    calculate the number of balloons Sandy has. -/
theorem sandy_balloons (total : ℕ) (alyssa : ℕ) (sally : ℕ) (h1 : total = 104) (h2 : alyssa = 37) (h3 : sally = 39) :
  total - alyssa - sally = 28 := by
  sorry

end sandy_balloons_l1035_103511


namespace distance_between_trees_l1035_103584

/-- Given a yard of length 1565 metres with 356 trees planted at equal distances
    (including one at each end), the distance between two consecutive trees
    is equal to 1565 / (356 - 1) metres. -/
theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) 
    (h1 : yard_length = 1565)
    (h2 : num_trees = 356) :
    (yard_length : ℚ) / (num_trees - 1) = 1565 / 355 :=
by sorry

end distance_between_trees_l1035_103584


namespace prob_at_least_6_heads_value_l1035_103561

/-- The probability of getting at least 6 heads when flipping a fair coin 8 times -/
def prob_at_least_6_heads : ℚ :=
  (Nat.choose 8 6 + Nat.choose 8 7 + Nat.choose 8 8) / 2^8

/-- Theorem stating that the probability of getting at least 6 heads
    when flipping a fair coin 8 times is 37/256 -/
theorem prob_at_least_6_heads_value :
  prob_at_least_6_heads = 37 / 256 := by
  sorry

end prob_at_least_6_heads_value_l1035_103561


namespace local_minimum_condition_l1035_103592

/-- The function f(x) = x(x - m)² has a local minimum at x = 2 if and only if m = 6 -/
theorem local_minimum_condition (m : ℝ) : 
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), x * (x - m)^2 ≥ 2 * (2 - m)^2) ↔ m = 6 :=
sorry

end local_minimum_condition_l1035_103592


namespace distance_difference_l1035_103523

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Grayson's first leg speed in mph -/
def grayson_speed1 : ℝ := 25

/-- Grayson's first leg time in hours -/
def grayson_time1 : ℝ := 1

/-- Grayson's second leg speed in mph -/
def grayson_speed2 : ℝ := 20

/-- Grayson's second leg time in hours -/
def grayson_time2 : ℝ := 0.5

/-- Rudy's speed in mph -/
def rudy_speed : ℝ := 10

/-- Rudy's time in hours -/
def rudy_time : ℝ := 3

/-- The difference in distance traveled between Grayson and Rudy -/
theorem distance_difference : 
  distance grayson_speed1 grayson_time1 + distance grayson_speed2 grayson_time2 - 
  distance rudy_speed rudy_time = 5 := by
  sorry

end distance_difference_l1035_103523


namespace evaluate_expression_l1035_103508

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 4) :
  y * (y - 2 * x) = 0 := by
  sorry

end evaluate_expression_l1035_103508


namespace inequality_solution_l1035_103568

theorem inequality_solution (x : ℕ+) : 4 - (x : ℝ) > 1 ↔ x = 1 ∨ x = 2 := by
  sorry

end inequality_solution_l1035_103568


namespace room_observation_ratio_l1035_103575

-- Define the room dimensions
def room_length : ℝ := 40
def room_width : ℝ := 40

-- Define the area observed by both guards
def area_observed_by_both : ℝ := 400

-- Define the total area of the room
def total_area : ℝ := room_length * room_width

-- Theorem to prove
theorem room_observation_ratio :
  total_area / area_observed_by_both = 4 := by
  sorry


end room_observation_ratio_l1035_103575


namespace smallest_integer_divisible_l1035_103551

theorem smallest_integer_divisible (n : ℕ) : n = 43179 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 21) = 48 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 21) = 64 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 21) = 75 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 21) = 108 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 21) = 48 * k₁ ∧ (n + 21) = 64 * k₂ ∧ (n + 21) = 75 * k₃ ∧ (n + 21) = 108 * k₄) := by
sorry

end smallest_integer_divisible_l1035_103551


namespace book_reading_ratio_l1035_103504

/-- The number of books William read last month -/
def william_last_month : ℕ := 6

/-- The number of books Brad read last month -/
def brad_last_month : ℕ := 3 * william_last_month

/-- The number of books Brad read this month -/
def brad_this_month : ℕ := 8

/-- The difference in total books read between William and Brad over two months -/
def difference_total : ℕ := 4

/-- The number of books William read this month -/
def william_this_month : ℕ := william_last_month + brad_last_month + brad_this_month + difference_total - (brad_last_month + brad_this_month)

theorem book_reading_ratio : 
  william_this_month / brad_this_month = 3 ∧ william_this_month % brad_this_month = 0 := by
  sorry

end book_reading_ratio_l1035_103504


namespace fraction_denominator_problem_l1035_103520

theorem fraction_denominator_problem (n d : ℤ) : 
  d = n - 4 ∧ n + 6 = 3 * d → d = 5 := by
  sorry

end fraction_denominator_problem_l1035_103520


namespace fourteen_n_divisibility_l1035_103557

theorem fourteen_n_divisibility (n d : ℕ) (p₁ p₂ p₃ : ℕ) 
  (h1 : 0 < n ∧ n < 200)
  (h2 : Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃)
  (h3 : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (h4 : n = p₁ * p₂ * p₃)
  (h5 : (14 * n) % d = 0) : 
  d = n := by
sorry

end fourteen_n_divisibility_l1035_103557


namespace other_divisor_problem_l1035_103507

theorem other_divisor_problem (n : ℕ) (h1 : n = 266) (h2 : n % 33 = 2) : 
  ∃ (x : ℕ), x ≠ 33 ∧ n % x = 2 ∧ x = 132 ∧ ∀ y : ℕ, y ≠ 33 → n % y = 2 → y ≤ x :=
by sorry

end other_divisor_problem_l1035_103507


namespace intersection_of_A_and_B_l1035_103502

-- Define the sets A and B
def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {5, 7, 8, 9}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {5, 8} := by
  sorry

end intersection_of_A_and_B_l1035_103502


namespace triangle_formation_l1035_103540

/-- A line in 2D space represented by ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three lines form a triangle --/
def form_triangle (l1 l2 l3 : Line) : Prop :=
  ∃ (x y : ℝ), (l1.a * x + l1.b * y = l1.c) ∧ 
                (l2.a * x + l2.b * y = l2.c) ∧ 
                (l3.a * x + l3.b * y = l3.c)

theorem triangle_formation (m : ℝ) : 
  ¬(form_triangle 
      ⟨1, 1, 2⟩  -- x + y = 2
      ⟨m, 1, 0⟩  -- mx + y = 0
      ⟨1, -1, 4⟩ -- x - y = 4
    ) ↔ m = 1/3 ∨ m = 1 ∨ m = -1 := by
  sorry

end triangle_formation_l1035_103540


namespace max_sum_sides_is_ten_l1035_103525

/-- Represents a configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  
/-- Represents a region formed by the intersection of lines -/
structure Region where
  num_sides : ℕ

/-- Represents two neighboring regions -/
structure NeighboringRegions where
  region1 : Region
  region2 : Region

/-- The maximum sum of sides for two neighboring regions in a configuration with 7 lines -/
def max_sum_sides (config : LineConfiguration) : ℕ :=
  10

/-- Theorem: The maximum sum of sides for two neighboring regions in a configuration with 7 lines is 10 -/
theorem max_sum_sides_is_ten (config : LineConfiguration) 
  (h : config.num_lines = 7) : 
  ∀ (neighbors : NeighboringRegions), 
    neighbors.region1.num_sides + neighbors.region2.num_sides ≤ max_sum_sides config :=
by
  sorry

#check max_sum_sides_is_ten

end max_sum_sides_is_ten_l1035_103525


namespace circle_equation_l1035_103543

/-- The equation of a circle with center (2, -1) and tangent to the line x - y + 1 = 0 is (x-2)² + (y+1)² = 8 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -1)
  let line (x y : ℝ) := x - y + 1 = 0
  let is_tangent (c : ℝ × ℝ) (r : ℝ) (l : ℝ → ℝ → Prop) := 
    ∃ p : ℝ × ℝ, l p.1 p.2 ∧ (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2
  let circle_eq (c : ℝ × ℝ) (r : ℝ) (x y : ℝ) := 
    (x - c.1)^2 + (y - c.2)^2 = r^2
  ∃ r : ℝ, is_tangent center r line → 
    circle_eq center r x y ↔ (x - 2)^2 + (y + 1)^2 = 8 :=
by
  sorry

end circle_equation_l1035_103543


namespace dave_ticket_difference_l1035_103545

theorem dave_ticket_difference (toys clothes : ℕ) 
  (h1 : toys = 12) 
  (h2 : clothes = 7) : 
  toys - clothes = 5 := by
  sorry

end dave_ticket_difference_l1035_103545


namespace reading_time_per_day_l1035_103576

-- Define the given conditions
def num_books : ℕ := 3
def num_days : ℕ := 10
def reading_rate : ℕ := 100 -- words per hour
def book1_words : ℕ := 200
def book2_words : ℕ := 400
def book3_words : ℕ := 300

-- Define the theorem
theorem reading_time_per_day :
  let total_words := book1_words + book2_words + book3_words
  let total_hours := total_words / reading_rate
  let total_minutes := total_hours * 60
  total_minutes / num_days = 54 := by
sorry


end reading_time_per_day_l1035_103576


namespace sam_money_value_l1035_103556

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The number of pennies Sam has -/
def num_pennies : ℕ := 9

/-- The number of quarters Sam has -/
def num_quarters : ℕ := 7

/-- The total value of Sam's money in dollars -/
def total_value : ℚ := num_pennies * penny_value + num_quarters * quarter_value

theorem sam_money_value : total_value = 184 / 100 := by sorry

end sam_money_value_l1035_103556


namespace gcd_factorial_eight_ten_l1035_103546

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem gcd_factorial_eight_ten : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end gcd_factorial_eight_ten_l1035_103546


namespace sphere_in_cube_surface_area_l1035_103578

/-- The surface area of a sphere inscribed in a cube with edge length 2 is 8π. -/
theorem sphere_in_cube_surface_area :
  let cube_edge : ℝ := 2
  let sphere_diameter : ℝ := cube_edge
  let sphere_radius : ℝ := sphere_diameter / 2
  let sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius ^ 2
  sphere_surface_area = 8 * Real.pi := by
  sorry

end sphere_in_cube_surface_area_l1035_103578


namespace solve_equation_l1035_103585

/-- The original equation -/
def original_equation (x a : ℚ) : Prop :=
  (2*x - 1) / 5 + 1 = (x + a) / 2

/-- The incorrect equation due to mistake -/
def incorrect_equation (x a : ℚ) : Prop :=
  2*(2*x - 1) + 1 = 5*(x + a)

/-- Theorem stating the correct values of a and x -/
theorem solve_equation :
  ∃ (a : ℚ), (incorrect_equation (-6) a) ∧ 
  (∀ x : ℚ, original_equation x a ↔ x = 3) ∧ 
  a = 1 := by sorry

end solve_equation_l1035_103585


namespace marble_arrangement_theorem_l1035_103513

/-- The number of blue marbles -/
def blue_marbles : ℕ := 7

/-- The maximum number of yellow marbles that can be arranged with the blue marbles
    such that the number of marbles with same-color right neighbors equals
    the number with different-color right neighbors -/
def max_yellow_marbles : ℕ := 19

/-- The total number of marbles -/
def total_marbles : ℕ := blue_marbles + max_yellow_marbles

/-- The number of ways to arrange the marbles satisfying the condition -/
def arrangement_count : ℕ := Nat.choose (max_yellow_marbles + blue_marbles + 1) blue_marbles

theorem marble_arrangement_theorem :
  arrangement_count % 1000 = 970 := by
  sorry

end marble_arrangement_theorem_l1035_103513


namespace geometric_mean_relationship_l1035_103590

theorem geometric_mean_relationship (m : ℝ) : 
  (m = 4 → m^2 = 2 * 8) ∧ ¬(m^2 = 2 * 8 → m = 4) := by
  sorry

end geometric_mean_relationship_l1035_103590


namespace c_used_car_for_13_hours_l1035_103516

/-- Represents the car rental scenario -/
structure CarRental where
  totalCost : ℝ
  aHours : ℝ
  bHours : ℝ
  bPaid : ℝ
  cHours : ℝ

/-- Theorem stating that under the given conditions, c used the car for 13 hours -/
theorem c_used_car_for_13_hours (rental : CarRental) 
  (h1 : rental.totalCost = 720)
  (h2 : rental.aHours = 9)
  (h3 : rental.bHours = 10)
  (h4 : rental.bPaid = 225) :
  rental.cHours = 13 := by
  sorry

#check c_used_car_for_13_hours

end c_used_car_for_13_hours_l1035_103516


namespace y_value_proof_l1035_103524

theorem y_value_proof (y : ℝ) (h : (9 : ℝ) / y^3 = y / 81) : y = 3 * Real.sqrt 3 := by
  sorry

end y_value_proof_l1035_103524


namespace greatest_digit_sum_base_seven_l1035_103547

def base_seven_representation (n : ℕ) : List ℕ := sorry

def digit_sum (digits : List ℕ) : ℕ := sorry

def is_valid_base_seven (digits : List ℕ) : Prop := sorry

theorem greatest_digit_sum_base_seven :
  ∃ (n : ℕ), n < 2890 ∧
    (∀ (m : ℕ), m < 2890 →
      digit_sum (base_seven_representation m) ≤ digit_sum (base_seven_representation n)) ∧
    digit_sum (base_seven_representation n) = 23 :=
  sorry

end greatest_digit_sum_base_seven_l1035_103547


namespace mairead_exercise_distance_l1035_103579

theorem mairead_exercise_distance :
  let run_distance : ℝ := 40
  let walk_distance : ℝ := (3/5) * run_distance
  let jog_distance : ℝ := (1/5) * walk_distance
  let total_distance : ℝ := run_distance + walk_distance + jog_distance
  total_distance = 64.8 := by
  sorry

end mairead_exercise_distance_l1035_103579


namespace locus_of_midpoint_l1035_103589

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 13

-- Define a point P on the circle
def point_P (x y : ℝ) : Prop := circle_O x y

-- Define Q as the foot of the perpendicular from P to the y-axis
def point_Q (x y : ℝ) : Prop := x = 0

-- Define M as the midpoint of PQ
def point_M (x y px py : ℝ) : Prop := x = px / 2 ∧ y = py

-- Theorem statement
theorem locus_of_midpoint :
  ∀ (x y px py : ℝ),
  point_P px py →
  point_Q 0 py →
  point_M x y px py →
  (x^2 / (13/4) + y^2 / 13 = 1) :=
by sorry

end locus_of_midpoint_l1035_103589


namespace man_age_year_l1035_103566

theorem man_age_year (x : ℕ) (birth_year : ℕ) : 
  (1850 ≤ birth_year) ∧ (birth_year ≤ 1900) →
  (x^2 = birth_year + x) →
  (birth_year + x = 1892) := by
  sorry

end man_age_year_l1035_103566


namespace max_steps_17_steps_17_possible_l1035_103515

/-- Represents the number of toothpicks used for n steps in Mandy's staircase -/
def toothpicks (n : ℕ) : ℕ := n * (n + 5)

/-- Theorem stating that 17 is the maximum number of steps that can be built with 380 toothpicks -/
theorem max_steps_17 :
  ∀ n : ℕ, toothpicks n ≤ 380 → n ≤ 17 :=
by
  sorry

/-- Theorem stating that 17 steps can indeed be built with 380 toothpicks -/
theorem steps_17_possible :
  toothpicks 17 ≤ 380 :=
by
  sorry

end max_steps_17_steps_17_possible_l1035_103515


namespace kenneth_earnings_l1035_103594

/-- Kenneth's earnings problem -/
theorem kenneth_earnings (earnings : ℝ) 
  (h1 : earnings * 0.1 + earnings * 0.15 + 75 + 80 + 405 = earnings) : 
  earnings = 746.67 := by
sorry

end kenneth_earnings_l1035_103594


namespace apples_per_box_l1035_103573

theorem apples_per_box (total_apples : ℕ) (num_boxes : ℕ) (apples_per_box : ℕ) : 
  total_apples = 49 → num_boxes = 7 → total_apples = num_boxes * apples_per_box → apples_per_box = 7 := by
  sorry

end apples_per_box_l1035_103573


namespace fraction_proportion_l1035_103598

theorem fraction_proportion (x y : ℚ) (h : y ≠ 0) :
  (x / y) / (2 / 5) = (3 / 7) / (6 / 5) → x / y = 1 / 7 := by
  sorry

end fraction_proportion_l1035_103598


namespace prob_at_least_three_same_value_l1035_103500

def num_dice : ℕ := 5
def num_sides : ℕ := 8

def prob_at_least_three_same : ℚ :=
  (num_dice.choose 3) * (1 / num_sides^2) * ((num_sides - 1) / num_sides)^2 +
  (num_dice.choose 4) * (1 / num_sides^3) * ((num_sides - 1) / num_sides) +
  (1 / num_sides^4)

theorem prob_at_least_three_same_value :
  prob_at_least_three_same = 526 / 4096 := by
  sorry

end prob_at_least_three_same_value_l1035_103500


namespace negative_sqrt_point_eight_one_equals_negative_point_nine_l1035_103569

theorem negative_sqrt_point_eight_one_equals_negative_point_nine :
  -Real.sqrt 0.81 = -0.9 := by
  sorry

end negative_sqrt_point_eight_one_equals_negative_point_nine_l1035_103569


namespace circumcircumcircumcoronene_tilings_l1035_103503

/-- Represents a tiling of a hexagon with edge length n using diamonds of side 1 -/
def HexagonTiling (n : ℕ) : Type := Unit

/-- The number of valid tilings for a hexagon with edge length n -/
def count_tilings (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of tilings for a hexagon with edge length 5 is 267227532 -/
theorem circumcircumcircumcoronene_tilings :
  count_tilings 5 = 267227532 := by sorry

end circumcircumcircumcoronene_tilings_l1035_103503


namespace arithmetic_sequence_formula_l1035_103583

theorem arithmetic_sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 4) (h2 : ∀ n : ℕ, a (n + 1) - a n = 3) :
  ∀ n : ℕ, a n = 3 * n + 1 := by
sorry

end arithmetic_sequence_formula_l1035_103583


namespace softball_team_ratio_l1035_103514

theorem softball_team_ratio :
  ∀ (men women : ℕ),
  men + women = 16 →
  women = men + 2 →
  (men : ℚ) / women = 7 / 9 :=
by
  sorry

end softball_team_ratio_l1035_103514


namespace trig_simplification_l1035_103536

open Real

theorem trig_simplification (α : ℝ) (n : ℤ) :
  ((-sin (α + π) + sin (-α) - tan (2*π + α)) / 
   (tan (α + π) + cos (-α) + cos (π - α)) = -1) ∧
  ((sin (α + n*π) + sin (α - n*π)) / 
   (sin (α + n*π) * cos (α - n*π)) = 
     if n % 2 = 0 then 2 / cos α else -2 / cos α) := by
  sorry

end trig_simplification_l1035_103536


namespace sebastians_high_school_students_l1035_103527

theorem sebastians_high_school_students (s m : ℕ) : 
  s = 4 * m →  -- Sebastian's high school has 4 times as many students as Mia's
  s + m = 3000 →  -- The total number of students in both schools is 3000
  s = 2400 :=  -- Sebastian's high school has 2400 students
by sorry

end sebastians_high_school_students_l1035_103527


namespace smallest_multiple_of_45_and_75_not_11_l1035_103533

theorem smallest_multiple_of_45_and_75_not_11 : 
  (∃ n : ℕ+, n * 45 = 225 ∧ n * 75 = 225) ∧ 
  (¬ ∃ m : ℕ+, m * 11 = 225) ∧
  (∀ k : ℕ+, k < 225 → ¬(∃ p : ℕ+, p * 45 = k ∧ p * 75 = k) ∨ (∃ q : ℕ+, q * 11 = k)) := by
  sorry

end smallest_multiple_of_45_and_75_not_11_l1035_103533


namespace sugar_measurement_l1035_103562

theorem sugar_measurement (required_sugar : Rat) (cup_capacity : Rat) (fills : Nat) : 
  required_sugar = 15/4 ∧ cup_capacity = 1/3 → fills = 12 := by
  sorry

end sugar_measurement_l1035_103562


namespace race_time_calculation_l1035_103577

/-- 
Given a 100-meter race where:
- Runner A beats runner B by 20 meters
- Runner B finishes the race in 45 seconds

This theorem proves that runner A finishes the race in 36 seconds.
-/
theorem race_time_calculation (race_distance : ℝ) (b_time : ℝ) (distance_difference : ℝ) 
  (h1 : race_distance = 100)
  (h2 : b_time = 45)
  (h3 : distance_difference = 20) : 
  ∃ (a_time : ℝ), a_time = 36 ∧ 
  (race_distance / a_time = (race_distance - distance_difference) / a_time) ∧
  ((race_distance - distance_difference) / a_time = race_distance / b_time) :=
sorry

end race_time_calculation_l1035_103577


namespace line_intersects_x_axis_l1035_103554

/-- The line equation 2y - 3x = 15 intersects the x-axis at the point (-5, 0) -/
theorem line_intersects_x_axis :
  ∃ (x : ℝ), 2 * 0 - 3 * x = 15 ∧ x = -5 := by
  sorry

end line_intersects_x_axis_l1035_103554


namespace find_divisor_l1035_103535

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 125 → 
  quotient = 8 → 
  remainder = 5 → 
  dividend = divisor * quotient + remainder →
  divisor = 15 := by
  sorry

end find_divisor_l1035_103535


namespace no_prime_factor_seven_mod_eight_l1035_103555

theorem no_prime_factor_seven_mod_eight (n : ℕ+) :
  ∀ p : ℕ, Prime p → p ∣ (2^(n : ℕ) + 1) → p % 8 ≠ 7 := by
  sorry

end no_prime_factor_seven_mod_eight_l1035_103555


namespace sqrt_sum_max_value_l1035_103534

theorem sqrt_sum_max_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 2 → Real.sqrt x + Real.sqrt y ≤ m :=
sorry

end sqrt_sum_max_value_l1035_103534


namespace curling_survey_probability_l1035_103544

/-- Represents the survey data and selection process for the Winter Olympic Games curling interest survey. -/
structure CurlingSurvey where
  total_participants : Nat
  male_to_female_ratio : Rat
  interested_ratio : Rat
  uninterested_females : Nat
  selected_interested : Nat
  chosen_promoters : Nat

/-- Calculates the probability of selecting at least one female from the chosen promoters. -/
def probability_at_least_one_female (survey : CurlingSurvey) : Rat :=
  sorry

/-- Theorem stating that given the survey conditions, the probability of selecting at least one female is 9/14. -/
theorem curling_survey_probability (survey : CurlingSurvey) 
  (h1 : survey.total_participants = 600)
  (h2 : survey.male_to_female_ratio = 2/1)
  (h3 : survey.interested_ratio = 2/3)
  (h4 : survey.uninterested_females = 50)
  (h5 : survey.selected_interested = 8)
  (h6 : survey.chosen_promoters = 2) :
  probability_at_least_one_female survey = 9/14 :=
sorry

end curling_survey_probability_l1035_103544


namespace two_digit_number_problem_l1035_103549

theorem two_digit_number_problem : ∃ x : ℕ, 
  10 ≤ x ∧ x < 100 ∧ 10 * x + 6 = x + 474 → x = 52 := by
  sorry

end two_digit_number_problem_l1035_103549


namespace pie_cost_l1035_103595

def mary_initial_amount : ℕ := 58
def mary_remaining_amount : ℕ := 52

theorem pie_cost : mary_initial_amount - mary_remaining_amount = 6 := by
  sorry

end pie_cost_l1035_103595


namespace great_circle_bisects_angle_l1035_103522

-- Define the sphere
def Sphere : Type := ℝ × ℝ × ℝ

-- Define the north pole
def N : Sphere := (0, 0, 1)

-- Define a great circle
def GreatCircle (p q : Sphere) : Type := sorry

-- Define a point on the equator
def OnEquator (p : Sphere) : Prop := sorry

-- Define equidistance from a point
def Equidistant (a b c : Sphere) : Prop := sorry

-- Define angle bisection on a sphere
def AngleBisector (a b c d : Sphere) : Prop := sorry

-- Theorem statement
theorem great_circle_bisects_angle 
  (A B C : Sphere) 
  (h1 : GreatCircle N A)
  (h2 : GreatCircle N B)
  (h3 : Equidistant N A B)
  (h4 : OnEquator C) :
  AngleBisector C N A B :=
sorry

end great_circle_bisects_angle_l1035_103522


namespace f_has_two_roots_l1035_103531

/-- The function f(x) = x^4 + 5x^3 + 6x^2 - 4x - 16 -/
def f (x : ℝ) : ℝ := x^4 + 5*x^3 + 6*x^2 - 4*x - 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4*x^3 + 15*x^2 + 12*x - 4

theorem f_has_two_roots :
  ∃! (a b : ℝ), a < b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end f_has_two_roots_l1035_103531


namespace problem_solution_l1035_103517

def f (x : ℝ) := |3*x - 2| + |x - 2|

theorem problem_solution :
  (∀ x : ℝ, f x ≤ 8 ↔ x ∈ Set.Icc (-1) 3) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ≠ 0 → f x ≥ (m^2 - m + 2) * |x|) → m ∈ Set.Icc 0 1) := by
  sorry

end problem_solution_l1035_103517


namespace equation_solution_l1035_103593

theorem equation_solution : ∃ x : ℝ, (5 + 3.5 * x = 2.5 * x - 25) ∧ (x = -30) := by
  sorry

end equation_solution_l1035_103593


namespace truncated_prism_edges_l1035_103560

/-- Represents a truncated rectangular prism -/
structure TruncatedPrism where
  originalEdges : ℕ
  normalTruncations : ℕ
  intersectingTruncations : ℕ

/-- Calculates the number of edges after truncation -/
def edgesAfterTruncation (p : TruncatedPrism) : ℕ :=
  p.originalEdges - p.intersectingTruncations +
  p.normalTruncations * 3 + p.intersectingTruncations * 4

/-- Theorem stating that the specific truncation scenario results in 33 edges -/
theorem truncated_prism_edges :
  ∀ p : TruncatedPrism,
  p.originalEdges = 12 ∧
  p.normalTruncations = 6 ∧
  p.intersectingTruncations = 1 →
  edgesAfterTruncation p = 33 :=
by
  sorry


end truncated_prism_edges_l1035_103560


namespace race_percentage_l1035_103518

theorem race_percentage (v_Q : ℝ) (h : v_Q > 0) : 
  let v_P := v_Q * (1 + 25/100)
  (300 / v_P = (300 - 60) / v_Q) → 
  ∃ (p : ℝ), v_P = v_Q * (1 + p/100) ∧ p = 25 :=
by sorry

end race_percentage_l1035_103518


namespace tank_fill_time_l1035_103501

-- Define the fill/drain rates for each pipe
def rate_A : ℚ := 1 / 10
def rate_B : ℚ := 1 / 20
def rate_C : ℚ := -(1 / 30)  -- Negative because it's draining

-- Define the combined rate
def combined_rate : ℚ := rate_A + rate_B + rate_C

-- Theorem to prove
theorem tank_fill_time :
  (1 : ℚ) / combined_rate = 60 / 7 := by sorry

end tank_fill_time_l1035_103501


namespace first_quadrant_sufficient_not_necessary_l1035_103581

-- Define the complex number z
def z (a : ℝ) : ℂ := a + (a + 1) * Complex.I

-- Define the condition for a point to be in the first quadrant
def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

-- Statement of the theorem
theorem first_quadrant_sufficient_not_necessary (a : ℝ) :
  (is_in_first_quadrant (z a) → a > -1) ∧
  ¬(a > -1 → is_in_first_quadrant (z a)) :=
sorry

end first_quadrant_sufficient_not_necessary_l1035_103581


namespace range_of_a_l1035_103580

-- Define a monotonically decreasing function
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : MonotonicallyDecreasing f) 
  (h2 : f (2 - a^2) > f a) : 
  a > 1 ∨ a < -2 := by
  sorry

end range_of_a_l1035_103580


namespace gear_rpm_problem_l1035_103591

/-- The number of revolutions per minute for gear q -/
def q_rpm : ℝ := 40

/-- The duration in minutes -/
def duration : ℝ := 0.5

/-- The difference in revolutions between gear q and gear p after 30 seconds -/
def revolution_difference : ℝ := 15

/-- The number of revolutions per minute for gear p -/
def p_rpm : ℝ := 10

theorem gear_rpm_problem :
  q_rpm * duration - revolution_difference = p_rpm * duration :=
by sorry

end gear_rpm_problem_l1035_103591


namespace least_number_for_divisibility_l1035_103587

theorem least_number_for_divisibility (n m : ℕ) (h : n = 1056 ∧ m = 26) :
  ∃ x : ℕ, (x = 10 ∧ (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0) :=
sorry

end least_number_for_divisibility_l1035_103587


namespace intersection_at_most_one_point_f_composition_half_l1035_103529

-- Statement B
theorem intersection_at_most_one_point (f : ℝ → ℝ) :
  ∃ (y : ℝ), ∀ (y' : ℝ), f 1 = y' → y = y' :=
sorry

-- Statement D
def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem f_composition_half : f (f (1/2)) = 1 :=
sorry

end intersection_at_most_one_point_f_composition_half_l1035_103529


namespace quintic_integer_root_counts_l1035_103558

/-- The set of possible numbers of integer roots (counting multiplicity) for a quintic polynomial with integer coefficients -/
def QuinticIntegerRootCounts : Set ℕ := {0, 1, 2, 3, 4, 5}

/-- A quintic polynomial with integer coefficients -/
structure QuinticPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The number of integer roots (counting multiplicity) of a quintic polynomial -/
def integerRootCount (p : QuinticPolynomial) : ℕ := sorry

theorem quintic_integer_root_counts (p : QuinticPolynomial) :
  integerRootCount p ∈ QuinticIntegerRootCounts := by sorry

end quintic_integer_root_counts_l1035_103558


namespace train_speed_l1035_103538

/-- The speed of a train crossing a platform -/
theorem train_speed (train_length platform_length : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  platform_length = 165 →
  crossing_time = 7.499400047996161 →
  ∃ (speed : ℝ), abs (speed - 132.01) < 0.01 ∧ 
  speed = (train_length + platform_length) / crossing_time * 3.6 := by
  sorry

end train_speed_l1035_103538


namespace quadratic_root_value_l1035_103528

theorem quadratic_root_value (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 - 2*x + a^2 - 1 = 0 ↔ x = 0 ∨ x ≠ 0) →
  (a - 1 ≠ 0) →
  a = -1 := by
sorry

end quadratic_root_value_l1035_103528


namespace total_amount_shared_l1035_103571

theorem total_amount_shared (z y x : ℝ) : 
  z = 150 →
  y = 1.2 * z →
  x = 1.25 * y →
  x + y + z = 555 :=
by
  sorry

end total_amount_shared_l1035_103571


namespace circle_center_l1035_103572

/-- The equation of a circle in the form (x - h)² + (y - k)² = r² 
    where (h, k) is the center and r is the radius -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The given equation of the circle -/
def GivenEquation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y = 16

theorem circle_center : 
  ∃ (r : ℝ), ∀ (x y : ℝ), GivenEquation x y ↔ CircleEquation 4 2 r x y :=
sorry

end circle_center_l1035_103572


namespace x_value_in_set_A_l1035_103564

-- Define the set A
def A (x : ℝ) : Set ℝ := {0, -1, x}

-- Theorem statement
theorem x_value_in_set_A (x : ℝ) (h1 : x^2 ∈ A x) (h2 : 0 ≠ -1 ∧ 0 ≠ x ∧ -1 ≠ x) : x = 1 := by
  sorry

end x_value_in_set_A_l1035_103564


namespace square_side_length_average_l1035_103541

theorem square_side_length_average : 
  let areas : List ℝ := [25, 64, 121, 196]
  let side_lengths := areas.map Real.sqrt
  (side_lengths.sum / side_lengths.length : ℝ) = 9.5 := by
  sorry

end square_side_length_average_l1035_103541


namespace trapezoid_area_is_half_sq_dm_l1035_103582

/-- A trapezoid with specific measurements -/
structure Trapezoid where
  smallBase : ℝ
  adjacentAngle : ℝ
  diagonalAngle : ℝ

/-- The area of a trapezoid with given measurements -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  0.5

/-- Theorem stating that a trapezoid with the given measurements has an area of 0.5 square decimeters -/
theorem trapezoid_area_is_half_sq_dm (t : Trapezoid) 
    (h1 : t.smallBase = 1)
    (h2 : t.adjacentAngle = 135 * π / 180)
    (h3 : t.diagonalAngle = 150 * π / 180) :
    trapezoidArea t = 0.5 := by
  sorry

end trapezoid_area_is_half_sq_dm_l1035_103582


namespace distance_difference_around_block_l1035_103599

/-- The difference in distance run around a square block -/
theorem distance_difference_around_block (block_side_length street_width : ℝ) :
  block_side_length = 500 →
  street_width = 25 →
  (4 * (block_side_length + 2 * street_width)) - (4 * block_side_length) = 200 := by
  sorry

end distance_difference_around_block_l1035_103599


namespace same_direction_condition_l1035_103574

/-- Two vectors are in the same direction if one is a positive scalar multiple of the other -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m > 0 ∧ a = (m * b.1, m * b.2)

/-- The condition for vectors a and b to be in the same direction -/
theorem same_direction_condition (k : ℝ) :
  same_direction (k, 2) (1, 1) ↔ k = 2 := by
  sorry

#check same_direction_condition

end same_direction_condition_l1035_103574


namespace trajectory_length_l1035_103597

/-- The curve y = x^3 - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The line x = 2 on which point A moves -/
def line_x_eq_2 (x : ℝ) : Prop := x = 2

/-- The tangent line to the curve at point (x₀, f x₀) -/
def tangent_line (x₀ : ℝ) (x a : ℝ) : Prop :=
  a = (3 * x₀^2 - 1) * (x - x₀) + f x₀

/-- The condition for point A(2, a) to have a tangent line to the curve -/
def has_tangent (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, tangent_line x₀ 2 a

/-- The statement to be proved -/
theorem trajectory_length :
  ∀ a : ℝ, line_x_eq_2 2 →
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    has_tangent a ∧
    (∀ x : ℝ, has_tangent a → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  (∃ a_min a_max : ℝ, 
    (∀ a' : ℝ, has_tangent a' → a_min ≤ a' ∧ a' ≤ a_max) ∧
    a_max - a_min = 8) :=
by sorry

end trajectory_length_l1035_103597


namespace bisection_is_best_method_l1035_103586

/-- Represents a transmission line with a fault -/
structure TransmissionLine :=
  (hasElectricityAtA : Bool)
  (hasElectricityAtB : Bool)
  (hasFault : Bool)

/-- Represents different methods to locate a fault -/
inductive FaultLocationMethod
  | Method618
  | FractionMethod
  | BisectionMethod
  | BlindManClimbingMethod

/-- Determines the best method to locate a fault in a transmission line -/
def bestFaultLocationMethod (line : TransmissionLine) : FaultLocationMethod :=
  FaultLocationMethod.BisectionMethod

/-- Theorem stating that the bisection method is the best for locating a fault
    in a transmission line with electricity at A but not at B -/
theorem bisection_is_best_method (line : TransmissionLine)
  (h1 : line.hasElectricityAtA = true)
  (h2 : line.hasElectricityAtB = false)
  (h3 : line.hasFault = true) :
  bestFaultLocationMethod line = FaultLocationMethod.BisectionMethod :=
by sorry

end bisection_is_best_method_l1035_103586


namespace average_of_expressions_l1035_103521

theorem average_of_expressions (x : ℚ) : 
  (1/3 : ℚ) * ((2*x + 8) + (5*x + 3) + (3*x + 9)) = 3*x + 2 → x = -14 := by
sorry

end average_of_expressions_l1035_103521

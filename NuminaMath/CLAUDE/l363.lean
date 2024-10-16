import Mathlib

namespace NUMINAMATH_CALUDE_interest_group_count_l363_36310

/-- The number of students who joined at least one interest group -/
def students_in_interest_groups (science_tech : ℕ) (speech : ℕ) (both : ℕ) : ℕ :=
  science_tech + speech - both

theorem interest_group_count : 
  students_in_interest_groups 65 35 20 = 80 := by
sorry

end NUMINAMATH_CALUDE_interest_group_count_l363_36310


namespace NUMINAMATH_CALUDE_expression_equality_l363_36308

theorem expression_equality : 
  |Real.sqrt 3 - 2| - (1 / 2)⁻¹ - 2 * Real.sin (π / 3) = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l363_36308


namespace NUMINAMATH_CALUDE_cereal_eating_time_l363_36379

/-- The time it takes for Mr. Fat and Mr. Thin to eat 5 pounds of cereal together -/
theorem cereal_eating_time (fat_rate thin_rate : ℚ) (total_cereal : ℚ) : 
  fat_rate = 1 / 15 →
  thin_rate = 1 / 45 →
  total_cereal = 5 →
  (total_cereal / (fat_rate + thin_rate) : ℚ) = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l363_36379


namespace NUMINAMATH_CALUDE_election_vote_difference_l363_36371

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 6450 →
  candidate_percentage = 31 / 100 →
  ⌊(1 - candidate_percentage) * total_votes⌋ - ⌊candidate_percentage * total_votes⌋ = 2451 :=
by sorry

end NUMINAMATH_CALUDE_election_vote_difference_l363_36371


namespace NUMINAMATH_CALUDE_multiples_of_4_and_5_between_100_and_350_l363_36331

theorem multiples_of_4_and_5_between_100_and_350 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n % 5 = 0 ∧ 100 < n ∧ n < 350) (Finset.range 350)).card = 12 :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_4_and_5_between_100_and_350_l363_36331


namespace NUMINAMATH_CALUDE_circle_circumference_l363_36314

theorem circle_circumference (r : ℝ) (d : ℝ) (C : ℝ) :
  (d = 2 * r) → (C = π * d ∨ C = 2 * π * r) :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_l363_36314


namespace NUMINAMATH_CALUDE_largest_number_exists_largest_largest_is_998_l363_36334

def numbers : List ℚ := [0.989, 0.998, 0.899, 0.9899, 0.8999]

theorem largest_number (n : ℚ) (h : n ∈ numbers) : n ≤ 0.998 := by
  sorry

theorem exists_largest : ∃ (n : ℚ), n ∈ numbers ∧ ∀ (m : ℚ), m ∈ numbers → m ≤ n := by
  sorry

theorem largest_is_998 : ∃ (n : ℚ), n ∈ numbers ∧ n = 0.998 ∧ ∀ (m : ℚ), m ∈ numbers → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_largest_number_exists_largest_largest_is_998_l363_36334


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l363_36386

theorem algebraic_expression_value (x : ℝ) :
  x^2 + x + 5 = 8 → 2*x^2 + 2*x - 4 = 2 := by
  sorry

#check algebraic_expression_value

end NUMINAMATH_CALUDE_algebraic_expression_value_l363_36386


namespace NUMINAMATH_CALUDE_fishing_problem_l363_36355

theorem fishing_problem (blaine_catch : ℕ) (keith_catch : ℕ) : 
  blaine_catch = 5 → 
  keith_catch = 2 * blaine_catch → 
  blaine_catch + keith_catch = 15 := by
sorry

end NUMINAMATH_CALUDE_fishing_problem_l363_36355


namespace NUMINAMATH_CALUDE_value_range_of_f_l363_36330

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- State the theorem
theorem value_range_of_f :
  ∀ y ∈ Set.Icc (-3) 5, ∃ x ∈ Set.Icc 0 2, f x = y ∧
  ∀ x ∈ Set.Icc 0 2, f x ∈ Set.Icc (-3) 5 :=
by sorry

end NUMINAMATH_CALUDE_value_range_of_f_l363_36330


namespace NUMINAMATH_CALUDE_third_game_difference_l363_36311

/-- The number of people who watched the second game -/
def second_game_viewers : ℕ := 80

/-- The number of people who watched the first game -/
def first_game_viewers : ℕ := second_game_viewers - 20

/-- The total number of people who watched the games last week -/
def last_week_total : ℕ := 200

/-- The total number of people who watched the games this week -/
def this_week_total : ℕ := last_week_total + 35

/-- The number of people who watched the third game -/
def third_game_viewers : ℕ := this_week_total - (first_game_viewers + second_game_viewers)

theorem third_game_difference : 
  third_game_viewers - second_game_viewers = 15 := by sorry

end NUMINAMATH_CALUDE_third_game_difference_l363_36311


namespace NUMINAMATH_CALUDE_cube_plane_angle_l363_36356

/-- Given a cube with a plane passing through a side of its base, dividing the volume
    in the ratio m:n (where m ≤ n), the angle α between this plane and the base of
    the cube is given by α = arctan(2m / (m + n)). -/
theorem cube_plane_angle (m n : ℝ) (h : 0 < m ∧ m ≤ n) : 
  ∃ (α : ℝ), α = Real.arctan (2 * m / (m + n)) ∧
  ∃ (V₁ V₂ : ℝ), V₁ / V₂ = m / n ∧
  V₁ = (1/2) * (Real.tan α) ∧
  V₂ = 1 - (1/2) * (Real.tan α) := by
sorry

end NUMINAMATH_CALUDE_cube_plane_angle_l363_36356


namespace NUMINAMATH_CALUDE_august_has_five_tuesdays_l363_36373

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  first_day : DayOfWeek

/-- Given a month, returns the number of occurrences of each day of the week -/
def count_days (m : Month) : DayOfWeek → Nat :=
  sorry

/-- Returns true if the given day occurs exactly five times in the month -/
def occurs_five_times (d : DayOfWeek) (m : Month) : Prop :=
  count_days m d = 5

/-- Theorem: If July has five Fridays, then August must have five Tuesdays -/
theorem august_has_five_tuesdays
  (july : Month)
  (august : Month)
  (h1 : july.days = 31)
  (h2 : august.days = 31)
  (h3 : occurs_five_times DayOfWeek.Friday july) :
  occurs_five_times DayOfWeek.Tuesday august :=
sorry

end NUMINAMATH_CALUDE_august_has_five_tuesdays_l363_36373


namespace NUMINAMATH_CALUDE_play_attendance_l363_36350

theorem play_attendance (total_people : ℕ) (adult_price child_price : ℚ) (total_receipts : ℚ) :
  total_people = 610 →
  adult_price = 2 →
  child_price = 1 →
  total_receipts = 960 →
  ∃ (adults children : ℕ),
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_receipts ∧
    children = 260 :=
by sorry

end NUMINAMATH_CALUDE_play_attendance_l363_36350


namespace NUMINAMATH_CALUDE_novel_contest_first_prize_l363_36351

/-- The first place prize in a novel contest --/
def first_place_prize (total_prize : ℕ) (num_winners : ℕ) (second_prize : ℕ) (third_prize : ℕ) (other_prize : ℕ) : ℕ :=
  total_prize - (second_prize + third_prize + (num_winners - 3) * other_prize)

/-- Theorem stating the first place prize is $200 given the contest conditions --/
theorem novel_contest_first_prize :
  first_place_prize 800 18 150 120 22 = 200 := by
  sorry

end NUMINAMATH_CALUDE_novel_contest_first_prize_l363_36351


namespace NUMINAMATH_CALUDE_books_loaned_out_correct_loaned_books_l363_36324

/-- Proves that the number of books loaned out is 160 given the initial and final book counts and return rate -/
theorem books_loaned_out 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (return_rate : ℚ) : ℕ :=
  let loaned_books := (initial_books - final_books) / (1 - return_rate)
  160

/-- The number of books loaned out is 160 -/
theorem correct_loaned_books : books_loaned_out 300 244 (65/100) = 160 := by
  sorry

end NUMINAMATH_CALUDE_books_loaned_out_correct_loaned_books_l363_36324


namespace NUMINAMATH_CALUDE_minimum_spotted_blueeyed_rabbits_l363_36358

theorem minimum_spotted_blueeyed_rabbits 
  (total : ℕ) (spotted : ℕ) (blueeyed : ℕ) 
  (h_total : total = 100)
  (h_spotted : spotted = 53)
  (h_blueeyed : blueeyed = 73) :
  ∃ (both : ℕ), both ≥ 26 ∧ 
    ∀ (x : ℕ), x < 26 → spotted + blueeyed - x > total :=
by sorry

end NUMINAMATH_CALUDE_minimum_spotted_blueeyed_rabbits_l363_36358


namespace NUMINAMATH_CALUDE_both_a_and_b_must_join_at_least_one_of_a_or_b_must_join_l363_36320

-- Define the total number of doctors
def total_doctors : ℕ := 20

-- Define the number of doctors to be chosen
def team_size : ℕ := 5

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem for part (1)
theorem both_a_and_b_must_join : 
  combination (total_doctors - 2) (team_size - 2) = 816 := by sorry

-- Theorem for part (2)
theorem at_least_one_of_a_or_b_must_join : 
  2 * combination (total_doctors - 2) (team_size - 1) + 
  combination (total_doctors - 2) (team_size - 2) = 5661 := by sorry

end NUMINAMATH_CALUDE_both_a_and_b_must_join_at_least_one_of_a_or_b_must_join_l363_36320


namespace NUMINAMATH_CALUDE_karthiks_weight_upper_bound_l363_36305

-- Define the variables
def lower_bound : ℝ := 55
def upper_bound : ℝ := 58
def average_weight : ℝ := 56.5

-- Define the theorem
theorem karthiks_weight_upper_bound (X : ℝ) 
  (h1 : X > 50)  -- Karthik's brother's lower bound
  (h2 : X ≤ 62)  -- Karthik's upper bound
  (h3 : X ≤ 58)  -- Karthik's father's upper bound
  (h4 : (lower_bound + X) / 2 = average_weight)  -- Average condition
  : X = upper_bound := by
  sorry

end NUMINAMATH_CALUDE_karthiks_weight_upper_bound_l363_36305


namespace NUMINAMATH_CALUDE_reciprocal_equation_solution_l363_36319

theorem reciprocal_equation_solution (x : ℝ) : 
  3 - 1 / (4 * (1 - x)) = 2 * (1 / (4 * (1 - x))) → x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_solution_l363_36319


namespace NUMINAMATH_CALUDE_chocolate_ratio_problem_l363_36377

/-- The number of dark chocolate bars sold given the ratio and white chocolate bars sold -/
def dark_chocolate_bars (white_ratio : ℕ) (dark_ratio : ℕ) (white_bars : ℕ) : ℕ :=
  (dark_ratio * white_bars) / white_ratio

/-- Theorem: Given a ratio of 4:3 for white to dark chocolate and 20 white chocolate bars sold,
    the number of dark chocolate bars sold is 15 -/
theorem chocolate_ratio_problem :
  dark_chocolate_bars 4 3 20 = 15 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_ratio_problem_l363_36377


namespace NUMINAMATH_CALUDE_vector_dot_product_equation_l363_36316

/-- Given vectors a, b, c, and a dot product equation, prove that x = 1 -/
theorem vector_dot_product_equation (a b c : ℝ × ℝ) (x : ℝ) :
  a = (1, 1) →
  b = (-1, 3) →
  c = (2, x) →
  (3 • a + b) • c = 10 →
  x = 1 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_equation_l363_36316


namespace NUMINAMATH_CALUDE_greatest_multiple_of_nine_with_unique_digits_mod_1000_l363_36317

/-- A function that checks if a natural number has all unique digits -/
def hasUniqueDigits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 9 with all unique digits -/
def M : ℕ := sorry

theorem greatest_multiple_of_nine_with_unique_digits_mod_1000 :
  M % 1000 = 981 := by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_nine_with_unique_digits_mod_1000_l363_36317


namespace NUMINAMATH_CALUDE_jewelry_restock_cost_l363_36327

/-- Represents the inventory and pricing information for a jewelry item -/
structure JewelryItem where
  name : String
  capacity : Nat
  current : Nat
  price : Nat
  discount1 : Nat
  discount1Threshold : Nat
  discount2 : Nat
  discount2Threshold : Nat

/-- Calculates the total cost for restocking jewelry items -/
def calculateTotalCost (items : List JewelryItem) : Rat :=
  let itemCosts := items.map (fun item =>
    let quantity := item.capacity - item.current
    let basePrice := quantity * item.price
    let discountedPrice :=
      if quantity >= item.discount2Threshold then
        basePrice * (1 - item.discount2 / 100)
      else if quantity >= item.discount1Threshold then
        basePrice * (1 - item.discount1 / 100)
      else
        basePrice
    discountedPrice)
  let subtotal := itemCosts.sum
  let shippingFee := subtotal * (2 / 100)
  subtotal + shippingFee

/-- Theorem stating that the total cost to restock the jewelry showroom is $257.04 -/
theorem jewelry_restock_cost :
  let necklaces : JewelryItem := ⟨"Necklace", 20, 8, 5, 10, 10, 15, 15⟩
  let rings : JewelryItem := ⟨"Ring", 40, 25, 8, 5, 20, 12, 30⟩
  let bangles : JewelryItem := ⟨"Bangle", 30, 17, 6, 8, 15, 18, 25⟩
  calculateTotalCost [necklaces, rings, bangles] = 257.04 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_restock_cost_l363_36327


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l363_36328

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation
variable (parallel : Plane → Plane → Prop)
variable (lineParallelToPlane : Line → Plane → Prop)

-- Define the subset relation for lines in planes
variable (subset : Line → Plane → Prop)

-- Define specific planes and lines
variable (α β : Plane)
variable (a b : Line)

-- State the theorem
theorem parallel_planes_condition 
  (h1 : subset a α)
  (h2 : subset b α) :
  (∀ (α β : Plane), parallel α β → lineParallelToPlane a β ∧ lineParallelToPlane b β) ∧ 
  (∃ (α β : Plane) (a b : Line), 
    subset a α ∧ 
    subset b α ∧ 
    lineParallelToPlane a β ∧ 
    lineParallelToPlane b β ∧ 
    ¬parallel α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l363_36328


namespace NUMINAMATH_CALUDE_candy_difference_l363_36339

-- Define the initial variables
def candy_given : ℝ := 6.25
def candy_left : ℝ := 4.75

-- Define the theorem
theorem candy_difference : candy_given - candy_left = 1.50 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_l363_36339


namespace NUMINAMATH_CALUDE_profit_maximized_at_optimal_price_l363_36312

/-- The profit function for a product with a cost of 30 yuan per item,
    where x is the selling price and (200 - x) is the quantity sold. -/
def profit_function (x : ℝ) : ℝ := -x^2 + 230*x - 6000

/-- The selling price that maximizes the profit. -/
def optimal_price : ℝ := 115

theorem profit_maximized_at_optimal_price :
  ∀ x : ℝ, profit_function x ≤ profit_function optimal_price :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_optimal_price_l363_36312


namespace NUMINAMATH_CALUDE_team_average_score_l363_36396

theorem team_average_score (player1 player2 player3 player4 : ℝ) 
  (h1 : player1 = 20)
  (h2 : player2 = player1 / 2)
  (h3 : player3 = 6 * player2)
  (h4 : player4 = 3 * player3) :
  (player1 + player2 + player3 + player4) / 4 = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_team_average_score_l363_36396


namespace NUMINAMATH_CALUDE_probability_at_least_two_same_l363_36341

theorem probability_at_least_two_same (n : Nat) (s : Nat) :
  n = 8 →
  s = 8 →
  (1 - (Nat.factorial n) / (s^n : ℚ)) = 415 / 416 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_same_l363_36341


namespace NUMINAMATH_CALUDE_proof_by_contradiction_elements_l363_36392

/-- Elements used as conditions in a proof by contradiction -/
inductive ProofByContradictionElement
  | NegatedConclusion
  | OriginalConditions
  | AxiomsTheoremsDefinitions
  | OriginalConclusion

/-- The set of elements that should be used in a proof by contradiction -/
def ValidProofByContradictionElements : Set ProofByContradictionElement :=
  {ProofByContradictionElement.NegatedConclusion,
   ProofByContradictionElement.OriginalConditions,
   ProofByContradictionElement.AxiomsTheoremsDefinitions}

/-- Theorem stating which elements should be used in a proof by contradiction -/
theorem proof_by_contradiction_elements :
  ValidProofByContradictionElements =
    {ProofByContradictionElement.NegatedConclusion,
     ProofByContradictionElement.OriginalConditions,
     ProofByContradictionElement.AxiomsTheoremsDefinitions} :=
by sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_elements_l363_36392


namespace NUMINAMATH_CALUDE_point_on_line_angle_with_x_axis_line_equation_correct_l363_36387

/-- The equation of a line passing through (2, 2) and making a 60° angle with the x-axis -/
def line_equation (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x - 2 * Real.sqrt 3 + 2

/-- The point (2, 2) lies on the line -/
theorem point_on_line : line_equation 2 2 := by sorry

/-- The angle between the line and the x-axis is 60° -/
theorem angle_with_x_axis : 
  Real.arctan (Real.sqrt 3) = 60 * π / 180 := by sorry

/-- The line equation is correct -/
theorem line_equation_correct (x y : ℝ) :
  line_equation x y ↔ 
    (∃ k : ℝ, y - 2 = k * (x - 2) ∧ 
              k = Real.tan (60 * π / 180)) := by sorry

end NUMINAMATH_CALUDE_point_on_line_angle_with_x_axis_line_equation_correct_l363_36387


namespace NUMINAMATH_CALUDE_impossible_sum_nine_l363_36300

/-- Represents the numbers on the faces of a cube -/
def CubeFaces := Fin 6 → Nat

/-- The sum of a pair of opposite faces -/
def oppositeFaceSum (faces : CubeFaces) (i j : Fin 6) : Nat :=
  faces i + faces j

/-- Predicate for a valid numbering of cube faces -/
def validCubeNumbering (faces : CubeFaces) : Prop :=
  (∀ i : Fin 6, faces i ∈ Finset.range 7) ∧ 
  (∀ i j : Fin 6, i ≠ j → faces i ≠ faces j) ∧
  (∃ i j : Fin 6, i ≠ j ∧ oppositeFaceSum faces i j = 11)

/-- Theorem: Given a valid cube numbering, the sum of 9 is impossible for any remaining pair of opposite faces -/
theorem impossible_sum_nine (faces : CubeFaces) (h : validCubeNumbering faces) :
  ∀ i j : Fin 6, i ≠ j → oppositeFaceSum faces i j ≠ 9 :=
sorry

end NUMINAMATH_CALUDE_impossible_sum_nine_l363_36300


namespace NUMINAMATH_CALUDE_brick_wall_pattern_l363_36302

/-- Represents a brick wall with a given number of rows and bricks -/
structure BrickWall where
  rows : ℕ
  total_bricks : ℕ
  bottom_row_bricks : ℕ

/-- Calculates the number of bricks in a given row -/
def bricks_in_row (wall : BrickWall) (row : ℕ) : ℕ :=
  wall.bottom_row_bricks - (row - 1)

theorem brick_wall_pattern (wall : BrickWall) 
  (h1 : wall.rows = 5)
  (h2 : wall.total_bricks = 50)
  (h3 : wall.bottom_row_bricks = 8) :
  ∀ row : ℕ, 1 < row → row ≤ wall.rows → 
    bricks_in_row wall row = bricks_in_row wall (row - 1) - 1 :=
by sorry

end NUMINAMATH_CALUDE_brick_wall_pattern_l363_36302


namespace NUMINAMATH_CALUDE_fraction_equality_l363_36364

theorem fraction_equality (x y : ℚ) :
  (2/5)^2 + (1/7)^2 = 25*x * ((1/3)^2 + (1/8)^2) / (73*y) →
  Real.sqrt x / Real.sqrt y = 356 / 175 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l363_36364


namespace NUMINAMATH_CALUDE_cyclic_ratio_sum_geq_two_l363_36342

theorem cyclic_ratio_sum_geq_two (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_ratio_sum_geq_two_l363_36342


namespace NUMINAMATH_CALUDE_merchant_problem_l363_36372

theorem merchant_problem (n : ℕ) : 
  (100 * n^2 : ℕ) / 100 * (2 * n) = 2662 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_merchant_problem_l363_36372


namespace NUMINAMATH_CALUDE_pentagon_x_coordinate_l363_36395

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculate the area of a pentagon -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Check if a pentagon has a horizontal line of symmetry -/
def hasHorizontalSymmetry (p : Pentagon) : Prop := sorry

theorem pentagon_x_coordinate :
  ∀ (p : Pentagon) (xc : ℝ),
    p.A = (0, 0) →
    p.B = (0, 6) →
    p.C = (xc, 12) →
    p.D = (6, 6) →
    p.E = (6, 0) →
    hasHorizontalSymmetry p →
    pentagonArea p = 60 →
    xc = 8 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_x_coordinate_l363_36395


namespace NUMINAMATH_CALUDE_max_sum_cubes_l363_36340

theorem max_sum_cubes (x y z w : ℝ) (h : x^2 + y^2 + z^2 + w^2 = 16) :
  ∃ (M : ℝ), (∀ a b c d : ℝ, a^2 + b^2 + c^2 + d^2 = 16 → a^3 + b^3 + c^3 + d^3 ≤ M) ∧
             (∃ p q r s : ℝ, p^2 + q^2 + r^2 + s^2 = 16 ∧ p^3 + q^3 + r^3 + s^3 = M) ∧
             M = 64 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l363_36340


namespace NUMINAMATH_CALUDE_max_value_theorem_l363_36346

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if all given digits are distinct -/
def distinct (x y z w : Digit) : Prop :=
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w

/-- Converts a four-digit number to its integer representation -/
def toInt (a b c d : Digit) : Nat :=
  1000 * a.val + 100 * b.val + 10 * c.val + d.val

/-- Main theorem -/
theorem max_value_theorem (x y z w v_1 v_2 v_3 v_4 : Digit) :
  distinct x y z w →
  (x.val * y.val * z.val + w.val = toInt v_1 v_2 v_3 v_4) →
  ∀ (a b c d : Digit), distinct a b c d →
    (a.val * b.val * c.val + d.val ≤ toInt v_1 v_2 v_3 v_4) →
  toInt v_1 v_2 v_3 v_4 = 9898 ∧ w.val = 98 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l363_36346


namespace NUMINAMATH_CALUDE_rectangle_area_l363_36376

/-- The area of a rectangular region bounded by y = a, y = a-2b, x = -2c, and x = d -/
theorem rectangle_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a - (a - 2 * b)) * (d - (-2 * c)) = 2 * b * d + 4 * b * c := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l363_36376


namespace NUMINAMATH_CALUDE_eva_math_score_difference_l363_36352

/-- Represents Eva's scores in a semester -/
structure SemesterScores where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Calculates the total score for a semester -/
def totalScore (scores : SemesterScores) : ℕ :=
  scores.maths + scores.arts + scores.science

/-- Represents Eva's scores for the year -/
structure YearScores where
  first : SemesterScores
  second : SemesterScores

/-- The problem statement -/
theorem eva_math_score_difference 
  (year : YearScores)
  (h1 : year.second.maths = 80)
  (h2 : year.second.arts = 90)
  (h3 : year.second.science = 90)
  (h4 : year.first.arts = year.second.arts - 15)
  (h5 : year.first.science = year.second.science - year.second.science / 3)
  (h6 : totalScore year.first + totalScore year.second = 485)
  : year.first.maths = year.second.maths + 10 := by
  sorry

end NUMINAMATH_CALUDE_eva_math_score_difference_l363_36352


namespace NUMINAMATH_CALUDE_xy_minus_x_equals_nine_l363_36374

theorem xy_minus_x_equals_nine (x y : ℝ) (hx : x = 3) (hy : y = 4) : x * y - x = 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_minus_x_equals_nine_l363_36374


namespace NUMINAMATH_CALUDE_freight_train_speed_l363_36384

/-- Proves that the speed of the freight train is 50 km/hr given the problem conditions --/
theorem freight_train_speed 
  (distance : ℝ) 
  (speed_difference : ℝ) 
  (express_speed : ℝ) 
  (time : ℝ) 
  (h1 : distance = 390) 
  (h2 : speed_difference = 30) 
  (h3 : express_speed = 80) 
  (h4 : time = 3) 
  (h5 : distance = (express_speed * time) + ((express_speed - speed_difference) * time)) : 
  express_speed - speed_difference = 50 := by
  sorry

end NUMINAMATH_CALUDE_freight_train_speed_l363_36384


namespace NUMINAMATH_CALUDE_prove_weekly_pay_l363_36313

def weekly_pay_problem (y_pay : ℝ) (x_percent : ℝ) : Prop :=
  let x_pay := x_percent * y_pay
  let total_pay := x_pay + y_pay
  y_pay = 263.64 ∧ x_percent = 1.2 → total_pay = 580.008

theorem prove_weekly_pay : weekly_pay_problem 263.64 1.2 := by
  sorry

end NUMINAMATH_CALUDE_prove_weekly_pay_l363_36313


namespace NUMINAMATH_CALUDE_tan_22_5_deg_over_one_minus_tan_squared_l363_36336

theorem tan_22_5_deg_over_one_minus_tan_squared (
  angle_22_5 : ℝ)
  (h1 : 45 * Real.pi / 180 = 2 * angle_22_5)
  (h2 : Real.tan (45 * Real.pi / 180) = 1)
  (h3 : ∀ θ : ℝ, Real.tan (2 * θ) = (2 * Real.tan θ) / (1 - Real.tan θ ^ 2)) :
  Real.tan angle_22_5 / (1 - Real.tan angle_22_5 ^ 2) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_over_one_minus_tan_squared_l363_36336


namespace NUMINAMATH_CALUDE_john_jury_duty_days_l363_36360

def jury_duty_days (jury_selection_days : ℕ) 
                   (trial_duration_multiplier : ℕ) 
                   (trial_extra_hours_per_day : ℕ) 
                   (deliberation_equivalent_full_days : ℕ) 
                   (deliberation_hours_per_day : ℕ) : ℕ :=
  let trial_days := jury_selection_days * trial_duration_multiplier
  let trial_extra_days := (trial_days * trial_extra_hours_per_day) / 24
  let deliberation_days := 
    (deliberation_equivalent_full_days * 24 + deliberation_hours_per_day - 1) / deliberation_hours_per_day
  jury_selection_days + trial_days + trial_extra_days + deliberation_days

theorem john_jury_duty_days : 
  jury_duty_days 2 4 3 6 14 = 22 := by sorry

end NUMINAMATH_CALUDE_john_jury_duty_days_l363_36360


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l363_36391

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 12) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 12 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l363_36391


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l363_36399

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle defined by its vertices -/
structure Rectangle where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Represents a circle defined by its center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculates the area of intersection between a rectangle and a circle -/
def intersectionArea (r : Rectangle) (c : Circle) : ℝ := sorry

/-- The main theorem stating the area of intersection -/
theorem intersection_area_theorem (r : Rectangle) (c : Circle) : 
  r.v1 = ⟨3, 9⟩ → 
  r.v2 = ⟨20, 9⟩ → 
  r.v3 = ⟨20, -6⟩ → 
  r.v4 = ⟨3, -6⟩ → 
  c.center = ⟨3, -6⟩ → 
  c.radius = 5 → 
  intersectionArea r c = 25 * Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_intersection_area_theorem_l363_36399


namespace NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l363_36304

theorem equidistant_point_on_y_axis : 
  ∃ y : ℝ, 
    ((-3 : ℝ) - 0)^2 + (0 - y)^2 = ((-2 : ℝ) - 0)^2 + (5 - y)^2 ∧ 
    y = 2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l363_36304


namespace NUMINAMATH_CALUDE_mod_eleven_problem_l363_36303

theorem mod_eleven_problem : ∃ n : ℕ, 0 ≤ n ∧ n < 11 ∧ 1234 % 11 = n := by
  sorry

end NUMINAMATH_CALUDE_mod_eleven_problem_l363_36303


namespace NUMINAMATH_CALUDE_shift_increasing_interval_l363_36359

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem shift_increasing_interval
  (h : IncreasingOn f (-2) 3) :
  IncreasingOn (fun x ↦ f (x + 5)) (-7) (-2) :=
sorry

end NUMINAMATH_CALUDE_shift_increasing_interval_l363_36359


namespace NUMINAMATH_CALUDE_selection_plans_count_l363_36309

def number_of_people : ℕ := 6
def number_of_cities : ℕ := 4
def number_to_select : ℕ := 4
def restricted_people : ℕ := 2
def restricted_city : ℕ := 1

theorem selection_plans_count :
  (number_of_people * (number_of_people - 1) * (number_of_people - 2) * (number_of_people - 3)) -
  (restricted_people * ((number_of_people - 1) * (number_of_people - 2) * (number_of_people - 3))) = 240 := by
  sorry

end NUMINAMATH_CALUDE_selection_plans_count_l363_36309


namespace NUMINAMATH_CALUDE_calculator_result_l363_36357

def special_key (x : ℚ) : ℚ := 1 / (1 - x)

theorem calculator_result :
  (Nat.iterate special_key 100 5 : ℚ) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_calculator_result_l363_36357


namespace NUMINAMATH_CALUDE_quadratic_points_range_l363_36368

/-- Given a quadratic function f(x) = -x^2 - 2x + 3, prove that if (a, m) and (a+2, n) are points on the graph of f, and m ≥ n, then a ≥ -2. -/
theorem quadratic_points_range (a m n : ℝ) : 
  (m = -a^2 - 2*a + 3) → 
  (n = -(a+2)^2 - 2*(a+2) + 3) → 
  (m ≥ n) → 
  (a ≥ -2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_points_range_l363_36368


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l363_36370

/-- A triangle with perimeter 60, two equal sides, and a difference of 21 between two sides has side lengths 27, 27, and 6. -/
theorem triangle_side_lengths :
  ∀ a b : ℝ,
  a > 0 ∧ b > 0 ∧
  2 * a + b = 60 ∧
  a - b = 21 →
  a = 27 ∧ b = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l363_36370


namespace NUMINAMATH_CALUDE_fourth_grade_students_l363_36343

theorem fourth_grade_students (initial : ℕ) (left : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 8 → left = 5 → new = 8 → final = initial - left + new → final = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l363_36343


namespace NUMINAMATH_CALUDE_no_common_root_for_quadratics_l363_36315

/-- Two quadratic polynomials with coefficients satisfying certain inequalities cannot have a common root -/
theorem no_common_root_for_quadratics (k m n l : ℝ) 
  (h1 : k > m) (h2 : m > n) (h3 : n > l) (h4 : l > 0) :
  ¬∃ x : ℝ, x^2 + m*x + n = 0 ∧ x^2 + k*x + l = 0 := by
  sorry


end NUMINAMATH_CALUDE_no_common_root_for_quadratics_l363_36315


namespace NUMINAMATH_CALUDE_smallest_taco_packages_l363_36383

/-- The number of tacos in each package -/
def tacos_per_package : ℕ := 4

/-- The number of taco shells in each package -/
def shells_per_package : ℕ := 6

/-- The minimum number of tacos and taco shells required -/
def min_required : ℕ := 60

/-- Proposition: The smallest number of taco packages to buy is 15 -/
theorem smallest_taco_packages : 
  (∃ (taco_packages shell_packages : ℕ),
    taco_packages * tacos_per_package = shell_packages * shells_per_package ∧
    taco_packages * tacos_per_package ≥ min_required ∧
    shell_packages * shells_per_package ≥ min_required ∧
    ∀ (t s : ℕ), 
      t * tacos_per_package = s * shells_per_package →
      t * tacos_per_package ≥ min_required →
      s * shells_per_package ≥ min_required →
      t ≥ taco_packages) →
  (∃ (shell_packages : ℕ),
    15 * tacos_per_package = shell_packages * shells_per_package ∧
    15 * tacos_per_package ≥ min_required ∧
    shell_packages * shells_per_package ≥ min_required) :=
by sorry

end NUMINAMATH_CALUDE_smallest_taco_packages_l363_36383


namespace NUMINAMATH_CALUDE_bucket_water_problem_l363_36306

/-- Given two equations representing the weight of a bucket with water,
    prove that the original amount of water is 3 kg and the bucket weighs 4 kg. -/
theorem bucket_water_problem (x y : ℝ) 
  (eq1 : 4 * x + y = 16)
  (eq2 : 6 * x + y = 22) :
  x = 3 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_bucket_water_problem_l363_36306


namespace NUMINAMATH_CALUDE_initial_pigeons_l363_36398

theorem initial_pigeons (initial final joined : ℕ) : 
  initial > 0 → 
  joined = 1 → 
  final = initial + joined → 
  final = 2 → 
  initial = 1 := by
sorry

end NUMINAMATH_CALUDE_initial_pigeons_l363_36398


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l363_36335

theorem set_equality_implies_sum_of_powers (a b : ℝ) : 
  ({a, 1, b/a} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → a^2016 + b^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l363_36335


namespace NUMINAMATH_CALUDE_probability_not_snowing_l363_36349

theorem probability_not_snowing (p_snow : ℚ) (h : p_snow = 2/5) : 1 - p_snow = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_snowing_l363_36349


namespace NUMINAMATH_CALUDE_initial_puppies_count_l363_36365

/-- The number of puppies Alyssa gave to her friends -/
def puppies_given : ℕ := 7

/-- The number of puppies Alyssa has now -/
def puppies_remaining : ℕ := 5

/-- The initial number of puppies Alyssa had -/
def initial_puppies : ℕ := puppies_given + puppies_remaining

theorem initial_puppies_count : initial_puppies = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l363_36365


namespace NUMINAMATH_CALUDE_combined_return_percentage_l363_36354

def investment1 : ℝ := 500
def investment2 : ℝ := 1500
def return1 : ℝ := 0.07
def return2 : ℝ := 0.23

def total_investment : ℝ := investment1 + investment2
def total_return : ℝ := investment1 * return1 + investment2 * return2

theorem combined_return_percentage :
  (total_return / total_investment) * 100 = 19 := by sorry

end NUMINAMATH_CALUDE_combined_return_percentage_l363_36354


namespace NUMINAMATH_CALUDE_ferry_river_crossing_l363_36345

/-- Two ferries crossing a river problem -/
theorem ferry_river_crossing (W : ℝ) : 
  W > 0 → -- Width of the river is positive
  (∃ (d₁ d₂ : ℝ), 
    d₁ = 700 ∧ -- First meeting point is 700 feet from one shore
    d₁ + d₂ = W ∧ -- Sum of distances at first meeting equals river width
    W + 400 + (W + (W - 400)) = 3 * W ∧ -- Total distance at second meeting
    2 * (W + 700) = 3 * W) → -- Relationship between meetings and river width
  W = 1400 := by
sorry

end NUMINAMATH_CALUDE_ferry_river_crossing_l363_36345


namespace NUMINAMATH_CALUDE_smallest_number_problem_l363_36382

theorem smallest_number_problem (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a ≤ b ∧ b ≤ c →
  b = 29 →
  c = b + 7 →
  (a + b + c) / 3 = 30 →
  a = 25 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_problem_l363_36382


namespace NUMINAMATH_CALUDE_fixed_point_exponential_l363_36390

/-- The function f(x) = a^(x+1) - 2 passes through the point (-1, -1) for all a > 0 and a ≠ 1 -/
theorem fixed_point_exponential (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) - 2
  f (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_l363_36390


namespace NUMINAMATH_CALUDE_only_j_has_inverse_l363_36318

-- Define the types for our functions
def Function : Type := ℝ → ℝ

-- Define properties for each function
def is_parabola_upward (f : Function) : Prop := sorry

def is_discontinuous_two_segments (f : Function) : Prop := sorry

def is_horizontal_line (f : Function) : Prop := sorry

def is_sine_function (f : Function) : Prop := sorry

def is_linear_positive_slope (f : Function) : Prop := sorry

-- Define what it means for a function to have an inverse
def has_inverse (f : Function) : Prop := sorry

-- State the theorem
theorem only_j_has_inverse 
  (F G H I J : Function)
  (hF : is_parabola_upward F)
  (hG : is_discontinuous_two_segments G)
  (hH : is_horizontal_line H)
  (hI : is_sine_function I)
  (hJ : is_linear_positive_slope J) :
  (¬ has_inverse F) ∧ 
  (¬ has_inverse G) ∧ 
  (¬ has_inverse H) ∧ 
  (¬ has_inverse I) ∧ 
  has_inverse J :=
sorry

end NUMINAMATH_CALUDE_only_j_has_inverse_l363_36318


namespace NUMINAMATH_CALUDE_m_range_l363_36369

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → m * x^3 - x^2 + 4*x + 3 ≥ 0) → 
  m ∈ Set.Icc (-6) (-2) := by
sorry

end NUMINAMATH_CALUDE_m_range_l363_36369


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l363_36385

theorem sqrt_of_sqrt_81 : ∃ (x : ℝ), x^2 = 81 ∧ (x = 3 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l363_36385


namespace NUMINAMATH_CALUDE_largest_non_attainable_sum_l363_36348

/-- The set of coin denominations in Limonia -/
def coin_denominations (n : ℕ) : Set ℕ :=
  {3*n - 1, 6*n + 1, 6*n + 4, 6*n + 7}

/-- A sum is attainable if it can be formed using the given coin denominations -/
def is_attainable (n : ℕ) (sum : ℕ) : Prop :=
  ∃ (a b c d : ℕ), sum = a*(3*n - 1) + b*(6*n + 1) + c*(6*n + 4) + d*(6*n + 7)

/-- The largest non-attainable sum in Limonia -/
def largest_non_attainable (n : ℕ) : ℕ := 6*n^2 + 4*n - 5

/-- Theorem: The largest non-attainable sum in Limonia is 6n^2 + 4n - 5 -/
theorem largest_non_attainable_sum (n : ℕ) :
  (∀ k > largest_non_attainable n, is_attainable n k) ∧
  ¬(is_attainable n (largest_non_attainable n)) := by
  sorry

end NUMINAMATH_CALUDE_largest_non_attainable_sum_l363_36348


namespace NUMINAMATH_CALUDE_johns_weight_bench_safety_percentage_l363_36326

/-- Proves that the percentage under the maximum weight that John wants to stay is 20% -/
theorem johns_weight_bench_safety_percentage 
  (bench_max_capacity : ℝ) 
  (johns_weight : ℝ) 
  (bar_weight : ℝ) 
  (h1 : bench_max_capacity = 1000) 
  (h2 : johns_weight = 250) 
  (h3 : bar_weight = 550) : 
  100 - (johns_weight + bar_weight) / bench_max_capacity * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_johns_weight_bench_safety_percentage_l363_36326


namespace NUMINAMATH_CALUDE_honey_harvest_calculation_l363_36322

/-- The amount of honey harvested last year -/
def last_year_harvest : ℕ := 8564 - 6085

/-- The increase in honey harvest this year -/
def harvest_increase : ℕ := 6085

/-- The total amount of honey harvested this year -/
def this_year_harvest : ℕ := 8564

theorem honey_harvest_calculation :
  last_year_harvest = 2479 :=
by sorry

end NUMINAMATH_CALUDE_honey_harvest_calculation_l363_36322


namespace NUMINAMATH_CALUDE_age_problem_solution_l363_36344

/-- Given three people a, b, and c, with their ages represented as natural numbers. -/
def age_problem (a b c : ℕ) : Prop :=
  -- The average age of a, b, and c is 27 years
  (a + b + c) / 3 = 27 ∧
  -- The average age of a and c is 29 years
  (a + c) / 2 = 29 →
  -- The age of b is 23 years
  b = 23

/-- Theorem stating that under the given conditions, b's age is 23 years -/
theorem age_problem_solution :
  ∀ a b c : ℕ, age_problem a b c :=
by
  sorry

end NUMINAMATH_CALUDE_age_problem_solution_l363_36344


namespace NUMINAMATH_CALUDE_no_solution_factorial_equation_l363_36389

theorem no_solution_factorial_equation :
  ∀ (k m : ℕ+), k.val.factorial + 48 ≠ 48 * (k.val + 1) ^ m.val := by
  sorry

end NUMINAMATH_CALUDE_no_solution_factorial_equation_l363_36389


namespace NUMINAMATH_CALUDE_smallest_consecutive_even_sum_l363_36337

theorem smallest_consecutive_even_sum (a : ℤ) : 
  (∃ (b c d e : ℤ), 
    (a + 2 = b) ∧ (b + 2 = c) ∧ (c + 2 = d) ∧ (d + 2 = e) ∧  -- Consecutive even integers
    (a % 2 = 0) ∧                                            -- First number is even
    (a + b + c + d + e = 380)) →                             -- Sum is 380
  a = 72 := by
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_even_sum_l363_36337


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l363_36380

theorem fraction_equality_sum (C D : ℝ) :
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ 5 →
    (D * x - 17) / (x^2 - 8*x + 15) = C / (x - 3) + 5 / (x - 5)) →
  C + D = 29/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l363_36380


namespace NUMINAMATH_CALUDE_unit_digit_7_power_6_cubed_l363_36394

def unit_digit (n : ℕ) : ℕ := n % 10

def power_7_cycle : List ℕ := [7, 9, 3, 1]

theorem unit_digit_7_power_6_cubed :
  unit_digit (7^(6^3)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_unit_digit_7_power_6_cubed_l363_36394


namespace NUMINAMATH_CALUDE_a_can_be_any_real_l363_36333

theorem a_can_be_any_real (a b c d : ℝ) 
  (h1 : (a / b) ^ 2 < (c / d) ^ 2)
  (h2 : b ≠ 0)
  (h3 : d ≠ 0)
  (h4 : c = -d) :
  ∃ (x : ℝ), x = a ∧ (x < 0 ∨ x = 0 ∨ x > 0) :=
by sorry

end NUMINAMATH_CALUDE_a_can_be_any_real_l363_36333


namespace NUMINAMATH_CALUDE_inverse_f_75_l363_36307

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 6

-- State the theorem
theorem inverse_f_75 : f⁻¹ 75 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_f_75_l363_36307


namespace NUMINAMATH_CALUDE_collinear_opposite_vectors_l363_36332

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

/-- Two vectors have opposite directions if their scalar multiple is negative -/
def opposite_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ a = k • b

theorem collinear_opposite_vectors (m : ℝ) :
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (1, m)
  collinear a b ∧ opposite_directions a b → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_collinear_opposite_vectors_l363_36332


namespace NUMINAMATH_CALUDE_area_increase_bound_l363_36361

/-- A convex polygon in a 2D plane -/
structure ConvexPolygon where
  -- Add necessary fields here
  isConvex : Bool

/-- The perimeter of a polygon -/
def perimeter (p : ConvexPolygon) : ℝ := sorry

/-- The area of a polygon -/
def area (p : ConvexPolygon) : ℝ := sorry

/-- The polygon resulting from moving all sides of p outward by distance h -/
def expandedPolygon (p : ConvexPolygon) (h : ℝ) : ConvexPolygon := sorry

theorem area_increase_bound (p : ConvexPolygon) (h : ℝ) (h_pos : h > 0) :
  area (expandedPolygon p h) - area p > perimeter p * h + π * h^2 := by
  sorry

end NUMINAMATH_CALUDE_area_increase_bound_l363_36361


namespace NUMINAMATH_CALUDE_alpha_more_cost_effective_regular_l363_36366

/-- Represents a fitness club with a monthly fee -/
structure FitnessClub where
  name : String
  monthlyFee : ℕ

/-- Calculates the yearly cost for a fitness club -/
def yearlyCost (club : FitnessClub) : ℕ :=
  club.monthlyFee * 12

/-- Calculates the cost per visit for a given number of visits -/
def costPerVisit (club : FitnessClub) (visits : ℕ) : ℚ :=
  (yearlyCost club : ℚ) / visits

/-- Represents the two attendance scenarios -/
inductive AttendancePattern
  | Regular
  | Sporadic

/-- Calculates the number of visits per year based on the attendance pattern -/
def visitsPerYear (pattern : AttendancePattern) : ℕ :=
  match pattern with
  | .Regular => 96
  | .Sporadic => 48

/-- The main theorem stating that Alpha is more cost-effective for regular attendance -/
theorem alpha_more_cost_effective_regular :
  let alpha : FitnessClub := ⟨"Alpha", 999⟩
  let beta : FitnessClub := ⟨"Beta", 1299⟩
  let regularVisits := visitsPerYear AttendancePattern.Regular
  costPerVisit alpha regularVisits < costPerVisit beta regularVisits :=
by sorry

end NUMINAMATH_CALUDE_alpha_more_cost_effective_regular_l363_36366


namespace NUMINAMATH_CALUDE_smallest_k_for_square_l363_36388

theorem smallest_k_for_square : ∃ (m : ℕ), 
  2016 * 2017 * 2018 * 2019 + 1 = m^2 ∧ 
  ∀ (k : ℕ), k < 1 → ¬∃ (n : ℕ), 2016 * 2017 * 2018 * 2019 + k = n^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_square_l363_36388


namespace NUMINAMATH_CALUDE_james_painting_fraction_l363_36393

/-- If a person can paint a wall in a given time, this function calculates
    the fraction of the wall they can paint in a shorter time period. -/
def fractionPainted (totalTime minutes : ℚ) : ℚ :=
  minutes / totalTime

theorem james_painting_fraction :
  fractionPainted 60 15 = 1/4 := by sorry

end NUMINAMATH_CALUDE_james_painting_fraction_l363_36393


namespace NUMINAMATH_CALUDE_angle_sum_equality_counterexample_l363_36347

theorem angle_sum_equality_counterexample :
  ∃ (angle1 angle2 : ℝ), 
    angle1 + angle2 = 90 ∧ angle1 = angle2 :=
by sorry

end NUMINAMATH_CALUDE_angle_sum_equality_counterexample_l363_36347


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l363_36323

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 5 * a 11 = 3 →
  a 3 + a 13 = 4 →
  a 15 / a 5 = 1/3 ∨ a 15 / a 5 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l363_36323


namespace NUMINAMATH_CALUDE_lottery_ratio_l363_36367

def lottery_problem (lottery_winnings : ℕ) (savings : ℕ) (fun_money : ℕ) : Prop :=
  let taxes := lottery_winnings / 2
  let after_taxes := lottery_winnings - taxes
  let investment := savings / 5
  let student_loans := after_taxes - (savings + investment + fun_money)
  (lottery_winnings = 12006 ∧ savings = 1000 ∧ fun_money = 2802) →
  (student_loans : ℚ) / after_taxes = 1 / 3

theorem lottery_ratio : 
  lottery_problem 12006 1000 2802 :=
sorry

end NUMINAMATH_CALUDE_lottery_ratio_l363_36367


namespace NUMINAMATH_CALUDE_dog_human_age_difference_l363_36301

/-- The ratio of dog years to human years -/
def dogYearRatio : ℕ := 7

/-- Calculates the age difference in dog years between a dog and a human of the same age in human years -/
def ageDifferenceInDogYears (humanAge : ℕ) : ℕ :=
  humanAge * dogYearRatio - humanAge

/-- Theorem stating that for a 3-year-old human and their 3-year-old dog, 
    the dog will be 18 years older in dog years -/
theorem dog_human_age_difference : ageDifferenceInDogYears 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_dog_human_age_difference_l363_36301


namespace NUMINAMATH_CALUDE_distance_covered_l363_36378

/-- Proves that the distance covered is 100 km given the conditions of the problem -/
theorem distance_covered (usual_speed usual_time increased_speed : ℝ) : 
  usual_speed = 20 →
  increased_speed = 25 →
  usual_speed * usual_time = increased_speed * (usual_time - 1) →
  usual_speed * usual_time = 100 := by
  sorry

#check distance_covered

end NUMINAMATH_CALUDE_distance_covered_l363_36378


namespace NUMINAMATH_CALUDE_probability_ratio_l363_36353

def num_balls : ℕ := 25
def num_bins : ℕ := 6

def probability_config_1 : ℚ :=
  (Nat.choose num_bins 2 * (Nat.factorial num_balls / (Nat.factorial 3 * Nat.factorial 3 * (Nat.factorial 5)^4))) /
  (num_bins^num_balls : ℚ)

def probability_config_2 : ℚ :=
  (Nat.choose num_bins 2 * Nat.choose (num_bins - 2) 2 * (Nat.factorial num_balls / (Nat.factorial 3 * Nat.factorial 3 * (Nat.factorial 4)^2 * (Nat.factorial 5)^2))) /
  (num_bins^num_balls : ℚ)

theorem probability_ratio :
  probability_config_1 / probability_config_2 = 625 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l363_36353


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l363_36375

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8/17)) = 15/17 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l363_36375


namespace NUMINAMATH_CALUDE_count_valid_pairs_l363_36397

def harmonic_mean (x y : ℕ+) : ℚ := 2 * (x * y) / (x + y)

def valid_pair (x y : ℕ+) : Prop :=
  x < y ∧ harmonic_mean x y = 1024

theorem count_valid_pairs : 
  ∃ (S : Finset (ℕ+ × ℕ+)), (∀ p ∈ S, valid_pair p.1 p.2) ∧ S.card = 9 ∧ 
  (∀ x y : ℕ+, valid_pair x y → (x, y) ∈ S) :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l363_36397


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l363_36363

theorem triangle_angle_sum (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 90) (h3 : b = 58) : c = 32 := by
  sorry

#check triangle_angle_sum

end NUMINAMATH_CALUDE_triangle_angle_sum_l363_36363


namespace NUMINAMATH_CALUDE_bird_migration_distance_l363_36362

/-- Calculates the total distance traveled by migrating birds -/
theorem bird_migration_distance 
  (num_birds : ℕ) 
  (distance_jim_disney : ℝ) 
  (distance_disney_london : ℝ) : 
  num_birds = 20 → 
  distance_jim_disney = 50 → 
  distance_disney_london = 60 → 
  (num_birds : ℝ) * (distance_jim_disney + distance_disney_london) = 2200 := by
  sorry

end NUMINAMATH_CALUDE_bird_migration_distance_l363_36362


namespace NUMINAMATH_CALUDE_optimal_seating_arrangement_l363_36329

/-- Represents the seating arrangement for children based on their heights. -/
structure SeatingArrangement where
  x : ℕ  -- Number of seats with three children
  y : ℕ  -- Number of seats with two children
  total_seats : ℕ
  group_a : ℕ  -- Children below 4 feet
  group_b : ℕ  -- Children between 4 and 4.5 feet
  group_c : ℕ  -- Children above 4.5 feet

/-- The seating arrangement satisfies all constraints. -/
def valid_arrangement (s : SeatingArrangement) : Prop :=
  s.x + s.y = s.total_seats ∧
  s.x ≤ s.group_a ∧
  2 * s.x + s.y ≤ s.group_b ∧
  s.y ≤ s.group_c

/-- The optimal seating arrangement exists and is unique. -/
theorem optimal_seating_arrangement :
  ∃! s : SeatingArrangement,
    s.total_seats = 7 ∧
    s.group_a = 5 ∧
    s.group_b = 8 ∧
    s.group_c = 6 ∧
    valid_arrangement s ∧
    s.x = 1 ∧
    s.y = 6 := by
  sorry


end NUMINAMATH_CALUDE_optimal_seating_arrangement_l363_36329


namespace NUMINAMATH_CALUDE_max_points_at_least_sqrt2_max_points_greater_sqrt2_l363_36381

-- Define a point on a unit sphere
def PointOnUnitSphere := ℝ × ℝ × ℝ

-- Distance function between two points on a unit sphere
def sphereDistance (p q : PointOnUnitSphere) : ℝ := sorry

-- Theorem for part a
theorem max_points_at_least_sqrt2 :
  ∀ (n : ℕ) (points : Fin n → PointOnUnitSphere),
    (∀ (i j : Fin n), i ≠ j → sphereDistance (points i) (points j) ≥ Real.sqrt 2) →
    n ≤ 6 :=
sorry

-- Theorem for part b
theorem max_points_greater_sqrt2 :
  ∀ (n : ℕ) (points : Fin n → PointOnUnitSphere),
    (∀ (i j : Fin n), i ≠ j → sphereDistance (points i) (points j) > Real.sqrt 2) →
    n ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_points_at_least_sqrt2_max_points_greater_sqrt2_l363_36381


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l363_36325

theorem unique_solution_for_exponential_equation :
  ∀ (a b n p : ℕ), 
    p.Prime → 
    2^a + p^b = n^(p-1) → 
    (a = 0 ∧ b = 1 ∧ n = 2 ∧ p = 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l363_36325


namespace NUMINAMATH_CALUDE_third_rectangle_area_l363_36338

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: Given three rectangles forming a larger rectangle without gaps or overlaps,
    where two rectangles have dimensions 3 cm × 8 cm and 2 cm × 5 cm,
    the area of the third rectangle must be 4 cm². -/
theorem third_rectangle_area :
  ∀ (r1 r2 r3 : Rectangle),
  r1.length = 3 ∧ r1.width = 8 ∧
  r2.length = 2 ∧ r2.width = 5 →
  r1.area + r2.area + r3.area = (r1.area + r2.area) →
  r3.area = 4 := by
  sorry

#check third_rectangle_area

end NUMINAMATH_CALUDE_third_rectangle_area_l363_36338


namespace NUMINAMATH_CALUDE_max_value_of_f_l363_36321

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 4 ∧ 
  (∀ x, x ∈ Set.Icc 0 4 → f x ≤ f c) ∧
  f c = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l363_36321

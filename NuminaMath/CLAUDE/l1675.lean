import Mathlib

namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1675_167522

/-- The maximum marks for an exam -/
def maximum_marks : ℝ := sorry

/-- The passing mark as a percentage of the maximum marks -/
def passing_percentage : ℝ := 0.45

/-- The marks obtained by the student -/
def student_marks : ℝ := 150

/-- The number of marks by which the student failed -/
def failing_margin : ℝ := 30

theorem exam_maximum_marks : 
  (passing_percentage * maximum_marks = student_marks + failing_margin) → 
  maximum_marks = 400 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1675_167522


namespace NUMINAMATH_CALUDE_calculate_expression_l1675_167517

theorem calculate_expression : 
  Real.sqrt 5 * (Real.sqrt 10 + 2) - 1 / (Real.sqrt 5 - 2) - Real.sqrt (1/2) = 
  (9 * Real.sqrt 2) / 2 + Real.sqrt 5 - 2 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_l1675_167517


namespace NUMINAMATH_CALUDE_salmon_sales_ratio_l1675_167536

/-- Given the first week's salmon sales and the total sales over two weeks,
    prove that the ratio of the second week's sales to the first week's sales is 3:1 -/
theorem salmon_sales_ratio (first_week : ℝ) (total : ℝ) :
  first_week = 50 →
  total = 200 →
  (total - first_week) / first_week = 3 := by
sorry

end NUMINAMATH_CALUDE_salmon_sales_ratio_l1675_167536


namespace NUMINAMATH_CALUDE_new_room_size_l1675_167558

/-- Given a bedroom and bathroom size, calculate the size of a new room that is twice as large as both combined -/
theorem new_room_size (bedroom : ℝ) (bathroom : ℝ) (new_room : ℝ) : 
  bedroom = 309 → bathroom = 150 → new_room = 2 * (bedroom + bathroom) → new_room = 918 := by
  sorry

end NUMINAMATH_CALUDE_new_room_size_l1675_167558


namespace NUMINAMATH_CALUDE_solution_value_l1675_167549

theorem solution_value (a b : ℝ) : 
  (2 : ℝ) * a + (-1 : ℝ) * b = -1 → 2 * a - b + 2017 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1675_167549


namespace NUMINAMATH_CALUDE_pyramid_total_area_l1675_167595

-- Define the square side length and pyramid height
def squareSide : ℝ := 6
def pyramidHeight : ℝ := 4

-- Define the structure of our pyramid
structure Pyramid where
  base : ℝ
  height : ℝ

-- Define our specific pyramid
def ourPyramid : Pyramid :=
  { base := squareSide,
    height := pyramidHeight }

-- Theorem statement
theorem pyramid_total_area (p : Pyramid) (h : p = ourPyramid) :
  let diagonal := p.base * Real.sqrt 2
  let slantHeight := Real.sqrt (p.height^2 + (diagonal/2)^2)
  let triangleHeight := Real.sqrt (slantHeight^2 - (p.base/2)^2)
  let squareArea := p.base^2
  let triangleArea := 4 * (p.base * triangleHeight / 2)
  squareArea + triangleArea = 96 := by sorry

end NUMINAMATH_CALUDE_pyramid_total_area_l1675_167595


namespace NUMINAMATH_CALUDE_grid_50_25_toothpicks_l1675_167533

/-- Calculates the number of toothpicks needed for a grid --/
def toothpicks_in_grid (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem: A grid of 50 by 25 toothpicks requires 2575 toothpicks --/
theorem grid_50_25_toothpicks :
  toothpicks_in_grid 50 25 = 2575 := by
  sorry

end NUMINAMATH_CALUDE_grid_50_25_toothpicks_l1675_167533


namespace NUMINAMATH_CALUDE_M_union_S_eq_M_l1675_167561

-- Define set M
def M : Set ℝ := {y | ∃ x, y = Real.exp (x * Real.log 2)}

-- Define set S
def S : Set ℝ := {x | x > 1}

-- Theorem to prove
theorem M_union_S_eq_M : M ∪ S = M := by
  sorry

end NUMINAMATH_CALUDE_M_union_S_eq_M_l1675_167561


namespace NUMINAMATH_CALUDE_sin_theta_value_l1675_167516

theorem sin_theta_value (θ : Real) 
  (h1 : 6 * Real.tan θ = 5 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = (-3 + 2 * Real.sqrt 34) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l1675_167516


namespace NUMINAMATH_CALUDE_min_value_of_m_minus_n_l1675_167523

noncomputable section

def f (x : ℝ) : ℝ := Real.log x + 1

def g (x : ℝ) : ℝ := 2 * Real.exp (x - 1/2)

theorem min_value_of_m_minus_n (m n : ℝ) (h : f m = g n) :
  ∃ (k : ℝ), k = 1/2 + Real.log 2 ∧ ∀ (p q : ℝ), f p = g q → m - n ≥ k :=
sorry

end

end NUMINAMATH_CALUDE_min_value_of_m_minus_n_l1675_167523


namespace NUMINAMATH_CALUDE_sum_of_nine_terms_l1675_167506

/-- An arithmetic sequence with sum Sₙ of first n terms, where a₄ = 9 and a₆ = 11 -/
structure ArithSeq where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)
  a4_eq_9 : a 4 = 9
  a6_eq_11 : a 6 = 11

/-- The sum of the first 9 terms of the arithmetic sequence is 90 -/
theorem sum_of_nine_terms (seq : ArithSeq) : seq.S 9 = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_nine_terms_l1675_167506


namespace NUMINAMATH_CALUDE_count_nines_to_hundred_l1675_167505

/-- Count of digit 9 in a single number -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Sum of count_nines for numbers from 1 to n -/
def sum_nines (n : ℕ) : ℕ := sorry

/-- The theorem stating that the count of 9s in numbers from 1 to 100 is 19 -/
theorem count_nines_to_hundred : sum_nines 100 = 19 := by sorry

end NUMINAMATH_CALUDE_count_nines_to_hundred_l1675_167505


namespace NUMINAMATH_CALUDE_total_dividend_is_825_l1675_167524

/-- Represents the investment scenario with two types of shares --/
structure Investment where
  total_amount : ℕ
  type_a_face_value : ℕ
  type_b_face_value : ℕ
  type_a_premium : ℚ
  type_b_discount : ℚ
  type_a_dividend_rate : ℚ
  type_b_dividend_rate : ℚ

/-- Calculates the total dividend received from the investment --/
def calculate_total_dividend (inv : Investment) : ℚ :=
  sorry

/-- Theorem stating that the total dividend received is 825 --/
theorem total_dividend_is_825 :
  let inv : Investment := {
    total_amount := 14400,
    type_a_face_value := 100,
    type_b_face_value := 100,
    type_a_premium := 1/5,
    type_b_discount := 1/10,
    type_a_dividend_rate := 7/100,
    type_b_dividend_rate := 1/20
  }
  calculate_total_dividend inv = 825 := by sorry

end NUMINAMATH_CALUDE_total_dividend_is_825_l1675_167524


namespace NUMINAMATH_CALUDE_town_population_growth_l1675_167582

/-- The final population after compound growth --/
def final_population (initial_population : ℕ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_population * (1 + growth_rate) ^ years

/-- Theorem stating the approximate final population after a decade --/
theorem town_population_growth : 
  ∃ (result : ℕ), 
    344251 ≤ result ∧ 
    result ≤ 344252 ∧ 
    result = ⌊final_population 175000 0.07 10⌋ := by
  sorry

end NUMINAMATH_CALUDE_town_population_growth_l1675_167582


namespace NUMINAMATH_CALUDE_equality_from_cubic_relations_equality_from_mixed_cubic_relations_l1675_167591

theorem equality_from_cubic_relations (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * (b^3 + c^3) = b * (c^3 + a^3) ∧ b * (c^3 + a^3) = c * (a^3 + b^3)) → 
  (a = b ∧ b = c) :=
by sorry

theorem equality_from_mixed_cubic_relations (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * (a^3 + b^3) = b * (b^3 + c^3) ∧ b * (b^3 + c^3) = c * (c^3 + a^3)) → 
  (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_equality_from_cubic_relations_equality_from_mixed_cubic_relations_l1675_167591


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1675_167593

theorem geometric_sequence_common_ratio (a : ℕ → ℚ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Definition of geometric sequence
  a 1 = 64 →                             -- First term condition
  a 2 = 8 →                              -- Second term condition
  a 2 / a 1 = 1 / 8 :=                   -- Conclusion: common ratio q = 1/8
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1675_167593


namespace NUMINAMATH_CALUDE_gcd_lcm_18_24_l1675_167508

theorem gcd_lcm_18_24 :
  (Nat.gcd 18 24 = 6) ∧ (Nat.lcm 18 24 = 72) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_18_24_l1675_167508


namespace NUMINAMATH_CALUDE_product_of_solutions_l1675_167538

theorem product_of_solutions : ∃ (x y : ℝ), 
  (abs x = 3 * (abs x - 2)) ∧ 
  (abs y = 3 * (abs y - 2)) ∧ 
  (x ≠ y) ∧ 
  (x * y = -9) :=
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l1675_167538


namespace NUMINAMATH_CALUDE_monica_savings_l1675_167563

def savings_pattern (week : ℕ) : ℕ :=
  let cycle := week % 20
  if cycle < 6 then 15 + 5 * cycle
  else if cycle < 12 then 40 - 5 * (cycle - 6)
  else if cycle < 18 then 15 + 5 * (cycle - 12)
  else 40 - 5 * (cycle - 18)

def total_savings : ℕ := (List.range 100).map savings_pattern |> List.sum

theorem monica_savings :
  total_savings = 1450 := by sorry

end NUMINAMATH_CALUDE_monica_savings_l1675_167563


namespace NUMINAMATH_CALUDE_smallest_c_value_l1675_167583

theorem smallest_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x : ℝ, a * Real.cos (b * x + c) ≤ a * Real.cos (b * (-π/4) + c)) →
  c ≥ π/4 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_value_l1675_167583


namespace NUMINAMATH_CALUDE_melissa_games_played_l1675_167554

/-- Given a player's points per game and total score, calculate the number of games played -/
def games_played (points_per_game : ℕ) (total_points : ℕ) : ℕ :=
  total_points / points_per_game

/-- Theorem: A player scoring 120 points per game with a total of 1200 points played 10 games -/
theorem melissa_games_played :
  games_played 120 1200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_melissa_games_played_l1675_167554


namespace NUMINAMATH_CALUDE_abs_neg_one_fourth_l1675_167532

theorem abs_neg_one_fourth : |(-1 : ℚ) / 4| = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_fourth_l1675_167532


namespace NUMINAMATH_CALUDE_ages_sum_l1675_167573

theorem ages_sum (a b c : ℕ) : 
  a = 16 + b + c → 
  a^2 = 1632 + (b + c)^2 → 
  a + b + c = 102 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l1675_167573


namespace NUMINAMATH_CALUDE_lines_properties_l1675_167512

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x - y + 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 1 = 0

-- Theorem statement
theorem lines_properties (a : ℝ) :
  -- 1. The lines are always perpendicular
  (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ → l₂ a x₂ y₂ → (x₁ - x₂) * (y₁ - y₂) = 0) ∧
  -- 2. l₁ passes through (0,1) and l₂ passes through (-1,0)
  l₁ a 0 1 ∧ l₂ a (-1) 0 ∧
  -- 3. The maximum distance from the intersection point to the origin is √2
  (∃ x y : ℝ, l₁ a x y ∧ l₂ a x y ∧
    ∀ x' y' : ℝ, l₁ a x' y' → l₂ a x' y' → x'^2 + y'^2 ≤ 2) ∧
  (∃ a₀ x₀ y₀ : ℝ, l₁ a₀ x₀ y₀ ∧ l₂ a₀ x₀ y₀ ∧ x₀^2 + y₀^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_lines_properties_l1675_167512


namespace NUMINAMATH_CALUDE_cycle_iteration_equivalence_l1675_167551

/-- A function that represents the k-th iteration of f -/
def iterate (f : α → α) : ℕ → α → α
  | 0, x => x
  | n + 1, x => f (iterate f n x)

/-- The main theorem -/
theorem cycle_iteration_equivalence
  {α : Type*} (f : α → α) (x₀ : α) (s k : ℕ) :
  (∃ (n : ℕ), iterate f s x₀ = x₀) →  -- x₀ belongs to a cycle of length s
  (k % s = 0 ↔ iterate f k x₀ = x₀) :=
sorry

end NUMINAMATH_CALUDE_cycle_iteration_equivalence_l1675_167551


namespace NUMINAMATH_CALUDE_factorization_proof_l1675_167518

theorem factorization_proof (b : ℝ) : 65 * b^2 + 195 * b = 65 * b * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1675_167518


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l1675_167507

/-- Given a train of length 2400 meters that takes 60 seconds to pass a point,
    calculate the time required for the same train to pass a platform of length 800 meters. -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (time_to_pass_point : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 2400)
  (h2 : time_to_pass_point = 60)
  (h3 : platform_length = 800) :
  (train_length + platform_length) / (train_length / time_to_pass_point) = 80 := by
  sorry

#check train_platform_passing_time

end NUMINAMATH_CALUDE_train_platform_passing_time_l1675_167507


namespace NUMINAMATH_CALUDE_milk_production_calculation_l1675_167528

/-- Calculates the total milk production for a herd of cows over a given number of days -/
def total_milk_production (num_cows : ℕ) (milk_per_cow_per_day : ℕ) (num_days : ℕ) : ℕ :=
  num_cows * milk_per_cow_per_day * num_days

/-- Theorem stating the total milk production for 120 cows over 15 days -/
theorem milk_production_calculation :
  total_milk_production 120 1362 15 = 2451600 := by
  sorry

#eval total_milk_production 120 1362 15

end NUMINAMATH_CALUDE_milk_production_calculation_l1675_167528


namespace NUMINAMATH_CALUDE_justin_tim_games_l1675_167578

/-- The total number of players in the four-square league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- Justin and Tim are two specific players -/
def justin_and_tim : ℕ := 2

/-- The number of remaining players after Justin and Tim -/
def remaining_players : ℕ := total_players - justin_and_tim

/-- The number of additional players needed in a game with Justin and Tim -/
def additional_players : ℕ := players_per_game - justin_and_tim

theorem justin_tim_games (total_players : ℕ) (players_per_game : ℕ) (justin_and_tim : ℕ) 
  (remaining_players : ℕ) (additional_players : ℕ) :
  total_players = 12 →
  players_per_game = 6 →
  justin_and_tim = 2 →
  remaining_players = total_players - justin_and_tim →
  additional_players = players_per_game - justin_and_tim →
  Nat.choose remaining_players additional_players = 210 :=
by sorry

end NUMINAMATH_CALUDE_justin_tim_games_l1675_167578


namespace NUMINAMATH_CALUDE_triangle_area_is_25_over_3_l1675_167520

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The area of a triangle given three lines that form its sides -/
def triangleArea (l1 l2 l3 : Line) : ℝ :=
  sorry

/-- The three lines that form the triangle -/
def line1 : Line := { slope := 2, intercept := 4 }
def line2 : Line := { slope := -1, intercept := 3 }
def line3 : Line := { slope := 0, intercept := 0 }

theorem triangle_area_is_25_over_3 :
  triangleArea line1 line2 line3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_25_over_3_l1675_167520


namespace NUMINAMATH_CALUDE_average_score_is_92_l1675_167531

def brief_scores : List Int := [10, -5, 0, 8, -3]
def xiao_ming_score : Int := 90
def xiao_ming_rank : Nat := 3

def actual_scores : List Int := brief_scores.map (λ x => xiao_ming_score + x)

theorem average_score_is_92 : 
  (actual_scores.sum : ℚ) / actual_scores.length = 92 := by sorry

end NUMINAMATH_CALUDE_average_score_is_92_l1675_167531


namespace NUMINAMATH_CALUDE_largest_integer_l1675_167592

theorem largest_integer (a b c : ℤ) : 
  (2 * a + 3 * b + 4 * c = 225) →
  (a + b + c = 60) →
  (a = 15 ∨ b = 15 ∨ c = 15) →
  (max a (max b c) = 25) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_l1675_167592


namespace NUMINAMATH_CALUDE_fred_balloon_count_l1675_167544

/-- The number of blue balloons Sally has -/
def sally_balloons : ℕ := 6

/-- The factor by which Fred has more balloons than Sally -/
def fred_factor : ℕ := 3

/-- The number of blue balloons Fred has -/
def fred_balloons : ℕ := sally_balloons * fred_factor

theorem fred_balloon_count : fred_balloons = 18 := by
  sorry

end NUMINAMATH_CALUDE_fred_balloon_count_l1675_167544


namespace NUMINAMATH_CALUDE_dilation_matrix_determinant_l1675_167581

theorem dilation_matrix_determinant :
  ∀ (E : Matrix (Fin 2) (Fin 2) ℝ),
  (∀ (i j : Fin 2), E i j = if i = j then 9 else 0) →
  Matrix.det E = 81 := by
sorry

end NUMINAMATH_CALUDE_dilation_matrix_determinant_l1675_167581


namespace NUMINAMATH_CALUDE_exists_unvisited_planet_l1675_167590

/-- A type representing a planet in the solar system -/
structure Planet where
  id : ℕ

/-- A function that returns the closest planet to a given planet -/
def closest_planet (planets : Finset Planet) : Planet → Planet :=
  sorry

theorem exists_unvisited_planet (n : ℕ) (h : n ≥ 1) :
  ∀ (planets : Finset Planet),
    Finset.card planets = 2 * n + 1 →
    (∀ p q : Planet, p ∈ planets → q ∈ planets → p ≠ q → 
      closest_planet planets p ≠ closest_planet planets q) →
    ∃ p : Planet, p ∈ planets ∧ 
      ∀ q : Planet, q ∈ planets → closest_planet planets q ≠ p :=
sorry

end NUMINAMATH_CALUDE_exists_unvisited_planet_l1675_167590


namespace NUMINAMATH_CALUDE_like_terms_imply_equal_exponents_l1675_167596

-- Define what it means for two terms to be "like terms"
def are_like_terms (term1 term2 : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ), term1 x y = a * (term2 x y)

-- State the theorem
theorem like_terms_imply_equal_exponents :
  are_like_terms (fun x y => 3 * x^4 * y^m) (fun x y => -2 * x^4 * y^2) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_equal_exponents_l1675_167596


namespace NUMINAMATH_CALUDE_remainder_fraction_l1675_167580

theorem remainder_fraction (x : ℝ) (h : x = 62.5) : 
  ((x + 5) * 2 / 5 - 5) / 44 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_remainder_fraction_l1675_167580


namespace NUMINAMATH_CALUDE_toothpicks_stage_20_l1675_167546

/-- The number of toothpicks in stage n of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  4 + 3 * (n - 1)

theorem toothpicks_stage_20 :
  toothpicks 20 = 61 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_stage_20_l1675_167546


namespace NUMINAMATH_CALUDE_problem_1_l1675_167541

theorem problem_1 : 7 - (-3) + (-4) - |(-8)| = -2 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1675_167541


namespace NUMINAMATH_CALUDE_prob_odd_females_committee_l1675_167509

/-- The number of men in the pool of candidates -/
def num_men : ℕ := 5

/-- The number of women in the pool of candidates -/
def num_women : ℕ := 4

/-- The size of the committee to be formed -/
def committee_size : ℕ := 3

/-- The probability of selecting a committee with an odd number of female members -/
def prob_odd_females : ℚ := 11 / 21

/-- Theorem stating that the probability of selecting a committee of three members
    with an odd number of female members from a pool of five men and four women,
    where all candidates are equally likely to be chosen, is 11/21 -/
theorem prob_odd_females_committee :
  let total_candidates := num_men + num_women
  let total_committees := Nat.choose total_candidates committee_size
  let committees_one_female := Nat.choose num_women 1 * Nat.choose num_men 2
  let committees_three_females := Nat.choose num_women 3 * Nat.choose num_men 0
  let favorable_outcomes := committees_one_female + committees_three_females
  (favorable_outcomes : ℚ) / total_committees = prob_odd_females := by
  sorry


end NUMINAMATH_CALUDE_prob_odd_females_committee_l1675_167509


namespace NUMINAMATH_CALUDE_unicorns_total_games_l1675_167571

theorem unicorns_total_games : 
  ∀ (initial_games initial_wins district_wins district_losses : ℕ),
    initial_wins = initial_games / 2 →
    district_wins = 8 →
    district_losses = 3 →
    (initial_wins + district_wins) * 100 = 55 * (initial_games + district_wins + district_losses) →
    initial_games + district_wins + district_losses = 50 := by
  sorry

end NUMINAMATH_CALUDE_unicorns_total_games_l1675_167571


namespace NUMINAMATH_CALUDE_total_cars_sold_l1675_167521

def cars_sold_day1 : ℕ := 14
def cars_sold_day2 : ℕ := 16
def cars_sold_day3 : ℕ := 27

theorem total_cars_sold : cars_sold_day1 + cars_sold_day2 + cars_sold_day3 = 57 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_sold_l1675_167521


namespace NUMINAMATH_CALUDE_target_hit_probability_l1675_167588

theorem target_hit_probability (p_A p_B : ℝ) (h_A : p_A = 9/10) (h_B : p_B = 8/9) :
  1 - (1 - p_A) * (1 - p_B) = 89/90 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1675_167588


namespace NUMINAMATH_CALUDE_circle_and_line_theorem_l1675_167594

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the line m
def line_m (x y : ℝ) : Prop := 3*x - 2*y = 0

-- Define the line l
def line_l (k x y : ℝ) : Prop := y = k*x + 2

-- Define points A and B
def point_A : ℝ × ℝ := (1, 3)
def point_B : ℝ × ℝ := (2, 2)

-- Define the dot product of OM and ON
def dot_product (k x₁ x₂ : ℝ) : ℝ := x₁*x₂ + (k*x₁ + 2)*(k*x₂ + 2)

theorem circle_and_line_theorem :
  -- 1. Circle C passes through A and B and is bisected by line m
  (circle_C point_A.1 point_A.2 ∧ circle_C point_B.1 point_B.2) ∧
  (∀ x y, circle_C x y → line_m x y → circle_C (2*2 - x) (2*3 - y)) →
  -- 2. No k exists such that line l intersects C at M and N where OM•ON = 6
  ¬∃ k : ℝ, ∃ x₁ x₂ : ℝ,
    x₁ ≠ x₂ ∧
    circle_C x₁ (k*x₁ + 2) ∧
    circle_C x₂ (k*x₂ + 2) ∧
    dot_product k x₁ x₂ = 6 :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_theorem_l1675_167594


namespace NUMINAMATH_CALUDE_seven_people_round_table_l1675_167514

def factorial (n : ℕ) : ℕ := Nat.factorial n

def roundTableArrangements (n : ℕ) : ℕ := factorial (n - 1)

theorem seven_people_round_table :
  roundTableArrangements 7 = 720 := by
  sorry

end NUMINAMATH_CALUDE_seven_people_round_table_l1675_167514


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1675_167542

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - k*x + 9 = (x - a)^2) → (k = 6 ∨ k = -6) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1675_167542


namespace NUMINAMATH_CALUDE_combination_equality_implies_five_l1675_167572

theorem combination_equality_implies_five (n : ℕ+) : 
  Nat.choose n 2 = Nat.choose (n - 1) 2 + Nat.choose (n - 1) 3 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_implies_five_l1675_167572


namespace NUMINAMATH_CALUDE_lcm_gcf_relation_l1675_167553

theorem lcm_gcf_relation (n : ℕ) :
  n ≠ 0 ∧ Nat.lcm n 24 = 48 ∧ Nat.gcd n 24 = 8 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_relation_l1675_167553


namespace NUMINAMATH_CALUDE_hotdog_eating_record_l1675_167513

/-- The hotdog eating record problem -/
theorem hotdog_eating_record 
  (total_time : ℕ) 
  (halfway_time : ℕ) 
  (halfway_hotdogs : ℕ) 
  (required_rate : ℕ) 
  (h1 : total_time = 10) 
  (h2 : halfway_time = total_time / 2) 
  (h3 : halfway_hotdogs = 20) 
  (h4 : required_rate = 11) : 
  halfway_hotdogs + required_rate * (total_time - halfway_time) = 75 := by
sorry


end NUMINAMATH_CALUDE_hotdog_eating_record_l1675_167513


namespace NUMINAMATH_CALUDE_square_diff_over_seventy_l1675_167598

theorem square_diff_over_seventy : (535^2 - 465^2) / 70 = 1000 := by sorry

end NUMINAMATH_CALUDE_square_diff_over_seventy_l1675_167598


namespace NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_for_q_l1675_167526

-- Define the conditions
def condition_p (x : ℝ) : Prop := x^2 - 3*x + 2 < 0

def condition_q (x : ℝ) : Prop := |x - 2| < 1

-- Theorem statement
theorem condition_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, condition_p x → condition_q x) ∧
  ¬(∀ x : ℝ, condition_q x → condition_p x) :=
by sorry


end NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_for_q_l1675_167526


namespace NUMINAMATH_CALUDE_sunflower_rose_height_difference_l1675_167565

/-- The height difference between a sunflower and a rose bush -/
theorem sunflower_rose_height_difference :
  let sunflower_height : ℚ := 9 + 3/5
  let rose_height : ℚ := 5 + 4/5
  sunflower_height - rose_height = 3 + 4/5 := by sorry

end NUMINAMATH_CALUDE_sunflower_rose_height_difference_l1675_167565


namespace NUMINAMATH_CALUDE_jakes_weight_l1675_167537

theorem jakes_weight (jake_weight sister_weight : ℝ) : 
  (0.8 * jake_weight = 2 * sister_weight) →
  (jake_weight + sister_weight = 168) →
  (jake_weight = 120) := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l1675_167537


namespace NUMINAMATH_CALUDE_rihlelo_symmetry_l1675_167559

/-- Represents a design pattern -/
structure Design where
  /-- The type of object the design is for -/
  objectType : String
  /-- The country of origin for the design -/
  origin : String
  /-- The number of lines of symmetry in the design -/
  symmetryLines : ℕ

/-- The rihlèlò design from Mozambique -/
def rihlelo : Design where
  objectType := "winnowing tray"
  origin := "Mozambique"
  symmetryLines := 4

/-- Theorem stating that the rihlèlò design has 4 lines of symmetry -/
theorem rihlelo_symmetry : rihlelo.symmetryLines = 4 := by
  sorry

end NUMINAMATH_CALUDE_rihlelo_symmetry_l1675_167559


namespace NUMINAMATH_CALUDE_isosceles_base_angles_equal_l1675_167545

/-- An isosceles triangle is a triangle with two sides of equal length -/
structure IsoscelesTriangle where
  points : Fin 3 → ℝ × ℝ
  isosceles : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
    dist (points i) (points j) = dist (points i) (points k)

/-- The base angles of an isosceles triangle are the angles opposite the equal sides -/
def base_angles (t : IsoscelesTriangle) : ℝ × ℝ := sorry

/-- In an isosceles triangle, the two base angles are equal -/
theorem isosceles_base_angles_equal (t : IsoscelesTriangle) : 
  (base_angles t).1 = (base_angles t).2 := by sorry

end NUMINAMATH_CALUDE_isosceles_base_angles_equal_l1675_167545


namespace NUMINAMATH_CALUDE_gretchen_desk_work_time_l1675_167589

/-- Represents the ratio of walking time to sitting time -/
def walkingSittingRatio : ℚ := 10 / 90

/-- Represents the total walking time in minutes -/
def totalWalkingTime : ℕ := 40

/-- Represents the time spent working at the desk in hours -/
def deskWorkTime : ℚ := 6

theorem gretchen_desk_work_time :
  walkingSittingRatio * (deskWorkTime * 60) = totalWalkingTime :=
sorry

end NUMINAMATH_CALUDE_gretchen_desk_work_time_l1675_167589


namespace NUMINAMATH_CALUDE_least_denominator_for_0711_l1675_167539

theorem least_denominator_for_0711 : 
  ∃ (m : ℕ+), (711 : ℚ)/1000 ≤ m/45 ∧ m/45 < (712 : ℚ)/1000 ∧ 
  ∀ (n : ℕ+) (k : ℕ+), n < 45 → ¬((711 : ℚ)/1000 ≤ k/n ∧ k/n < (712 : ℚ)/1000) :=
by sorry

end NUMINAMATH_CALUDE_least_denominator_for_0711_l1675_167539


namespace NUMINAMATH_CALUDE_coordinates_of_G_l1675_167540

/-- Given a line segment OH with O at (0, 0) and H at (12, 0), 
    and a point G on the same vertical line as H,
    if the line from G through the midpoint M of OH intersects the y-axis at P(0, -4),
    then G has coordinates (12, 4) -/
theorem coordinates_of_G (O H G M P : ℝ × ℝ) : 
  O = (0, 0) →
  H = (12, 0) →
  G.1 = H.1 →
  M = ((O.1 + H.1) / 2, (O.2 + H.2) / 2) →
  P = (0, -4) →
  (∃ t : ℝ, G = t • (M - P) + P) →
  G = (12, 4) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_G_l1675_167540


namespace NUMINAMATH_CALUDE_card_game_probabilities_l1675_167597

-- Define the cards for A and B
def A_cards : Finset ℕ := {2, 3}
def B_cards : Finset ℕ := {1, 2, 3, 4}

-- Define a function to check if a sum is odd
def is_odd_sum (a b : ℕ) : Bool := (a + b) % 2 = 1

-- Define a function to check if B wins
def B_wins (a b : ℕ) : Bool := b > a

theorem card_game_probabilities :
  -- Probability of B drawing two cards with an odd sum
  (Finset.filter (fun p => is_odd_sum p.1 p.2) (Finset.product B_cards B_cards)).card / (Finset.product B_cards B_cards).card = 2/3 ∧
  -- Probability of B winning when A and B each draw a card
  (Finset.filter (fun p => B_wins p.1 p.2) (Finset.product A_cards B_cards)).card / (Finset.product A_cards B_cards).card = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_card_game_probabilities_l1675_167597


namespace NUMINAMATH_CALUDE_bird_lake_swans_l1675_167567

theorem bird_lake_swans (total_birds : ℕ) (duck_fraction : ℚ) : 
  total_birds = 108 →
  duck_fraction = 5/6 →
  (1 - duck_fraction) * total_birds = 18 :=
by sorry

end NUMINAMATH_CALUDE_bird_lake_swans_l1675_167567


namespace NUMINAMATH_CALUDE_two_corners_are_diagonal_endpoints_l1675_167525

/-- A structure representing a checkered rectangle divided into dominoes with diagonals -/
structure CheckeredRectangle where
  rows : ℕ
  cols : ℕ
  dominoes : List (Nat × Nat × Nat × Nat)
  diagonals : List (Nat × Nat × Nat × Nat)

/-- Predicate to check if a point is a corner of the rectangle -/
def is_corner (r : CheckeredRectangle) (x y : ℕ) : Prop :=
  (x = 0 ∨ x = r.cols - 1) ∧ (y = 0 ∨ y = r.rows - 1)

/-- Predicate to check if a point is an endpoint of any diagonal -/
def is_diagonal_endpoint (r : CheckeredRectangle) (x y : ℕ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℕ), (x1, y1, x2, y2) ∈ r.diagonals ∧ ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2))

/-- The main theorem stating that exactly two corners are diagonal endpoints -/
theorem two_corners_are_diagonal_endpoints (r : CheckeredRectangle) 
  (h1 : ∀ (x1 y1 x2 y2 : ℕ), (x1, y1, x2, y2) ∈ r.dominoes → 
    ((x2 = x1 + 1 ∧ y2 = y1) ∨ (x2 = x1 ∧ y2 = y1 + 1)))
  (h2 : ∀ (x1 y1 x2 y2 : ℕ), (x1, y1, x2, y2) ∈ r.diagonals → 
    ∃ (x3 y3 x4 y4 : ℕ), (x3, y3, x4, y4) ∈ r.dominoes ∧ 
    ((x1 = x3 ∧ y1 = y3 ∧ x2 = x4 ∧ y2 = y4) ∨ (x1 = x4 ∧ y1 = y4 ∧ x2 = x3 ∧ y2 = y3)))
  (h3 : ∀ (x1 y1 x2 y2 x3 y3 x4 y4 : ℕ), 
    (x1, y1, x2, y2) ∈ r.diagonals → (x3, y3, x4, y4) ∈ r.diagonals → 
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧ (x1 ≠ x4 ∨ y1 ≠ y4) ∧ (x2 ≠ x3 ∨ y2 ≠ y3) ∧ (x2 ≠ x4 ∨ y2 ≠ y4)) :
  ∃! (c1 c2 : ℕ × ℕ), 
    c1 ≠ c2 ∧ 
    is_corner r c1.1 c1.2 ∧ 
    is_corner r c2.1 c2.2 ∧ 
    is_diagonal_endpoint r c1.1 c1.2 ∧ 
    is_diagonal_endpoint r c2.1 c2.2 ∧ 
    (∀ (x y : ℕ), is_corner r x y → (x, y) ≠ c1 → (x, y) ≠ c2 → ¬is_diagonal_endpoint r x y) :=
sorry

end NUMINAMATH_CALUDE_two_corners_are_diagonal_endpoints_l1675_167525


namespace NUMINAMATH_CALUDE_perpendicular_slope_product_perpendicular_slope_product_contrapositive_l1675_167529

-- Define the lines
def Line (k b : ℝ) := {(x, y) : ℝ × ℝ | y = k * x + b}

-- Define perpendicularity for lines
def Perpendicular (l₁ l₂ : ℝ × ℝ → Prop) : Prop :=
  ∃ k₁ b₁ k₂ b₂ : ℝ, l₁ = Line k₁ b₁ ∧ l₂ = Line k₂ b₂ ∧ k₁ * k₂ = -1

-- State the theorem
theorem perpendicular_slope_product (k₁ k₂ b₁ b₂ : ℝ) :
  (Perpendicular (Line k₁ b₁) (Line k₂ b₂) → k₁ * k₂ = -1) ∧
  (Perpendicular (Line k₁ b₁) (Line k₂ b₂) ↔ k₁ * k₂ = -1) :=
sorry

-- State the contrapositive
theorem perpendicular_slope_product_contrapositive (k₁ k₂ b₁ b₂ : ℝ) :
  (k₁ * k₂ ≠ -1 → ¬Perpendicular (Line k₁ b₁) (Line k₂ b₂)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_slope_product_perpendicular_slope_product_contrapositive_l1675_167529


namespace NUMINAMATH_CALUDE_inequality_proof_l1675_167550

theorem inequality_proof (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1675_167550


namespace NUMINAMATH_CALUDE_allan_brought_two_balloons_l1675_167577

/-- The number of balloons Allan and Jake had in total -/
def total_balloons : ℕ := 6

/-- The number of balloons Jake brought -/
def jake_balloons : ℕ := 4

/-- The number of balloons Allan brought -/
def allan_balloons : ℕ := total_balloons - jake_balloons

theorem allan_brought_two_balloons : allan_balloons = 2 := by
  sorry

end NUMINAMATH_CALUDE_allan_brought_two_balloons_l1675_167577


namespace NUMINAMATH_CALUDE_g_difference_l1675_167579

/-- The function g(n) as defined in the problem -/
def g (n : ℤ) : ℚ := (1 / 4 : ℚ) * n^2 * (n + 1) * (n + 3) + 1

/-- Theorem stating the difference between g(m) and g(m-1) -/
theorem g_difference (m : ℤ) : g m - g (m - 1) = (3 / 4 : ℚ) * m^2 * (m + 5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l1675_167579


namespace NUMINAMATH_CALUDE_solution_equality_l1675_167555

-- Define the function F
def F (a b c : ℝ) : ℝ := a * b^3 + c

-- Theorem statement
theorem solution_equality :
  ∃ a : ℝ, F a 2 3 = F a 3 4 ∧ a = -1/19 := by
  sorry

end NUMINAMATH_CALUDE_solution_equality_l1675_167555


namespace NUMINAMATH_CALUDE_second_divisor_problem_l1675_167568

theorem second_divisor_problem : ∃ (D : ℕ+) (N : ℕ), N % 35 = 25 ∧ N % D = 4 ∧ D = 17 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l1675_167568


namespace NUMINAMATH_CALUDE_choose_two_from_three_l1675_167556

theorem choose_two_from_three (n : ℕ) (k : ℕ) : n = 3 ∧ k = 2 → Nat.choose n k = 3 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_three_l1675_167556


namespace NUMINAMATH_CALUDE_todd_total_gum_l1675_167586

/-- The number of gum pieces Todd has now, given his initial amount and the amount he received. -/
def total_gum (initial : ℕ) (received : ℕ) : ℕ := initial + received

/-- Todd's initial number of gum pieces -/
def todd_initial : ℕ := 38

/-- Number of gum pieces Todd received from Steve -/
def steve_gave : ℕ := 16

/-- Theorem stating that Todd's total gum pieces is 54 -/
theorem todd_total_gum : total_gum todd_initial steve_gave = 54 := by
  sorry

end NUMINAMATH_CALUDE_todd_total_gum_l1675_167586


namespace NUMINAMATH_CALUDE_sea_world_trip_savings_l1675_167569

def trip_cost (parking : ℕ) (entrance : ℕ) (meal : ℕ) (souvenirs : ℕ) (hotel : ℕ) : ℕ :=
  parking + entrance + meal + souvenirs + hotel

def gas_cost (distance : ℕ) (mpg : ℕ) (price_per_gallon : ℕ) : ℕ :=
  (2 * distance / mpg) * price_per_gallon

def additional_savings (total_cost : ℕ) (current_savings : ℕ) : ℕ :=
  total_cost - current_savings

theorem sea_world_trip_savings : 
  let current_savings : ℕ := 28
  let parking : ℕ := 10
  let entrance : ℕ := 55
  let meal : ℕ := 25
  let souvenirs : ℕ := 40
  let hotel : ℕ := 80
  let distance : ℕ := 165
  let mpg : ℕ := 30
  let price_per_gallon : ℕ := 3
  
  let total_trip_cost := trip_cost parking entrance meal souvenirs hotel
  let total_gas_cost := gas_cost distance mpg price_per_gallon
  let total_cost := total_trip_cost + total_gas_cost
  
  additional_savings total_cost current_savings = 215 := by
  sorry

end NUMINAMATH_CALUDE_sea_world_trip_savings_l1675_167569


namespace NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l1675_167543

theorem fgh_supermarkets_in_us (total : ℕ) (difference : ℕ) (us_count : ℕ) : 
  total = 60 →
  difference = 22 →
  us_count = total - difference →
  us_count = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l1675_167543


namespace NUMINAMATH_CALUDE_equation_solution_l1675_167562

theorem equation_solution : 
  ∀ x : ℝ, (x + 1)^2 - 144 = 0 ↔ x = 11 ∨ x = -13 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1675_167562


namespace NUMINAMATH_CALUDE_system_solution_l1675_167515

theorem system_solution (x y t : ℝ) :
  (x^2 + t = 1 ∧ (x + y) * t = 0 ∧ y^2 + t = 1) ↔
  ((t = 0 ∧ ((x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1))) ∨
   (0 < t ∧ t < 1 ∧ ((x = Real.sqrt (1 - t) ∧ y = -Real.sqrt (1 - t)) ∨
                     (x = -Real.sqrt (1 - t) ∧ y = Real.sqrt (1 - t))))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1675_167515


namespace NUMINAMATH_CALUDE_candy_distribution_l1675_167500

theorem candy_distribution (x : ℕ) : 
  x > 500 ∧ 
  x % 21 = 5 ∧ 
  x % 22 = 3 →
  x ≥ 509 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l1675_167500


namespace NUMINAMATH_CALUDE_square_sum_geq_product_l1675_167570

theorem square_sum_geq_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c ≥ a * b * c) : a^2 + b^2 + c^2 ≥ a * b * c := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_l1675_167570


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3s_l1675_167576

/-- The position function of a particle -/
def S (t : ℝ) : ℝ := 2 * t^3 + t

/-- The velocity function of a particle -/
def V (t : ℝ) : ℝ := 6 * t^2 + 1

theorem instantaneous_velocity_at_3s :
  V 3 = 55 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3s_l1675_167576


namespace NUMINAMATH_CALUDE_stating_ball_338_position_l1675_167587

/-- 
Given a circular arrangement of 1000 cups where balls are placed in every 7th cup 
starting from cup 1, this function calculates the cup number for the nth ball.
-/
def ball_position (n : ℕ) : ℕ := 
  (1 + (n - 1) * 7) % 1000

/-- 
Theorem stating that the 338th ball will be placed in cup 359 
in the described arrangement.
-/
theorem ball_338_position : ball_position 338 = 359 := by
  sorry

#eval ball_position 338  -- This line is for verification purposes

end NUMINAMATH_CALUDE_stating_ball_338_position_l1675_167587


namespace NUMINAMATH_CALUDE_sum_is_positive_difference_is_negative_four_l1675_167501

variables (a b : ℝ)

def A : ℝ := a^2 - 2*a*b + b^2
def B : ℝ := a^2 + 2*a*b + b^2

theorem sum_is_positive (h : a ≠ b) : A a b + B a b > 0 := by
  sorry

theorem difference_is_negative_four (h : a * b = 1) : A a b - B a b = -4 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_positive_difference_is_negative_four_l1675_167501


namespace NUMINAMATH_CALUDE_correct_multiplication_result_l1675_167535

theorem correct_multiplication_result : ∃ (n : ℕ), 
  (987 * n = 559989) ∧ 
  (∃ (a b : ℕ), 559981 = 550000 + a * 100 + b * 10 + 1 ∧ a ≠ 9 ∧ b ≠ 8) :=
by sorry

end NUMINAMATH_CALUDE_correct_multiplication_result_l1675_167535


namespace NUMINAMATH_CALUDE_least_satisfying_number_l1675_167574

def satisfies_conditions (n : ℕ) : Prop :=
  n % 10 = 9 ∧ n % 11 = 10 ∧ n % 12 = 11 ∧ n % 13 = 12

theorem least_satisfying_number : 
  satisfies_conditions 8579 ∧ 
  ∀ m : ℕ, m < 8579 → ¬(satisfies_conditions m) :=
by sorry

end NUMINAMATH_CALUDE_least_satisfying_number_l1675_167574


namespace NUMINAMATH_CALUDE_rainy_days_count_l1675_167548

theorem rainy_days_count (n : ℕ) : 
  (∃ (rainy_days non_rainy_days : ℕ),
    rainy_days + non_rainy_days = 7 ∧
    n * rainy_days + 5 * non_rainy_days = 22 ∧
    5 * non_rainy_days - n * rainy_days = 8) →
  (∃ (rainy_days : ℕ), rainy_days = 4) :=
by sorry

end NUMINAMATH_CALUDE_rainy_days_count_l1675_167548


namespace NUMINAMATH_CALUDE_third_number_in_ratio_l1675_167557

theorem third_number_in_ratio (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a : ℚ) / 5 = (b : ℚ) / 6 ∧ (b : ℚ) / 6 = (c : ℚ) / 8 →
  a + c = b + 49 →
  b = 42 := by sorry

end NUMINAMATH_CALUDE_third_number_in_ratio_l1675_167557


namespace NUMINAMATH_CALUDE_park_boats_l1675_167502

theorem park_boats (total_boats : ℕ) (large_capacity : ℕ) (small_capacity : ℕ) 
  (h1 : total_boats = 42)
  (h2 : large_capacity = 6)
  (h3 : small_capacity = 4)
  (h4 : ∃ (large_boats small_boats : ℕ), 
    large_boats + small_boats = total_boats ∧ 
    large_capacity * large_boats = 2 * small_capacity * small_boats) :
  ∃ (large_boats small_boats : ℕ), 
    large_boats = 24 ∧ 
    small_boats = 18 ∧ 
    large_boats + small_boats = total_boats ∧ 
    large_capacity * large_boats = 2 * small_capacity * small_boats :=
by sorry

end NUMINAMATH_CALUDE_park_boats_l1675_167502


namespace NUMINAMATH_CALUDE_consecutive_even_count_l1675_167504

def is_consecutive_even (a b : ℕ) : Prop := b = a + 2

def sum_consecutive_even (start : ℕ) (count : ℕ) : ℕ :=
  (count * (2 * start + count - 1))

theorem consecutive_even_count :
  ∃ (count : ℕ), 
    sum_consecutive_even 80 count = 246 ∧
    count = 3 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_count_l1675_167504


namespace NUMINAMATH_CALUDE_num_installments_is_40_l1675_167584

/-- Proves that the number of installments is 40 given the payment conditions --/
theorem num_installments_is_40 
  (n : ℕ) -- Total number of installments
  (h1 : n ≥ 20) -- At least 20 installments
  (first_20_payment : ℕ := 410) -- First 20 payments
  (remaining_payment : ℕ := 475) -- Remaining payments
  (average_payment : ℚ := 442.5) -- Average payment
  (h2 : (20 * first_20_payment + (n - 20) * remaining_payment : ℚ) / n = average_payment) -- Average payment equation
  : n = 40 := by
  sorry

end NUMINAMATH_CALUDE_num_installments_is_40_l1675_167584


namespace NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_factors_l1675_167566

theorem largest_number_from_hcf_lcm_factors (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 40)
  (lcm_eq : Nat.lcm a b = 40 * 11 * 12) :
  max a b = 480 := by
sorry

end NUMINAMATH_CALUDE_largest_number_from_hcf_lcm_factors_l1675_167566


namespace NUMINAMATH_CALUDE_smallest_interesting_number_l1675_167547

theorem smallest_interesting_number : 
  ∃ (n : ℕ), n = 1800 ∧ 
  (∀ (m : ℕ), m < n → ¬(∃ (k : ℕ), 2 * m = k ^ 2) ∨ ¬(∃ (l : ℕ), 15 * m = l ^ 3)) ∧
  (∃ (k : ℕ), 2 * n = k ^ 2) ∧
  (∃ (l : ℕ), 15 * n = l ^ 3) := by
sorry

end NUMINAMATH_CALUDE_smallest_interesting_number_l1675_167547


namespace NUMINAMATH_CALUDE_a_less_than_b_less_than_c_l1675_167575

theorem a_less_than_b_less_than_c : ∀ a b c : ℝ,
  a = Real.log (1/2) →
  b = Real.sin (1/2) →
  c = 2^(-1/2 : ℝ) →
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_a_less_than_b_less_than_c_l1675_167575


namespace NUMINAMATH_CALUDE_sin_2phi_value_l1675_167527

theorem sin_2phi_value (φ : ℝ) 
  (h : ∫ x in (0)..(Real.pi / 2), Real.sin (x - φ) = Real.sqrt 7 / 4) : 
  Real.sin (2 * φ) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_2phi_value_l1675_167527


namespace NUMINAMATH_CALUDE_angle_C_measure_side_ratio_bounds_l1675_167510

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_A : 0 < A
  pos_B : 0 < B
  pos_C : 0 < C
  sum_angles : A + B + C = π

variable (t : Triangle)

/-- First theorem: If sin(2C - π/2) = 1/2 and a² + b² < c², then C = 2π/3 -/
theorem angle_C_measure (h1 : sin (2 * t.C - π/2) = 1/2) (h2 : t.a^2 + t.b^2 < t.c^2) :
  t.C = 2*π/3 := by sorry

/-- Second theorem: If C = 2π/3, then 1 < (a + b)/c ≤ 2√3/3 -/
theorem side_ratio_bounds (h : t.C = 2*π/3) :
  1 < (t.a + t.b) / t.c ∧ (t.a + t.b) / t.c ≤ 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_angle_C_measure_side_ratio_bounds_l1675_167510


namespace NUMINAMATH_CALUDE_sum_remainder_mod_11_l1675_167564

theorem sum_remainder_mod_11 : 
  (101234 + 101235 + 101236 + 101237 + 101238 + 101239 + 101240) % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_11_l1675_167564


namespace NUMINAMATH_CALUDE_range_of_circle_l1675_167560

theorem range_of_circle (x y : ℝ) (h : x^2 + y^2 = 4*x) :
  ∃ (z : ℝ), z = x^2 + y^2 ∧ 0 ≤ z ∧ z ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_range_of_circle_l1675_167560


namespace NUMINAMATH_CALUDE_range_of_x_l1675_167530

theorem range_of_x (x : ℝ) : 
  (|x - 1| + |x - 2| = 1) → (1 ≤ x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1675_167530


namespace NUMINAMATH_CALUDE_extra_interest_proof_l1675_167503

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem extra_interest_proof (principal : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ) :
  principal = 15000 →
  rate1 = 0.15 →
  rate2 = 0.12 →
  time = 2 →
  simple_interest principal rate1 time - simple_interest principal rate2 time = 900 := by
  sorry

end NUMINAMATH_CALUDE_extra_interest_proof_l1675_167503


namespace NUMINAMATH_CALUDE_largest_710_double_correct_l1675_167511

/-- Converts a base-10 number to its base-7 representation as a list of digits --/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Interprets a list of digits as a base-10 number --/
def fromDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a number is a 7-10 double --/
def is710Double (n : ℕ) : Prop :=
  fromDigits (toBase7 n) = 2 * n

/-- The largest 7-10 double --/
def largest710Double : ℕ := 315

theorem largest_710_double_correct :
  is710Double largest710Double ∧
  ∀ n : ℕ, n > largest710Double → ¬is710Double n :=
sorry

end NUMINAMATH_CALUDE_largest_710_double_correct_l1675_167511


namespace NUMINAMATH_CALUDE_project_hours_total_l1675_167585

/-- Given the conditions of the project hours charged by Pat, Kate, and Mark, 
    prove that the total number of hours charged is 144. -/
theorem project_hours_total (k p m : ℕ) : 
  p = 2 * k →          -- Pat charged twice as much as Kate
  3 * p = m →          -- Pat charged 1/3 as much as Mark
  m = k + 80 →         -- Mark charged 80 more hours than Kate
  k + p + m = 144 :=   -- Total hours charged
by sorry

end NUMINAMATH_CALUDE_project_hours_total_l1675_167585


namespace NUMINAMATH_CALUDE_parallel_vectors_k_equals_two_l1675_167599

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, prove that if they are parallel, then k = 2 -/
theorem parallel_vectors_k_equals_two (k : ℝ) :
  let a : ℝ × ℝ := (k - 1, k)
  let b : ℝ × ℝ := (1, 2)
  are_parallel a b → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_equals_two_l1675_167599


namespace NUMINAMATH_CALUDE_employee_count_l1675_167552

theorem employee_count :
  ∀ (E : ℕ) (M : ℝ),
    M = 0.99 * (E : ℝ) →
    M - 299.9999999999997 = 0.98 * (E : ℝ) →
    E = 30000 :=
by
  sorry

end NUMINAMATH_CALUDE_employee_count_l1675_167552


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l1675_167534

theorem arithmetic_expression_equals_24 : (8 * 10 - 8) / 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l1675_167534


namespace NUMINAMATH_CALUDE_cone_hemisphere_relation_cone_base_radius_is_10_5_l1675_167519

/-- Represents a cone with a hemisphere resting on its base --/
structure ConeWithHemisphere where
  cone_height : ℝ
  hemisphere_radius : ℝ
  cone_base_radius : ℝ

/-- Checks if the configuration is valid --/
def is_valid_configuration (c : ConeWithHemisphere) : Prop :=
  c.cone_height > 0 ∧ c.hemisphere_radius > 0 ∧ c.cone_base_radius > c.hemisphere_radius

/-- Theorem stating the relationship between cone dimensions and hemisphere --/
theorem cone_hemisphere_relation (c : ConeWithHemisphere) 
  (h_valid : is_valid_configuration c)
  (h_height : c.cone_height = 9)
  (h_radius : c.hemisphere_radius = 3) :
  c.cone_base_radius = 10.5 := by
  sorry

/-- Main theorem proving the base radius of the cone --/
theorem cone_base_radius_is_10_5 :
  ∃ c : ConeWithHemisphere, 
    is_valid_configuration c ∧ 
    c.cone_height = 9 ∧ 
    c.hemisphere_radius = 3 ∧ 
    c.cone_base_radius = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_cone_hemisphere_relation_cone_base_radius_is_10_5_l1675_167519

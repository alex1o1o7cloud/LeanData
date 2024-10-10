import Mathlib

namespace wildflower_color_difference_l1504_150441

/-- Given the following conditions about wildflowers:
  * The total number of wildflowers is 44
  * There are 13 yellow and white flowers
  * There are 17 red and yellow flowers
  * There are 14 red and white flowers

  Prove that the number of flowers containing red minus
  the number of flowers containing white equals 4.
-/
theorem wildflower_color_difference
  (total : ℕ)
  (yellow_white : ℕ)
  (red_yellow : ℕ)
  (red_white : ℕ)
  (h_total : total = 44)
  (h_yellow_white : yellow_white = 13)
  (h_red_yellow : red_yellow = 17)
  (h_red_white : red_white = 14) :
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by
  sorry

end wildflower_color_difference_l1504_150441


namespace inverse_proportion_l1504_150421

/-- Given that p and q are inversely proportional, prove that if p = 30 when q = 4, 
    then p = 240/11 when q = 5.5 -/
theorem inverse_proportion (p q : ℝ) (h : ∃ k : ℝ, ∀ x y : ℝ, p * q = k) 
    (h1 : p = 30 ∧ q = 4) : 
    (p = 240/11 ∧ q = 5.5) := by
  sorry

end inverse_proportion_l1504_150421


namespace repeating_block_length_seven_thirteenths_l1504_150445

/-- The length of the smallest repeating block in the decimal expansion of 7/13 is 6. -/
theorem repeating_block_length_seven_thirteenths : 
  ∃ (d : ℕ) (n : ℕ), d = 6 ∧ 7 * (10^d - 1) = 13 * n :=
by sorry

end repeating_block_length_seven_thirteenths_l1504_150445


namespace triangle_side_length_l1504_150426

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 10 →
  c = 3 →
  Real.cos A = 1/4 →
  b^2 + c^2 - a^2 = 2 * b * c * Real.cos A →
  b = 2 :=
by sorry

end triangle_side_length_l1504_150426


namespace simplify_expression_l1504_150494

theorem simplify_expression : (8 * (10 ^ 12)) / (4 * (10 ^ 4)) = 200000000 := by
  sorry

end simplify_expression_l1504_150494


namespace quadratic_minimum_values_l1504_150436

/-- A quadratic function f(x) = mx^2 - 2mx + 2 with m ≠ 0 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * m * x + 2

/-- The theorem stating the conditions and conclusion -/
theorem quadratic_minimum_values (m : ℝ) :
  m ≠ 0 →
  (∀ x, -2 ≤ x → x < 2 → f m x ≥ -2) →
  (∃ x, -2 ≤ x ∧ x < 2 ∧ f m x = -2) →
  (m = 4 ∨ m = -1/2) :=
by sorry

end quadratic_minimum_values_l1504_150436


namespace expression_evaluation_l1504_150483

/-- Given a = 4 and b = -3, prove that 2a^2 - 3b^2 + 4ab = -43 -/
theorem expression_evaluation (a b : ℤ) (ha : a = 4) (hb : b = -3) :
  2 * a^2 - 3 * b^2 + 4 * a * b = -43 := by
  sorry

end expression_evaluation_l1504_150483


namespace sphere_surface_area_l1504_150453

theorem sphere_surface_area (c : Real) (h : c = 2 * Real.pi) :
  ∃ (r : Real), 
    c = 2 * Real.pi * r ∧ 
    4 * Real.pi * r^2 = 4 * Real.pi :=
by sorry

end sphere_surface_area_l1504_150453


namespace square_plus_reciprocal_square_l1504_150405

theorem square_plus_reciprocal_square (x : ℝ) (h : x ≠ 0) :
  x^2 + 1/x^2 = 7 → x^4 + 1/x^4 = 47 := by
  sorry

end square_plus_reciprocal_square_l1504_150405


namespace reasonable_prize_distribution_l1504_150478

/-- The most reasonable prize distribution for a math competition problem --/
theorem reasonable_prize_distribution
  (total_prize : ℝ)
  (prob_A : ℝ)
  (prob_B : ℝ)
  (h_total : total_prize = 190)
  (h_prob_A : prob_A = 3/4)
  (h_prob_B : prob_B = 4/5)
  (h_prob_valid : 0 ≤ prob_A ∧ prob_A ≤ 1 ∧ 0 ≤ prob_B ∧ prob_B ≤ 1) :
  let expected_A := (prob_A * (1 - prob_B) * total_prize + prob_A * prob_B * (total_prize / 2))
  let expected_B := (prob_B * (1 - prob_A) * total_prize + prob_A * prob_B * (total_prize / 2))
  expected_A = 90 ∧ expected_B = 100 :=
by sorry


end reasonable_prize_distribution_l1504_150478


namespace problem_statement_l1504_150470

theorem problem_statement (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : x^2 + y^2 - x*y = 4) : 
  x^4 + y^4 + x^3*y + x*y^3 = 36 := by
sorry

end problem_statement_l1504_150470


namespace rolling_cube_dot_path_length_l1504_150429

/-- The path length of a dot on a rolling cube -/
theorem rolling_cube_dot_path_length :
  let cube_side : ℝ := 2
  let dot_distance : ℝ := 2 / 3
  let path_length : ℝ := (4 * Real.pi * Real.sqrt 10) / 3
  cube_side > 0 ∧ 0 < dot_distance ∧ dot_distance < cube_side →
  path_length = 4 * (Real.pi * Real.sqrt (dot_distance^2 + cube_side^2)) / 2 :=
by sorry


end rolling_cube_dot_path_length_l1504_150429


namespace function_is_constant_l1504_150439

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def is_continuous (f : ℝ → ℝ) : Prop := Continuous f

def satisfies_inequality (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (3 * x - 2) ≤ f x ∧ f x ≤ f (2 * x - 1)

-- State the theorem
theorem function_is_constant
  (h_continuous : is_continuous f)
  (h_inequality : satisfies_inequality f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end function_is_constant_l1504_150439


namespace count_valid_pairs_l1504_150410

def is_valid_pair (A B : ℕ+) : Prop :=
  12 ∣ A ∧ 12 ∣ B ∧
  20 ∣ A ∧ 20 ∣ B ∧
  45 ∣ A ∧ 45 ∣ B ∧
  Nat.lcm A B = 4320

theorem count_valid_pairs :
  ∃! (pairs : Finset (ℕ+ × ℕ+)), 
    (∀ p ∈ pairs, is_valid_pair p.1 p.2) ∧
    (∀ A B, is_valid_pair A B → (A, B) ∈ pairs) ∧
    pairs.card = 11 := by
  sorry

end count_valid_pairs_l1504_150410


namespace circle_intersection_parallelogram_l1504_150409

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if two circles intersect non-tangentially -/
def nonTangentialIntersection (c1 c2 : Circle) : Prop :=
  sorry

/-- Finds the intersection points of two circles -/
def circleIntersection (c1 c2 : Circle) : Set Point :=
  sorry

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (a b c d : Point) : Prop :=
  sorry

theorem circle_intersection_parallelogram 
  (k1 k2 k3 : Circle)
  (P : Point)
  (A B : Point)
  (D C : Point)
  (h1 : k1.radius = k2.radius ∧ k2.radius = k3.radius)
  (h2 : nonTangentialIntersection k1 k2 ∧ nonTangentialIntersection k2 k3 ∧ nonTangentialIntersection k3 k1)
  (h3 : P ∈ circleIntersection k1 k2 ∩ circleIntersection k2 k3 ∩ circleIntersection k3 k1)
  (h4 : A = k1.center)
  (h5 : B = k2.center)
  (h6 : D ∈ circleIntersection k1 k3 ∧ D ≠ P)
  (h7 : C ∈ circleIntersection k2 k3 ∧ C ≠ P)
  : isParallelogram A B C D :=
sorry

end circle_intersection_parallelogram_l1504_150409


namespace sqrt_difference_power_l1504_150462

theorem sqrt_difference_power (A B : ℤ) : 
  ∃ A B : ℤ, (Real.sqrt 1969 - Real.sqrt 1968) ^ 1969 = A * Real.sqrt 1969 - B * Real.sqrt 1968 ∧ 
  1969 * A^2 - 1968 * B^2 = 1 := by
  sorry

end sqrt_difference_power_l1504_150462


namespace imaginary_part_of_z_l1504_150402

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.I) : 
  z.im = 1/2 := by
  sorry

end imaginary_part_of_z_l1504_150402


namespace geralds_bag_contains_40_apples_l1504_150498

/-- The number of bags Pam has -/
def pams_bags : ℕ := 10

/-- The total number of apples Pam has -/
def pams_total_apples : ℕ := 1200

/-- The number of apples in each of Gerald's bags -/
def geralds_bag_apples : ℕ := pams_total_apples / (3 * pams_bags)

/-- Theorem stating that each of Gerald's bags contains 40 apples -/
theorem geralds_bag_contains_40_apples : geralds_bag_apples = 40 := by
  sorry

end geralds_bag_contains_40_apples_l1504_150498


namespace prob_second_draw_3_eq_11_48_l1504_150417

-- Define the boxes and their initial contents
def box1 : Finset ℕ := {1, 1, 2, 3}
def box2 : Finset ℕ := {1, 1, 3}
def box3 : Finset ℕ := {1, 1, 1, 2, 2}

-- Define the probability of drawing a ball from a box
def prob_draw (box : Finset ℕ) (label : ℕ) : ℚ :=
  (box.filter (λ x => x = label)).card / box.card

-- Define the probability of the second draw being 3
def prob_second_draw_3 : ℚ :=
  (prob_draw box1 1 * prob_draw (box1 ∪ {1}) 3) +
  (prob_draw box1 2 * prob_draw (box2 ∪ {2}) 3) +
  (prob_draw box1 3 * prob_draw (box3 ∪ {3}) 3)

-- Theorem statement
theorem prob_second_draw_3_eq_11_48 : prob_second_draw_3 = 11 / 48 := by
  sorry


end prob_second_draw_3_eq_11_48_l1504_150417


namespace triangle_angle_measure_l1504_150424

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Area formula
  b^2 / (3 * Real.sin B) = (1/2) * a * c * Real.sin B →
  -- Given condition
  6 * Real.cos A * Real.cos C = 1 →
  -- Given side length
  b = 3 →
  -- Conclusion
  B = π/3 := by sorry

end triangle_angle_measure_l1504_150424


namespace expression_evaluation_l1504_150466

theorem expression_evaluation (m n : ℤ) (hm : m = -1) (hn : n = 2) :
  3 * m^2 * n - 2 * m * n^2 - 4 * m^2 * n + m * n^2 = 2 := by
  sorry

end expression_evaluation_l1504_150466


namespace car_speed_time_relation_l1504_150406

theorem car_speed_time_relation (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) :
  distance = 540 ∧ original_time = 12 ∧ new_speed = 60 →
  (distance / new_speed) / original_time = 3 / 4 :=
by
  sorry

end car_speed_time_relation_l1504_150406


namespace second_frog_hops_second_frog_hops_proof_l1504_150442

/-- Given three frogs hopping across a road, prove the number of hops taken by the second frog -/
theorem second_frog_hops : ℕ → ℕ → ℕ → Prop :=
  fun frog1 frog2 frog3 =>
    frog1 = 4 * frog2 ∧            -- First frog takes 4 times as many hops as the second
    frog2 = 2 * frog3 ∧            -- Second frog takes twice as many hops as the third
    frog1 + frog2 + frog3 = 99 →   -- Total hops is 99
    frog2 = 18                     -- Second frog takes 18 hops

theorem second_frog_hops_proof : ∃ (frog1 frog2 frog3 : ℕ), second_frog_hops frog1 frog2 frog3 := by
  sorry

end second_frog_hops_second_frog_hops_proof_l1504_150442


namespace green_eyed_brunettes_l1504_150425

theorem green_eyed_brunettes (total : ℕ) (blueEyedBlondes : ℕ) (brunettes : ℕ) (greenEyed : ℕ) :
  total = 60 →
  blueEyedBlondes = 20 →
  brunettes = 35 →
  greenEyed = 25 →
  ∃ (greenEyedBrunettes : ℕ),
    greenEyedBrunettes = 10 ∧
    greenEyedBrunettes ≤ brunettes ∧
    greenEyedBrunettes ≤ greenEyed ∧
    blueEyedBlondes + (brunettes - greenEyedBrunettes) + greenEyed = total :=
by
  sorry

end green_eyed_brunettes_l1504_150425


namespace ashoks_marks_l1504_150438

theorem ashoks_marks (total_subjects : ℕ) (average_6_subjects : ℝ) (marks_6th_subject : ℝ) :
  total_subjects = 6 →
  average_6_subjects = 80 →
  marks_6th_subject = 110 →
  let total_marks := average_6_subjects * total_subjects
  let marks_5_subjects := total_marks - marks_6th_subject
  let average_5_subjects := marks_5_subjects / 5
  average_5_subjects = 74 := by
sorry

end ashoks_marks_l1504_150438


namespace gcd_of_specific_numbers_l1504_150423

theorem gcd_of_specific_numbers : Nat.gcd 55555555 111111111 = 11111111 := by
  sorry

end gcd_of_specific_numbers_l1504_150423


namespace cafe_problem_l1504_150459

/-- The number of local offices that ordered sandwiches -/
def num_offices : ℕ := 3

/-- The number of sandwiches ordered by each office -/
def sandwiches_per_office : ℕ := 10

/-- The number of sandwiches ordered by each customer in half of the group -/
def sandwiches_per_customer : ℕ := 4

/-- The total number of sandwiches made by the café -/
def total_sandwiches : ℕ := 54

/-- The number of customers in the group that arrived at the café -/
def num_customers : ℕ := 12

theorem cafe_problem :
  num_offices * sandwiches_per_office +
  (num_customers / 2) * sandwiches_per_customer =
  total_sandwiches :=
by sorry

end cafe_problem_l1504_150459


namespace twenty_paise_coins_count_l1504_150481

/-- Given a total of 344 coins consisting of 20 paise and 25 paise coins,
    with a total value of Rs. 71, prove that the number of 20 paise coins is 300. -/
theorem twenty_paise_coins_count :
  ∀ (x y : ℕ),
  x + y = 344 →
  20 * x + 25 * y = 7100 →
  x = 300 :=
by sorry

end twenty_paise_coins_count_l1504_150481


namespace lance_licks_l1504_150416

/-- The number of licks it takes Dan to get to the center of a lollipop -/
def dan_licks : ℕ := 58

/-- The number of licks it takes Michael to get to the center of a lollipop -/
def michael_licks : ℕ := 63

/-- The number of licks it takes Sam to get to the center of a lollipop -/
def sam_licks : ℕ := 70

/-- The number of licks it takes David to get to the center of a lollipop -/
def david_licks : ℕ := 70

/-- The average number of licks it takes for all 5 people to get to the center of a lollipop -/
def average_licks : ℕ := 60

/-- The number of people in the group -/
def num_people : ℕ := 5

/-- The theorem stating how many licks it takes Lance to get to the center of a lollipop -/
theorem lance_licks : 
  (num_people * average_licks) - (dan_licks + michael_licks + sam_licks + david_licks) = 39 := by
  sorry

end lance_licks_l1504_150416


namespace probability_of_two_red_balls_l1504_150473

def total_balls : ℕ := 7 + 5 + 4

def red_balls : ℕ := 7

def balls_picked : ℕ := 2

def probability_both_red : ℚ := 175 / 1000

theorem probability_of_two_red_balls :
  (Nat.choose red_balls balls_picked : ℚ) / (Nat.choose total_balls balls_picked : ℚ) = probability_both_red :=
by sorry

end probability_of_two_red_balls_l1504_150473


namespace rectangular_solid_volume_l1504_150418

/-- The volume of a rectangular solid with given face areas and a dimension relation -/
theorem rectangular_solid_volume (a b c : ℝ) 
  (side_area : a * b = 15)
  (front_area : b * c = 10)
  (top_area : a * c = 6)
  (dimension_relation : b = 2 * a ∨ a = 2 * b ∨ c = 2 * a ∨ a = 2 * c ∨ c = 2 * b ∨ b = 2 * c) :
  a * b * c = 12 := by
  sorry

end rectangular_solid_volume_l1504_150418


namespace square_from_equation_l1504_150472

theorem square_from_equation (x y z : ℕ) 
  (h : x^2 + y^2 + z^2 = 2*(x*y + y*z + z*x)) :
  ∃ (a b c : ℕ), x = a^2 ∧ y = b^2 ∧ z = c^2 := by
  sorry

end square_from_equation_l1504_150472


namespace sally_bought_48_eggs_l1504_150437

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Sally bought -/
def dozens_bought : ℕ := 4

/-- Theorem: Sally bought 48 eggs -/
theorem sally_bought_48_eggs : dozens_bought * eggs_per_dozen = 48 := by
  sorry

end sally_bought_48_eggs_l1504_150437


namespace james_weekly_pistachio_expense_l1504_150452

/-- Represents the cost of pistachios in dollars per can. -/
def cost_per_can : ℝ := 10

/-- Represents the amount of pistachios in ounces per can. -/
def ounces_per_can : ℝ := 5

/-- Represents the amount of pistachios James eats in ounces every 5 days. -/
def ounces_per_five_days : ℝ := 30

/-- Represents the number of days in a week. -/
def days_in_week : ℝ := 7

/-- Proves that James spends $84 per week on pistachios. -/
theorem james_weekly_pistachio_expense : 
  (cost_per_can / ounces_per_can) * (ounces_per_five_days / 5) * days_in_week = 84 := by
  sorry

end james_weekly_pistachio_expense_l1504_150452


namespace password_probability_l1504_150458

/-- Represents the set of symbols used in the password -/
def SymbolSet : Finset Char := {'!', '@', '#', '$', '%'}

/-- Represents the set of favorable symbols -/
def FavorableSymbols : Finset Char := {'$', '%', '@'}

/-- Represents the set of two-digit numbers (00 to 99) -/
def TwoDigitNumbers : Finset Nat := Finset.range 100

/-- Represents the set of even two-digit numbers -/
def EvenTwoDigitNumbers : Finset Nat := TwoDigitNumbers.filter (fun n => n % 2 = 0)

/-- The probability of Alice's password meeting the specific criteria -/
theorem password_probability : 
  (EvenTwoDigitNumbers.card : ℚ) / TwoDigitNumbers.card * 
  (FavorableSymbols.card : ℚ) / SymbolSet.card * 
  (EvenTwoDigitNumbers.card : ℚ) / TwoDigitNumbers.card = 3 / 20 := by
  sorry


end password_probability_l1504_150458


namespace right_triangle_sides_l1504_150407

theorem right_triangle_sides (x Δ : ℝ) (hx : x > 0) (hΔ : Δ > 0) :
  (x + 2*Δ)^2 = x^2 + (x + Δ)^2 ↔ x = (Δ*(-1 + 2*Real.sqrt 7))/2 := by
  sorry

end right_triangle_sides_l1504_150407


namespace stating_distinguishable_triangles_count_l1504_150433

/-- Represents the number of available colors for the triangles -/
def num_colors : ℕ := 8

/-- Represents the number of small triangles needed to construct a large triangle -/
def triangles_per_large : ℕ := 4

/-- 
Calculates the number of distinguishable large equilateral triangles that can be constructed
given the number of available colors and the number of small triangles per large triangle.
-/
def count_distinguishable_triangles (colors : ℕ) (triangles : ℕ) : ℕ :=
  colors * (colors - 1) * (colors - 2) * (colors - 3)

/-- 
Theorem stating that the number of distinguishable large equilateral triangles
that can be constructed under the given conditions is 1680.
-/
theorem distinguishable_triangles_count :
  count_distinguishable_triangles num_colors triangles_per_large = 1680 := by
  sorry


end stating_distinguishable_triangles_count_l1504_150433


namespace total_chocolate_bars_l1504_150465

/-- The number of chocolate bars in a massive crate -/
def chocolateBarsInCrate (largeBozesPerCrate mediumBoxesPerLarge smallBoxesPerMedium barsPerSmall : ℕ) : ℕ :=
  largeBozesPerCrate * mediumBoxesPerLarge * smallBoxesPerMedium * barsPerSmall

/-- Theorem: The massive crate contains 153,900 chocolate bars -/
theorem total_chocolate_bars :
  chocolateBarsInCrate 10 19 27 30 = 153900 := by
  sorry

#eval chocolateBarsInCrate 10 19 27 30

end total_chocolate_bars_l1504_150465


namespace beta_values_l1504_150420

theorem beta_values (β : ℂ) (h1 : β ≠ 1) 
  (h2 : Complex.abs (β^3 - 1) = 3 * Complex.abs (β - 1))
  (h3 : Complex.abs (β^6 - 1) = 6 * Complex.abs (β - 1)) :
  β = Complex.I * 2 * Real.sqrt 2 ∨ β = Complex.I * (-2) * Real.sqrt 2 :=
sorry

end beta_values_l1504_150420


namespace power_product_equals_sum_l1504_150434

theorem power_product_equals_sum (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_product_equals_sum_l1504_150434


namespace smallest_n_for_square_root_solution_l1504_150469

def is_square_integer (x : ℚ) : Prop :=
  ∃ m : ℤ, x = m^2

theorem smallest_n_for_square_root (n : ℕ) : Prop :=
  n ≥ 2 ∧ 
  is_square_integer ((n + 1) * (2 * n + 1) / 6) ∧
  ∀ k : ℕ, k ≥ 2 ∧ k < n → ¬is_square_integer ((k + 1) * (2 * k + 1) / 6)

theorem solution : smallest_n_for_square_root 337 := by
  sorry

end smallest_n_for_square_root_solution_l1504_150469


namespace smallest_common_factor_l1504_150476

theorem smallest_common_factor (n : ℕ) : 
  (∃ (k : ℕ), k > 1 ∧ k ∣ (11*n - 4) ∧ k ∣ (8*n - 5)) ∧ 
  (∀ (m : ℕ), m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (11*m - 4) ∧ k ∣ (8*m - 5))) → 
  n = 15 :=
sorry

end smallest_common_factor_l1504_150476


namespace zongzi_sales_l1504_150477

/-- The cost and profit calculation for zongzi sales during the Dragon Boat Festival --/
theorem zongzi_sales (x : ℝ) (m : ℝ) : 
  /- Cost price after festival -/
  (∀ y, y > 0 → 240 / y - 4 = 240 / (y + 2) → y = x) →
  /- Total cost constraint -/
  ((12 : ℝ) * m + 10 * (400 - m) ≤ 4600) →
  /- Profit calculation -/
  (∀ w, w = 2 * m + 2400) →
  /- Conclusions -/
  (x = 10 ∧ m = 300 ∧ (2 * 300 + 2400 = 3000)) := by
  sorry


end zongzi_sales_l1504_150477


namespace sin_10_50_70_equals_one_eighth_l1504_150448

theorem sin_10_50_70_equals_one_eighth :
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) = 1 / 8 := by
  sorry

end sin_10_50_70_equals_one_eighth_l1504_150448


namespace multiplication_error_l1504_150430

theorem multiplication_error (a b : ℕ) (h1 : 100 ≤ a ∧ a < 1000) (h2 : 100 ≤ b ∧ b < 1000) :
  (∃ n : ℕ, 10000 * a + b = n * (a * b)) → (∃ n : ℕ, 10000 * a + b = 73 * (a * b)) :=
by sorry

end multiplication_error_l1504_150430


namespace additional_tank_capacity_l1504_150455

theorem additional_tank_capacity
  (existing_tanks : ℕ)
  (fish_per_existing_tank : ℕ)
  (additional_tanks : ℕ)
  (total_fish : ℕ)
  (h1 : existing_tanks = 3)
  (h2 : fish_per_existing_tank = 15)
  (h3 : additional_tanks = 3)
  (h4 : total_fish = 75) :
  (total_fish - existing_tanks * fish_per_existing_tank) / additional_tanks = 10 :=
by sorry

end additional_tank_capacity_l1504_150455


namespace product_sum_equals_power_l1504_150484

theorem product_sum_equals_power : 
  2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) * (3^32 + 1) * (3^64 + 1) + 1 = 3^128 := by
  sorry

end product_sum_equals_power_l1504_150484


namespace range_of_c_l1504_150412

def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^y < c^x

def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 + x + (1/2) * c > 0

theorem range_of_c (c : ℝ) (h_c : c > 0) 
  (h_or : p c ∨ q c) (h_not_and : ¬(p c ∧ q c)) : 
  c ∈ Set.Ioc 0 (1/2) ∪ Set.Ici 1 := by
  sorry

end range_of_c_l1504_150412


namespace factor_of_polynomial_l1504_150474

theorem factor_of_polynomial (c d : ℤ) : 
  (∀ x : ℝ, x^2 - x - 1 = 0 → c * x^19 + d * x^18 + 1 = 0) ↔ 
  (c = 1597 ∧ d = -2584) :=
by sorry

end factor_of_polynomial_l1504_150474


namespace min_x_coord_midpoint_l1504_150411

/-- Given a segment AB of length 3 with endpoints on the parabola y^2 = x,
    the minimum x-coordinate of the midpoint M of AB is 5/4 -/
theorem min_x_coord_midpoint (A B M : ℝ × ℝ) :
  (A.2^2 = A.1) →  -- A is on the parabola y^2 = x
  (B.2^2 = B.1) →  -- B is on the parabola y^2 = x
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 9 →  -- AB has length 3
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  M.1 ≥ 5/4 :=
sorry

end min_x_coord_midpoint_l1504_150411


namespace camping_probability_l1504_150422

theorem camping_probability (p_rain p_tents_on_time : ℝ) : 
  p_rain = 1 / 2 →
  p_tents_on_time = 1 / 2 →
  (p_rain * (1 - p_tents_on_time)) = 1 / 4 :=
by
  sorry

end camping_probability_l1504_150422


namespace escalator_steps_l1504_150443

/-- The number of steps on an escalator between two floors -/
def N : ℕ := 47

/-- The number of steps Jack walks while on the moving escalator -/
def jack_steps : ℕ := 29

/-- The number of steps Jill walks while on the moving escalator -/
def jill_steps : ℕ := 11

/-- Jill's travel time is twice Jack's -/
def time_ratio : ℕ := 2

theorem escalator_steps :
  N - jill_steps = time_ratio * (N - jack_steps) :=
sorry

end escalator_steps_l1504_150443


namespace train_length_l1504_150480

/-- The length of a train given its passing times over different distances -/
theorem train_length (tree_time platform_time platform_length : ℝ) 
  (h1 : tree_time = 120)
  (h2 : platform_time = 170)
  (h3 : platform_length = 500)
  (h4 : tree_time > 0)
  (h5 : platform_time > 0)
  (h6 : platform_length > 0) :
  let train_length := (platform_time * platform_length) / (platform_time - tree_time)
  train_length = 1200 := by
sorry


end train_length_l1504_150480


namespace quadratic_sequence_proof_l1504_150451

/-- A quadratic function passing through the origin with given derivative -/
noncomputable def f (x : ℝ) : ℝ := sorry

/-- The sequence a_n -/
def a (n : ℕ+) : ℝ := sorry

/-- The sum of the first n terms of a_n -/
def S (n : ℕ+) : ℝ := sorry

/-- The sequence b_n -/
def b (n : ℕ+) : ℝ := sorry

/-- The sum of the first n terms of b_n -/
def T (n : ℕ+) : ℝ := sorry

theorem quadratic_sequence_proof 
  (h1 : f 0 = 0)
  (h2 : ∀ x, deriv f x = 6 * x - 2)
  (h3 : ∀ n : ℕ+, S n = f n) :
  (∀ n : ℕ+, a n = 6 * n - 5) ∧
  (∀ m : ℝ, (∀ n : ℕ+, T n ≥ m / 20) ↔ m ≤ 60 / 7) :=
by sorry

end quadratic_sequence_proof_l1504_150451


namespace square_field_area_l1504_150414

/-- The area of a square field given the time and speed of a horse running around it -/
theorem square_field_area (time : ℝ) (speed : ℝ) : 
  time = 10 → speed = 12 → (time * speed / 4) ^ 2 = 900 := by sorry

end square_field_area_l1504_150414


namespace bird_nest_twigs_l1504_150475

theorem bird_nest_twigs (twigs_in_circle : ℕ) (additional_twigs_per_weave : ℕ) (twigs_still_needed : ℕ) :
  twigs_in_circle = 12 →
  additional_twigs_per_weave = 6 →
  twigs_still_needed = 48 →
  (twigs_in_circle * additional_twigs_per_weave - twigs_still_needed : ℚ) / (twigs_in_circle * additional_twigs_per_weave) = 1 / 3 :=
by sorry

end bird_nest_twigs_l1504_150475


namespace nectar_water_percentage_l1504_150431

/-- Given that 1.7 kg of nectar yields 1 kg of honey, and the honey contains 15% water,
    prove that the nectar contains 50% water. -/
theorem nectar_water_percentage :
  ∀ (nectar_weight honey_weight : ℝ) 
    (honey_water_percentage nectar_water_percentage : ℝ),
  nectar_weight = 1.7 →
  honey_weight = 1 →
  honey_water_percentage = 15 →
  nectar_water_percentage = 
    (nectar_weight * honey_water_percentage / 100 + (nectar_weight - honey_weight)) / 
    nectar_weight * 100 →
  nectar_water_percentage = 50 := by
  sorry

end nectar_water_percentage_l1504_150431


namespace tan_x_equals_sqrt_three_l1504_150449

theorem tan_x_equals_sqrt_three (x : ℝ) 
  (h : Real.sin (x + Real.pi / 9) = Real.cos (x + Real.pi / 18) + Real.cos (x - Real.pi / 18)) : 
  Real.tan x = Real.sqrt 3 := by
  sorry

end tan_x_equals_sqrt_three_l1504_150449


namespace clockwise_rotation_240_l1504_150491

/-- The angle formed by rotating a ray clockwise around its endpoint -/
def clockwise_rotation (angle : ℝ) : ℝ := -angle

/-- Theorem: The angle formed by rotating a ray 240° clockwise around its endpoint is -240° -/
theorem clockwise_rotation_240 : clockwise_rotation 240 = -240 := by
  sorry

end clockwise_rotation_240_l1504_150491


namespace lines_intersect_at_point_l1504_150419

def line1 (t : ℚ) : ℚ × ℚ := (2 + 3*t, 2 - 4*t)
def line2 (u : ℚ) : ℚ × ℚ := (4 + 5*u, -8 + 3*u)

def intersection_point : ℚ × ℚ := (-123/141, 454/141)

theorem lines_intersect_at_point :
  ∃ (t u : ℚ), line1 t = line2 u ∧ line1 t = intersection_point :=
by sorry

end lines_intersect_at_point_l1504_150419


namespace charlie_snowballs_l1504_150486

theorem charlie_snowballs (lucy_snowballs : ℕ) (charlie_snowballs : ℕ) : 
  lucy_snowballs = 19 → 
  charlie_snowballs = lucy_snowballs + 31 → 
  charlie_snowballs = 50 := by
  sorry

end charlie_snowballs_l1504_150486


namespace table_runner_coverage_l1504_150488

theorem table_runner_coverage (
  total_runner_area : ℝ)
  (table_area : ℝ)
  (two_layer_area : ℝ)
  (three_layer_area : ℝ)
  (h1 : total_runner_area = 212)
  (h2 : table_area = 175)
  (h3 : two_layer_area = 24)
  (h4 : three_layer_area = 24)
  : ∃ (coverage_percentage : ℝ),
    abs (coverage_percentage - 52.57) < 0.01 ∧
    coverage_percentage = (total_runner_area - 2 * two_layer_area - 3 * three_layer_area) / table_area * 100 := by
  sorry

end table_runner_coverage_l1504_150488


namespace intersection_when_m_is_one_intersection_equals_A_iff_l1504_150460

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | (x - 2*m + 1)*(x - m + 2) < 0}
def B : Set ℝ := {x | 1 ≤ x + 1 ∧ x + 1 ≤ 4}

-- Theorem 1: When m = 1, A ∩ B = {x | 0 ≤ x < 1}
theorem intersection_when_m_is_one :
  A 1 ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

-- Theorem 2: A ∩ B = A if and only if m ∈ {-1, 2}
theorem intersection_equals_A_iff :
  ∀ m : ℝ, A m ∩ B = A m ↔ m = -1 ∨ m = 2 := by sorry

end intersection_when_m_is_one_intersection_equals_A_iff_l1504_150460


namespace quadratic_equation_solution_l1504_150456

theorem quadratic_equation_solution :
  ∃! y : ℝ, y^2 + 6*y + 8 = -(y + 2)*(y + 6) :=
by
  -- The proof goes here
  sorry

end quadratic_equation_solution_l1504_150456


namespace lcd_of_fractions_l1504_150401

def fractions : List Nat := [3, 4, 5, 8, 9, 11]

theorem lcd_of_fractions : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 3 4) 5) 8) 9) 11 = 3960 := by
  sorry

end lcd_of_fractions_l1504_150401


namespace intersection_of_A_and_B_l1504_150467

def A : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def B : Set ℝ := {-1, 2, 3, 6}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end intersection_of_A_and_B_l1504_150467


namespace magical_stack_131_l1504_150403

/-- Definition of a magical stack -/
def is_magical_stack (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≤ n ∧ b > n ∧ b ≤ 2*n ∧
  (a = 2*a - 1 ∨ b = 2*(b - n))

/-- Theorem: A stack with 392 cards where card 131 retains its position is magical -/
theorem magical_stack_131 :
  ∃ (n : ℕ), 2*n = 392 ∧ is_magical_stack n ∧ 131 ≤ n ∧ 131 = 2*131 - 1 := by
  sorry

end magical_stack_131_l1504_150403


namespace interval_intersection_l1504_150446

theorem interval_intersection (x : ℝ) : 
  (2 < 3*x ∧ 3*x < 3) ∧ (2 < 4*x ∧ 4*x < 3) ↔ (2/3 < x ∧ x < 3/4) :=
by sorry

end interval_intersection_l1504_150446


namespace power_function_property_l1504_150493

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

theorem power_function_property (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = Real.sqrt 2 → f 3 = Real.sqrt 3 := by
  sorry

end power_function_property_l1504_150493


namespace candy_chocolate_difference_l1504_150487

theorem candy_chocolate_difference (initial_candy : ℕ) (additional_candy : ℕ) (chocolate : ℕ) :
  initial_candy = 38 →
  additional_candy = 36 →
  chocolate = 16 →
  (initial_candy + additional_candy) - chocolate = 58 := by
  sorry

end candy_chocolate_difference_l1504_150487


namespace fourth_power_divisor_count_l1504_150413

theorem fourth_power_divisor_count (n : ℕ+) : ∃ d : ℕ, 
  (∀ k : ℕ, k ∣ n^4 ↔ k ≤ d) ∧ d % 4 = 1 := by
  sorry

end fourth_power_divisor_count_l1504_150413


namespace coffee_mixture_cost_theorem_l1504_150400

/-- The cost of the more expensive coffee per pound -/
def expensive_coffee_cost : ℝ := 7.28

/-- The cost of the cheaper coffee per pound -/
def cheaper_coffee_cost : ℝ := 6.42

/-- The amount of cheaper coffee in pounds -/
def cheaper_coffee_amount : ℝ := 7

/-- The amount of expensive coffee in pounds -/
def expensive_coffee_amount : ℝ := 68.25

/-- The price of the mixture per pound -/
def mixture_price : ℝ := 7.20

/-- The total amount of coffee in the mixture -/
def total_coffee_amount : ℝ := cheaper_coffee_amount + expensive_coffee_amount

theorem coffee_mixture_cost_theorem :
  cheaper_coffee_amount * cheaper_coffee_cost +
  expensive_coffee_amount * expensive_coffee_cost =
  total_coffee_amount * mixture_price :=
by sorry

end coffee_mixture_cost_theorem_l1504_150400


namespace bird_ratio_l1504_150432

/-- Represents the number of birds caught by a cat during the day. -/
def birds_day : ℕ := 8

/-- Represents the total number of birds caught by a cat. -/
def birds_total : ℕ := 24

/-- Represents the number of birds caught by a cat at night. -/
def birds_night : ℕ := birds_total - birds_day

/-- The theorem states that the ratio of birds caught at night to birds caught during the day is 2:1. -/
theorem bird_ratio : birds_night / birds_day = 2 := by
  sorry

end bird_ratio_l1504_150432


namespace arccos_neg_sqrt3_div2_l1504_150492

theorem arccos_neg_sqrt3_div2 :
  Real.arccos (-Real.sqrt 3 / 2) = 5 * Real.pi / 6 := by
  sorry

end arccos_neg_sqrt3_div2_l1504_150492


namespace seedling_problem_l1504_150485

-- Define variables for seedling prices
variable (x y : ℚ)

-- Define conditions
def condition1 : Prop := 3 * x + 2 * y = 12
def condition2 : Prop := x + 3 * y = 11

-- Define total number of seedlings
def total_seedlings : ℕ := 200

-- Define value multiplier
def value_multiplier : ℕ := 100

-- Define minimum total value
def min_total_value : ℕ := 50000

-- Theorem to prove
theorem seedling_problem (h1 : condition1 x y) (h2 : condition2 x y) :
  x = 2 ∧ y = 3 ∧
  ∃ m : ℕ, m ≥ 100 ∧
  m ≤ total_seedlings ∧
  2 * value_multiplier * (total_seedlings - m) + 3 * value_multiplier * m ≥ min_total_value ∧
  ∀ n : ℕ, n < m →
    2 * value_multiplier * (total_seedlings - n) + 3 * value_multiplier * n < min_total_value :=
by
  sorry

end seedling_problem_l1504_150485


namespace min_value_expression_l1504_150489

theorem min_value_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 27 / n ≥ 6 ∧
  ((n : ℝ) / 3 + 27 / n = 6 ↔ n = 9) :=
by sorry

end min_value_expression_l1504_150489


namespace equation_solutions_l1504_150450

def satisfies_equation (a b c : ℤ) : Prop :=
  (abs (a + 3) : ℤ) + b^2 + 4*c^2 - 14*b - 12*c + 55 = 0

def solution_set : Set (ℤ × ℤ × ℤ) :=
  {(-2, 8, 2), (-2, 6, 2), (-4, 8, 2), (-4, 6, 2), (-1, 7, 2), (-1, 7, 1), (-5, 7, 2), (-5, 7, 1)}

theorem equation_solutions :
  ∀ (a b c : ℤ), satisfies_equation a b c ↔ (a, b, c) ∈ solution_set :=
by sorry

end equation_solutions_l1504_150450


namespace beef_cost_theorem_l1504_150440

/-- Calculates the total cost of beef given the number of packs, pounds per pack, and price per pound. -/
def total_cost (num_packs : ℕ) (pounds_per_pack : ℕ) (price_per_pound : ℚ) : ℚ :=
  (num_packs * pounds_per_pack : ℚ) * price_per_pound

/-- Proves that given 5 packs of beef, 4 pounds per pack, and a price of $5.50 per pound, the total cost is $110. -/
theorem beef_cost_theorem :
  total_cost 5 4 (11/2) = 110 := by
  sorry

end beef_cost_theorem_l1504_150440


namespace max_value_implies_ratio_l1504_150461

/-- Given a function f(x) = x^3 + ax^2 + bx - a^2 - 7a that reaches its maximum value of 10 at x = 1,
    prove that a/b = -2/3 -/
theorem max_value_implies_ratio (a b : ℝ) :
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x - a^2 - 7*a
  (∀ x, f x ≤ f 1) ∧ (f 1 = 10) → a/b = -2/3 := by
  sorry

end max_value_implies_ratio_l1504_150461


namespace parabola_reflection_difference_l1504_150479

/-- Represents a quadratic function of the form ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The function representing the original parabola translated up by 3 units --/
def f (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c + 3

/-- The function representing the reflected parabola translated down by 3 units --/
def g (p : Parabola) (x : ℝ) : ℝ :=
  -p.a * x^2 - p.b * x - p.c - 3

/-- Theorem stating that (f-g)(x) equals 2ax^2 + 2bx + 2c + 6 --/
theorem parabola_reflection_difference (p : Parabola) (x : ℝ) :
  f p x - g p x = 2 * p.a * x^2 + 2 * p.b * x + 2 * p.c + 6 := by
  sorry


end parabola_reflection_difference_l1504_150479


namespace existence_of_xy_l1504_150471

theorem existence_of_xy (f g : ℝ → ℝ) : ∃ x y : ℝ, 
  x ∈ Set.Icc 0 1 ∧ 
  y ∈ Set.Icc 0 1 ∧ 
  |f x + g y - x * y| ≥ 1/4 := by
  sorry

end existence_of_xy_l1504_150471


namespace smallest_total_books_l1504_150435

/-- Represents the number of books in the library -/
structure LibraryBooks where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Checks if the given LibraryBooks satisfies the required ratios -/
def satisfiesRatios (books : LibraryBooks) : Prop :=
  3 * books.chemistry = 2 * books.physics ∧ 
  4 * books.biology = 3 * books.chemistry

/-- Calculates the total number of books -/
def totalBooks (books : LibraryBooks) : ℕ :=
  books.physics + books.chemistry + books.biology

/-- Theorem: The smallest possible total number of books satisfying the conditions is 3003 -/
theorem smallest_total_books : 
  ∃ (books : LibraryBooks), 
    satisfiesRatios books ∧ 
    totalBooks books > 3000 ∧
    totalBooks books = 3003 ∧
    ∀ (other : LibraryBooks), 
      satisfiesRatios other → 
      totalBooks other > 3000 → 
      totalBooks other ≥ totalBooks books := by
  sorry

end smallest_total_books_l1504_150435


namespace snack_bar_employees_l1504_150495

theorem snack_bar_employees (total : ℕ) (buffet dining : ℕ) (two_restaurants all_restaurants : ℕ) : 
  total = 39 →
  buffet = 17 →
  dining = 18 →
  two_restaurants = 4 →
  all_restaurants = 2 →
  ∃ (snack_bar : ℕ), 
    snack_bar = total - (buffet + dining - two_restaurants - all_restaurants) := by
  sorry

end snack_bar_employees_l1504_150495


namespace equation_holds_for_three_l1504_150427

-- Define the equation we want to prove
def equation (n : ℕ) : Prop :=
  (((17 * Real.sqrt 5 + 38) ^ (1 / n : ℝ)) + ((17 * Real.sqrt 5 - 38) ^ (1 / n : ℝ))) = 2 * Real.sqrt 5

-- Theorem statement
theorem equation_holds_for_three : 
  equation 3 := by sorry

end equation_holds_for_three_l1504_150427


namespace negative_multiplication_result_l1504_150482

theorem negative_multiplication_result : (-4 : ℚ) * (-3/2 : ℚ) = 6 := by
  sorry

end negative_multiplication_result_l1504_150482


namespace lead_percentage_in_mixture_l1504_150463

/-- Proves that the percentage of lead in a mixture is 25% given the specified conditions -/
theorem lead_percentage_in_mixture
  (cobalt_percent : Real)
  (copper_percent : Real)
  (lead_weight : Real)
  (copper_weight : Real)
  (h1 : cobalt_percent = 0.15)
  (h2 : copper_percent = 0.60)
  (h3 : lead_weight = 5)
  (h4 : copper_weight = 12)
  : (lead_weight / (copper_weight / copper_percent)) * 100 = 25 := by
  sorry


end lead_percentage_in_mixture_l1504_150463


namespace grasshoppers_on_plant_count_l1504_150454

def total_grasshoppers : ℕ := 31
def baby_grasshoppers_dozens : ℕ := 2

def grasshoppers_on_plant : ℕ := total_grasshoppers - (baby_grasshoppers_dozens * 12)

theorem grasshoppers_on_plant_count : grasshoppers_on_plant = 7 := by
  sorry

end grasshoppers_on_plant_count_l1504_150454


namespace factorial_different_remainders_l1504_150415

theorem factorial_different_remainders (n : ℕ) : n ≥ 2 →
  (∀ i j, 1 ≤ i ∧ i < j ∧ j < n → Nat.factorial i % n ≠ Nat.factorial j % n) ↔ n = 2 ∨ n = 3 := by
  sorry

end factorial_different_remainders_l1504_150415


namespace chess_tournament_games_l1504_150457

/-- The number of games played in a chess tournament with n participants,
    where each participant plays exactly one game with each of the others. -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 19 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 171. -/
theorem chess_tournament_games :
  tournament_games 19 = 171 := by
  sorry

end chess_tournament_games_l1504_150457


namespace unique_representation_theorem_l1504_150404

-- Define a type for representing a person (boy or girl)
inductive Person : Type
| boy : Person
| girl : Person

-- Define a function to convert a natural number to a list of 5 binary digits
def toBinaryDigits (n : Nat) : List Bool :=
  List.reverse (List.take 5 (List.map (fun i => n / 2^i % 2 = 1) (List.range 5)))

-- Define a function to convert a list of binary digits to a list of persons
def binaryToPersons (bits : List Bool) : List Person :=
  List.map (fun b => if b then Person.boy else Person.girl) bits

-- Define a function to convert a list of persons back to a natural number
def personsToNumber (persons : List Person) : Nat :=
  List.foldl (fun acc p => 2 * acc + match p with
    | Person.boy => 1
    | Person.girl => 0) 0 persons

-- Theorem statement
theorem unique_representation_theorem (n : Nat) (h : n > 0 ∧ n ≤ 31) :
  ∃! (arrangement : List Person),
    arrangement.length = 5 ∧
    personsToNumber arrangement = n :=
  sorry

end unique_representation_theorem_l1504_150404


namespace sum_of_first_seven_primes_with_units_digit_3_l1504_150468

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def has_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def first_seven_primes_with_units_digit_3 : List ℕ := [3, 13, 23, 43, 53, 73, 83]

theorem sum_of_first_seven_primes_with_units_digit_3 :
  (∀ n ∈ first_seven_primes_with_units_digit_3, is_prime n ∧ has_units_digit_3 n) →
  (∀ p : ℕ, is_prime p → has_units_digit_3 p → 
    p ∉ first_seven_primes_with_units_digit_3 → 
    p > (List.maximum first_seven_primes_with_units_digit_3).getD 0) →
  List.sum first_seven_primes_with_units_digit_3 = 291 := by
sorry

end sum_of_first_seven_primes_with_units_digit_3_l1504_150468


namespace solution_set_of_inequality_l1504_150496

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem solution_set_of_inequality (f : ℝ → ℝ) (hf_odd : OddFunction f)
    (hf_2 : f 2 = 0) (hf_deriv : ∀ x > 0, (x * (deriv f x) - f x) / x^2 < 0) :
    {x : ℝ | x^2 * f x > 0} = {x : ℝ | x < -2 ∨ (0 < x ∧ x < 2)} := by
  sorry

end solution_set_of_inequality_l1504_150496


namespace quadratic_minimum_l1504_150497

theorem quadratic_minimum (c : ℝ) : 
  (1/3 : ℝ) * c^2 + 6*c + 4 ≥ (1/3 : ℝ) * (-9)^2 + 6*(-9) + 4 := by
  sorry

end quadratic_minimum_l1504_150497


namespace sum_factorials_mod_5_l1504_150499

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials : ℕ := 
  (factorial 1) + (factorial 2) + (factorial 3) + (factorial 4) + 
  (factorial 5) + (factorial 6) + (factorial 7) + (factorial 8) + 
  (factorial 9) + (factorial 10)

theorem sum_factorials_mod_5 : sum_factorials % 5 = 3 := by
  sorry

end sum_factorials_mod_5_l1504_150499


namespace investment_sum_l1504_150408

theorem investment_sum (raghu_investment : ℝ) : raghu_investment = 2400 →
  let trishul_investment := raghu_investment * 0.9
  let vishal_investment := trishul_investment * 1.1
  raghu_investment + trishul_investment + vishal_investment = 6936 := by
sorry

end investment_sum_l1504_150408


namespace arithmetic_sequence_sin_problem_l1504_150464

theorem arithmetic_sequence_sin_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 6 = 10 * Real.pi / 3 →                    -- given condition
  Real.sin (a 4 + a 7) = -Real.sqrt 3 / 2 :=        -- conclusion to prove
by
  sorry

end arithmetic_sequence_sin_problem_l1504_150464


namespace ratio_problem_l1504_150444

theorem ratio_problem (x y : ℝ) (h : (3 * x^2 - y) / (x + y) = 1/2) :
  x / y = 3 / (6 * x - 1) := by sorry

end ratio_problem_l1504_150444


namespace linear_function_increasing_y_l1504_150447

theorem linear_function_increasing_y (k b y₁ y₂ : ℝ) :
  k > 0 →
  y₁ = k * (-1) - b →
  y₂ = k * 2 - b →
  y₁ < y₂ := by
  sorry

end linear_function_increasing_y_l1504_150447


namespace price_changes_l1504_150490

/-- The original price of an item that, after a 5% decrease and a 40% increase,
    results in a price $1352.06 less than twice the original price. -/
def original_price : ℝ := 2018

theorem price_changes (x : ℝ) (hx : x = original_price) :
  let price_after_decrease := 0.95 * x
  let price_after_increase := price_after_decrease * 1.4
  price_after_increase = 2 * x - 1352.06 := by sorry

end price_changes_l1504_150490


namespace sqrt_x_plus_one_real_l1504_150428

theorem sqrt_x_plus_one_real (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 :=
by sorry

end sqrt_x_plus_one_real_l1504_150428

import Mathlib

namespace infinite_solutions_equation_l2645_264590

theorem infinite_solutions_equation :
  ∃ (x y z : ℕ → ℕ), ∀ n : ℕ,
    (x n)^2 + (x n + 1)^2 = (y n)^2 ∧
    z n = 2 * (x n) + 1 ∧
    (z n)^2 = 2 * (y n)^2 - 1 :=
by sorry

end infinite_solutions_equation_l2645_264590


namespace consecutive_natural_numbers_sum_l2645_264579

theorem consecutive_natural_numbers_sum (a : ℕ) : 
  (∃ (x : ℕ), x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 50) → 
  (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = 50) → 
  (a + 2 = 10) := by
  sorry

#check consecutive_natural_numbers_sum

end consecutive_natural_numbers_sum_l2645_264579


namespace luke_good_games_l2645_264500

theorem luke_good_games (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (non_working_games : ℕ) :
  games_from_friend = 2 →
  games_from_garage_sale = 2 →
  non_working_games = 2 →
  games_from_friend + games_from_garage_sale - non_working_games = 2 :=
by
  sorry

end luke_good_games_l2645_264500


namespace female_puppies_count_l2645_264570

theorem female_puppies_count (total : ℕ) (male : ℕ) (ratio : ℚ) : ℕ :=
  let female := total - male
  have h1 : total = 12 := by sorry
  have h2 : male = 10 := by sorry
  have h3 : ratio = 1/5 := by sorry
  have h4 : (female : ℚ) / male = ratio := by sorry
  2

#check female_puppies_count

end female_puppies_count_l2645_264570


namespace simplify_fraction_l2645_264503

theorem simplify_fraction : (75 : ℚ) / 225 = 1 / 3 := by
  sorry

end simplify_fraction_l2645_264503


namespace average_of_three_numbers_l2645_264575

theorem average_of_three_numbers (y : ℝ) : (15 + 25 + y) / 3 = 23 → y = 29 := by
  sorry

end average_of_three_numbers_l2645_264575


namespace sheepdog_speed_l2645_264554

/-- Proves that a sheepdog running at the specified speed can catch a sheep in the given time --/
theorem sheepdog_speed (sheep_speed : ℝ) (initial_distance : ℝ) (catch_time : ℝ) 
  (h1 : sheep_speed = 12)
  (h2 : initial_distance = 160)
  (h3 : catch_time = 20) :
  (initial_distance + sheep_speed * catch_time) / catch_time = 20 := by
  sorry

#check sheepdog_speed

end sheepdog_speed_l2645_264554


namespace ratio_c_d_equals_two_thirds_l2645_264511

theorem ratio_c_d_equals_two_thirds
  (x y c d : ℝ)
  (h1 : 8 * x - 5 * y = c)
  (h2 : 10 * y - 12 * x = d)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hd : d ≠ 0) :
  c / d = 2 / 3 := by
sorry

end ratio_c_d_equals_two_thirds_l2645_264511


namespace trapezoid_perimeter_l2645_264599

/-- A trapezoid with specific side lengths -/
structure Trapezoid where
  EG : ℝ
  FH : ℝ
  GH : ℝ
  EF : ℝ
  is_trapezoid : EF > GH
  parallel_bases : EF = 2 * GH

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.EG + t.FH + t.GH + t.EF

/-- Theorem: The perimeter of the given trapezoid is 183 units -/
theorem trapezoid_perimeter :
  ∃ t : Trapezoid, t.EG = 35 ∧ t.FH = 40 ∧ t.GH = 36 ∧ perimeter t = 183 := by
  sorry

end trapezoid_perimeter_l2645_264599


namespace set_equals_interval_l2645_264520

-- Define the set S as {x | -1 < x ≤ 3}
def S : Set ℝ := {x | -1 < x ∧ x ≤ 3}

-- Define the interval (-1,3]
def I : Set ℝ := Set.Ioc (-1) 3

-- Theorem statement
theorem set_equals_interval : S = I := by sorry

end set_equals_interval_l2645_264520


namespace sin_negative_1560_degrees_l2645_264552

theorem sin_negative_1560_degrees : Real.sin ((-1560 : ℝ) * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_negative_1560_degrees_l2645_264552


namespace twelve_factorial_mod_thirteen_l2645_264574

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem twelve_factorial_mod_thirteen : 
  factorial 12 % 13 = 12 := by sorry

end twelve_factorial_mod_thirteen_l2645_264574


namespace permutation_inequalities_l2645_264572

/-- Given a set X of n elements and 0 ≤ k ≤ n, a_{n,k} is the maximum number of permutations
    acting on X such that every two of them have at least k components in common -/
def a (n k : ℕ) : ℕ := sorry

/-- Given a set X of n elements and 0 ≤ k ≤ n, b_{n,k} is the maximum number of permutations
    acting on X such that every two of them have at most k components in common -/
def b (n k : ℕ) : ℕ := sorry

theorem permutation_inequalities (n k : ℕ) (h : k ≤ n) :
  a n k * b n (k - 1) ≤ n! ∧ ∀ p : ℕ, Nat.Prime p → a p 2 = p! / 2 := by
  sorry

end permutation_inequalities_l2645_264572


namespace squared_difference_of_quadratic_roots_l2645_264502

theorem squared_difference_of_quadratic_roots : ∀ p q : ℝ,
  (2 * p^2 + 7 * p - 15 = 0) →
  (2 * q^2 + 7 * q - 15 = 0) →
  (p - q)^2 = 169 / 4 := by
  sorry

end squared_difference_of_quadratic_roots_l2645_264502


namespace octal_computation_l2645_264513

/-- Converts a decimal number to its octal representation -/
def toOctal (n : ℕ) : ℕ := sorry

/-- Multiplies two octal numbers -/
def octalMultiply (a b : ℕ) : ℕ := sorry

/-- Divides an octal number by another octal number -/
def octalDivide (a b : ℕ) : ℕ := sorry

theorem octal_computation : 
  let a := toOctal 254
  let b := toOctal 170
  let c := toOctal 4
  octalDivide (octalMultiply a b) c = 3156 := by sorry

end octal_computation_l2645_264513


namespace solution_set_f_gt_5_range_of_a_l2645_264539

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- Theorem for the solution set of f(x) > 5
theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -3 ∨ x > 2} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 - 2*a) → -1 ≤ a ∧ a ≤ 3 := by sorry

end solution_set_f_gt_5_range_of_a_l2645_264539


namespace subset_sum_property_l2645_264536

theorem subset_sum_property (n : ℕ) (hn : n > 1) :
  ∀ (S : Finset ℕ), S ⊆ Finset.range (2 * n) → S.card = n + 2 →
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b = c :=
by sorry

end subset_sum_property_l2645_264536


namespace molecular_weight_is_265_21_l2645_264550

/-- Atomic weight of Aluminium in amu -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Oxygen in amu -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen in amu -/
def H_weight : ℝ := 1.01

/-- Atomic weight of Silicon in amu -/
def Si_weight : ℝ := 28.09

/-- Atomic weight of Nitrogen in amu -/
def N_weight : ℝ := 14.01

/-- Number of Aluminium atoms in the compound -/
def Al_count : ℕ := 2

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 6

/-- Number of Hydrogen atoms in the compound -/
def H_count : ℕ := 3

/-- Number of Silicon atoms in the compound -/
def Si_count : ℕ := 2

/-- Number of Nitrogen atoms in the compound -/
def N_count : ℕ := 4

/-- The molecular weight of the compound -/
def molecular_weight : ℝ :=
  Al_count * Al_weight + O_count * O_weight + H_count * H_weight +
  Si_count * Si_weight + N_count * N_weight

theorem molecular_weight_is_265_21 : molecular_weight = 265.21 := by
  sorry

end molecular_weight_is_265_21_l2645_264550


namespace money_problem_l2645_264598

theorem money_problem (a b : ℚ) : 
  (4 * a + 2 * b = 92) ∧ (6 * a - 4 * b = 60) → 
  (a = 122 / 7) ∧ (b = 78 / 7) := by
  sorry

end money_problem_l2645_264598


namespace number_of_boys_l2645_264526

theorem number_of_boys (total : ℕ) (boys : ℕ) : 
  total = 900 →
  (total - boys : ℚ) = (boys : ℚ) * (total : ℚ) / 100 →
  boys = 90 := by
sorry

end number_of_boys_l2645_264526


namespace function_composition_properties_l2645_264585

theorem function_composition_properties :
  (¬ ∃ (f g : ℝ → ℝ), ∀ x, f (g x) = x^2 ∧ g (f x) = x^3) ∧
  (∃ (f g : ℝ → ℝ), ∀ x, f (g x) = x^2 ∧ g (f x) = x^4) := by
  sorry

end function_composition_properties_l2645_264585


namespace remainder_problem_l2645_264538

theorem remainder_problem (D : ℕ) (h1 : D = 13) (h2 : 698 % D = 9) (h3 : (242 + 698) % D = 4) :
  242 % D = 8 := by
  sorry

end remainder_problem_l2645_264538


namespace expected_attacked_squares_theorem_l2645_264593

/-- The number of squares on a chessboard. -/
def chessboardSize : ℕ := 64

/-- The number of rooks placed on the chessboard. -/
def numberOfRooks : ℕ := 3

/-- The probability that a specific square is not attacked by a single rook. -/
def probNotAttackedByOneRook : ℚ := 49 / 64

/-- The expected number of squares under attack when three rooks are randomly placed on a chessboard. -/
def expectedAttackedSquares : ℚ :=
  chessboardSize * (1 - probNotAttackedByOneRook ^ numberOfRooks)

/-- Theorem stating that the expected number of squares under attack is equal to the calculated value. -/
theorem expected_attacked_squares_theorem :
  expectedAttackedSquares = 64 * (1 - (49/64)^3) :=
sorry

end expected_attacked_squares_theorem_l2645_264593


namespace equation_solution_l2645_264543

theorem equation_solution : ∃! y : ℝ, (128 : ℝ) ^ (y + 1) / (8 : ℝ) ^ (y + 1) = (64 : ℝ) ^ (3 * y - 2) ∧ y = 8 / 7 := by
  sorry

end equation_solution_l2645_264543


namespace smallest_n_square_cube_l2645_264588

theorem smallest_n_square_cube : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 → 
    (∃ (y : ℕ), 4 * x = y^2) → 
    (∃ (z : ℕ), 5 * x = z^3) → 
    n ≤ x) ∧
  n = 25 :=
sorry

end smallest_n_square_cube_l2645_264588


namespace exists_158_consecutive_not_div_17_exists_div_17_in_159_consecutive_l2645_264583

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem 1: There exists a sequence of 158 consecutive integers where the sum of digits of each number is not divisible by 17
theorem exists_158_consecutive_not_div_17 : 
  ∃ (start : ℕ), ∀ (i : ℕ), i < 158 → ¬(17 ∣ sum_of_digits (start + i)) :=
sorry

-- Theorem 2: For any sequence of 159 consecutive integers, there exists at least one integer in the sequence whose sum of digits is divisible by 17
theorem exists_div_17_in_159_consecutive (start : ℕ) : 
  ∃ (i : ℕ), i < 159 ∧ (17 ∣ sum_of_digits (start + i)) :=
sorry

end exists_158_consecutive_not_div_17_exists_div_17_in_159_consecutive_l2645_264583


namespace smallest_n_for_coloring_property_l2645_264545

def is_valid_coloring (n : ℕ) (coloring : ℕ → Bool) : Prop :=
  ∀ x y z w, x ≤ n ∧ y ≤ n ∧ z ≤ n ∧ w ≤ n →
    coloring x = coloring y ∧ coloring y = coloring z ∧ coloring z = coloring w →
    x + y + z ≠ w

theorem smallest_n_for_coloring_property : 
  (∀ n < 11, ∃ coloring, is_valid_coloring n coloring) ∧
  (∀ coloring, ¬ is_valid_coloring 11 coloring) :=
sorry

end smallest_n_for_coloring_property_l2645_264545


namespace debate_team_max_groups_l2645_264553

/-- Given a debate team with boys and girls, calculate the maximum number of groups
    that can be formed with a minimum number of boys and girls per group. -/
def max_groups (num_boys num_girls min_boys_per_group min_girls_per_group : ℕ) : ℕ :=
  min (num_boys / min_boys_per_group) (num_girls / min_girls_per_group)

/-- Theorem stating that for a debate team with 31 boys and 32 girls,
    where each group must have at least 2 boys and 3 girls,
    the maximum number of groups that can be formed is 10. -/
theorem debate_team_max_groups :
  max_groups 31 32 2 3 = 10 := by
  sorry

end debate_team_max_groups_l2645_264553


namespace locus_of_T_is_tangents_to_C_perp_to_L_l2645_264532

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Represents a point in the plane -/
def Point := ℝ × ℝ

/-- The fixed circle C -/
def C : Circle := sorry

/-- The line L passing through the center of C -/
def L : Line := sorry

/-- A variable point P on L -/
def P : Point := sorry

/-- The circle K centered at P and passing through the center of C -/
def K : Circle := sorry

/-- A point T on K where a common tangent to C and K meets K -/
def T : Point := sorry

/-- The locus of point T -/
def locus_of_T : Set Point := sorry

/-- The pair of tangents to C which are perpendicular to L -/
def tangents_to_C_perp_to_L : Set Point := sorry

theorem locus_of_T_is_tangents_to_C_perp_to_L :
  locus_of_T = tangents_to_C_perp_to_L := by sorry

end locus_of_T_is_tangents_to_C_perp_to_L_l2645_264532


namespace log_a_equals_three_l2645_264518

theorem log_a_equals_three (a : ℝ) (h1 : a > 0) (h2 : a^(2/3) = 4/9) : 
  Real.log a / Real.log (2/3) = 3 := by
  sorry

end log_a_equals_three_l2645_264518


namespace cubic_common_roots_l2645_264563

theorem cubic_common_roots (a b : ℝ) : 
  (∃ r s : ℝ, r ≠ s ∧ 
    r^3 + a*r^2 + 17*r + 12 = 0 ∧ 
    r^3 + b*r^2 + 23*r + 15 = 0 ∧
    s^3 + a*s^2 + 17*s + 12 = 0 ∧ 
    s^3 + b*s^2 + 23*s + 15 = 0) →
  a = -10 ∧ b = -11 := by
sorry

end cubic_common_roots_l2645_264563


namespace min_shared_side_length_l2645_264565

/-- Given two triangles EFG and HFG sharing side FG, with specified side lengths,
    prove that the smallest possible integral value for FG is 15. -/
theorem min_shared_side_length (EF HG : ℝ) (EG HF : ℝ) : EF = 6 → EG = 15 → HG = 10 → HF = 25 →
  ∃ (FG : ℕ), FG = 15 ∧ ∀ (x : ℕ), (x : ℝ) > EG - EF ∧ (x : ℝ) > HF - HG → x ≥ FG :=
by sorry

end min_shared_side_length_l2645_264565


namespace triangle_inequality_in_necklace_l2645_264517

theorem triangle_inequality_in_necklace :
  ∀ (a : ℕ → ℕ),
  (∀ n, 290 ≤ a n ∧ a n ≤ 2023) →
  (∀ m n, m ≠ n → a m ≠ a n) →
  ∃ i, a i + a (i + 1) > a (i + 2) ∧
       a i + a (i + 2) > a (i + 1) ∧
       a (i + 1) + a (i + 2) > a i :=
by sorry

end triangle_inequality_in_necklace_l2645_264517


namespace series_growth_l2645_264577

theorem series_growth (n : ℕ) (h : n > 1) :
  (Finset.range (2^(n+1) - 1)).card - (Finset.range (2^n - 1)).card = 2^n :=
sorry

end series_growth_l2645_264577


namespace power_tower_mod_500_l2645_264567

theorem power_tower_mod_500 : 7^(7^(7^7)) ≡ 343 [ZMOD 500] := by sorry

end power_tower_mod_500_l2645_264567


namespace fraction_equality_l2645_264528

theorem fraction_equality (X : ℚ) : (2/5 : ℚ) * (5/9 : ℚ) * X = 0.11111111111111112 → X = 1/2 := by
  sorry

end fraction_equality_l2645_264528


namespace myrtle_hens_eggs_per_day_l2645_264592

/-- The number of eggs laid by Myrtle's hens -/
theorem myrtle_hens_eggs_per_day :
  ∀ (num_hens : ℕ) (days_gone : ℕ) (eggs_taken : ℕ) (eggs_dropped : ℕ) (eggs_remaining : ℕ),
  num_hens = 3 →
  days_gone = 7 →
  eggs_taken = 12 →
  eggs_dropped = 5 →
  eggs_remaining = 46 →
  ∃ (eggs_per_hen_per_day : ℕ),
    eggs_per_hen_per_day = 3 ∧
    num_hens * eggs_per_hen_per_day * days_gone - eggs_taken - eggs_dropped = eggs_remaining :=
by
  sorry


end myrtle_hens_eggs_per_day_l2645_264592


namespace line_equation_l2645_264589

/-- Given a line y = kx + b passing through points (-1, 0) and (0, 3),
    prove that its equation is y = 3x + 3 -/
theorem line_equation (k b : ℝ) : 
  (k * (-1) + b = 0) → (b = 3) → (k = 3 ∧ b = 3) := by
  sorry

end line_equation_l2645_264589


namespace regular_polygon_sides_l2645_264529

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (180 - 360 / n : ℝ) = 160 → n = 18 := by
  sorry

end regular_polygon_sides_l2645_264529


namespace house_glass_panels_l2645_264548

/-- The number of glass panels per window -/
def panels_per_window : ℕ := 4

/-- The number of double windows downstairs -/
def double_windows_downstairs : ℕ := 6

/-- The number of single windows upstairs -/
def single_windows_upstairs : ℕ := 8

/-- The total number of glass panels in the house -/
def total_panels : ℕ := panels_per_window * (2 * double_windows_downstairs + single_windows_upstairs)

theorem house_glass_panels :
  total_panels = 80 :=
by sorry

end house_glass_panels_l2645_264548


namespace bus_profit_analysis_l2645_264522

/-- Represents the monthly profit of a bus service -/
def monthly_profit (passengers : ℕ) : ℤ :=
  2 * passengers - 4000

theorem bus_profit_analysis :
  let break_even := 2000
  let profit_4230 := monthly_profit 4230
  -- 1. Independent variable is passengers, dependent is profit (implicit in the function definition)
  -- 2. Break-even point
  monthly_profit break_even = 0 ∧
  -- 3. Profit for 4230 passengers
  profit_4230 = 4460 := by
  sorry

end bus_profit_analysis_l2645_264522


namespace bulb_replacement_probabilities_l2645_264594

/-- Represents the probability of a bulb lasting more than a given number of years -/
def bulb_survival_prob (years : ℕ) : ℝ :=
  match years with
  | 1 => 0.8
  | 2 => 0.3
  | _ => 0 -- Assuming 0 probability for other years

/-- The number of lamps in the conference room -/
def num_lamps : ℕ := 3

/-- Calculates the probability of not replacing any bulbs during the first replacement -/
def prob_no_replace : ℝ := (bulb_survival_prob 1) ^ num_lamps

/-- Calculates the probability of replacing exactly 2 bulbs during the first replacement -/
def prob_replace_two : ℝ := 
  (Nat.choose num_lamps 2 : ℝ) * (bulb_survival_prob 1) * (1 - bulb_survival_prob 1)^2

/-- Calculates the probability that a bulb needs to be replaced during the second replacement -/
def prob_replace_second : ℝ := 
  (1 - bulb_survival_prob 1)^2 + (bulb_survival_prob 1) * (1 - bulb_survival_prob 2)

theorem bulb_replacement_probabilities :
  (prob_no_replace = 0.512) ∧
  (prob_replace_two = 0.096) ∧
  (prob_replace_second = 0.6) :=
sorry

end bulb_replacement_probabilities_l2645_264594


namespace log_sum_equals_zero_l2645_264581

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The main theorem -/
theorem log_sum_equals_zero
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_prod : a 3 * a 5 * a 7 = 1) :
  Real.log (a 1) + Real.log (a 9) = 0 :=
sorry

end log_sum_equals_zero_l2645_264581


namespace village_foods_monthly_sales_l2645_264549

/-- Represents the monthly sales data for Village Foods --/
structure VillageFoodsSales where
  customers : ℕ
  lettucePerCustomer : ℕ
  lettucePrice : ℚ
  tomatoesPerCustomer : ℕ
  tomatoPrice : ℚ

/-- Calculates the total monthly sales from lettuce and tomatoes --/
def totalMonthlySales (s : VillageFoodsSales) : ℚ :=
  s.customers * (s.lettucePerCustomer * s.lettucePrice + s.tomatoesPerCustomer * s.tomatoPrice)

/-- Theorem stating that the total monthly sales for the given conditions is $2000 --/
theorem village_foods_monthly_sales :
  let sales := VillageFoodsSales.mk 500 2 1 4 (1/2)
  totalMonthlySales sales = 2000 := by sorry

end village_foods_monthly_sales_l2645_264549


namespace sequence_periodicity_l2645_264595

def sequence_rule (a : ℕ) (u : ℕ → ℕ) : Prop :=
  ∀ n, (Even (u n) → u (n + 1) = (u n) / 2) ∧
       (Odd (u n) → u (n + 1) = a + u n)

theorem sequence_periodicity (a : ℕ) (u : ℕ → ℕ) 
  (h1 : Odd a) 
  (h2 : a > 0) 
  (h3 : sequence_rule a u) :
  ∃ k : ℕ, ∃ p : ℕ, p > 0 ∧ ∀ n ≥ k, u (n + p) = u n :=
sorry

end sequence_periodicity_l2645_264595


namespace plan1_more_profitable_l2645_264556

/-- Represents the monthly production and profit of a factory with two wastewater treatment plans -/
structure FactoryProduction where
  x : ℕ  -- Number of products produced per month
  y1 : ℤ -- Monthly profit for Plan 1 in yuan
  y2 : ℤ -- Monthly profit for Plan 2 in yuan

/-- Calculates the monthly profit for Plan 1 -/
def plan1Profit (x : ℕ) : ℤ :=
  24 * x - 30000

/-- Calculates the monthly profit for Plan 2 -/
def plan2Profit (x : ℕ) : ℤ :=
  18 * x

/-- Theorem stating that Plan 1 yields more profit when producing 6000 products per month -/
theorem plan1_more_profitable :
  let production : FactoryProduction := {
    x := 6000,
    y1 := plan1Profit 6000,
    y2 := plan2Profit 6000
  }
  production.y1 > production.y2 :=
by sorry

end plan1_more_profitable_l2645_264556


namespace complex_number_absolute_value_squared_l2645_264566

theorem complex_number_absolute_value_squared (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  (z + Complex.abs z = 1 + 12 * Complex.I) → Complex.abs z ^ 2 = 5256 := by
  sorry

end complex_number_absolute_value_squared_l2645_264566


namespace f_and_g_properties_l2645_264547

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1| - |1 - x|
def g (a b x : ℝ) : ℝ := |x + a^2| + |x - b^2|

-- State the theorem
theorem f_and_g_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x, x ∈ {x : ℝ | f x ≥ 1} ↔ x ∈ Set.Ici (1/2)) ∧
  (∀ x, f x ≤ g a b x) := by
  sorry

end f_and_g_properties_l2645_264547


namespace arithmetic_progression_result_l2645_264508

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℕ → ℝ  -- The nth term of the progression
  S : ℕ → ℝ  -- The sum of the first n terms

/-- Theorem stating the result for the given arithmetic progression -/
theorem arithmetic_progression_result (ap : ArithmeticProgression) 
  (h1 : ap.a 1 + ap.a 3 = 5)
  (h2 : ap.S 4 = 20) :
  (ap.S 8 - 2 * ap.S 4) / (ap.S 6 - ap.S 4 - ap.S 2) = 10 := by
  sorry

end arithmetic_progression_result_l2645_264508


namespace tom_shares_problem_l2645_264515

theorem tom_shares_problem (initial_cost : ℕ) (sold_shares : ℕ) (sold_price : ℕ) (total_profit : ℕ) :
  initial_cost = 3 →
  sold_shares = 10 →
  sold_price = 4 →
  total_profit = 40 →
  ∃ (initial_shares : ℕ), 
    initial_shares = sold_shares ∧
    sold_shares * (sold_price - initial_cost) = total_profit :=
by sorry

end tom_shares_problem_l2645_264515


namespace min_white_surface_area_l2645_264591

/-- Represents a cube with some faces painted gray and others white -/
structure PaintedCube where
  grayFaces : Fin 6 → Bool

/-- The number of identical structures -/
def numStructures : Nat := 7

/-- The number of cubes in each structure -/
def cubesPerStructure : Nat := 8

/-- The number of additional white cubes -/
def additionalWhiteCubes : Nat := 8

/-- The edge length of each small cube in cm -/
def smallCubeEdgeLength : ℝ := 1

/-- The total number of cubes used to construct the large cube -/
def totalCubes : Nat := numStructures * cubesPerStructure + additionalWhiteCubes

/-- The edge length of the large cube in terms of small cubes -/
def largeCubeEdgeLength : Nat := 4

/-- The surface area of the large cube in cm² -/
def largeCubeSurfaceArea : ℝ := 6 * (largeCubeEdgeLength * largeCubeEdgeLength : ℝ) * smallCubeEdgeLength ^ 2

/-- A function to calculate the maximum possible gray surface area -/
def maxGraySurfaceArea : ℝ := 84

/-- Theorem stating that the minimum white surface area is 12 cm² -/
theorem min_white_surface_area :
  largeCubeSurfaceArea - maxGraySurfaceArea = 12 := by sorry

end min_white_surface_area_l2645_264591


namespace monkey_climbing_l2645_264586

/-- Monkey's tree climbing problem -/
theorem monkey_climbing (tree_height : ℝ) (hop_distance : ℝ) (total_time : ℕ) 
  (h1 : tree_height = 21)
  (h2 : hop_distance = 3)
  (h3 : total_time = 19) :
  ∃ (slip_distance : ℝ), 
    slip_distance = 2 ∧ 
    (hop_distance - slip_distance) * (total_time - 1 : ℝ) + hop_distance = tree_height :=
by sorry

end monkey_climbing_l2645_264586


namespace grandmothers_current_age_l2645_264562

/-- The age of Minyoung's grandmother this year, given that Minyoung is 7 years old this year
    and her grandmother turns 65 when Minyoung turns 10. -/
def grandmothers_age (minyoung_age : ℕ) (grandmother_future_age : ℕ) (years_until_future : ℕ) : ℕ :=
  grandmother_future_age - years_until_future

/-- Proof that Minyoung's grandmother is 62 years old this year. -/
theorem grandmothers_current_age :
  grandmothers_age 7 65 3 = 62 := by
  sorry

end grandmothers_current_age_l2645_264562


namespace barney_towel_shortage_l2645_264582

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents Barney's towel situation -/
structure TowelSituation where
  totalTowels : ℕ
  towelsPerDay : ℕ
  extraTowelsUsed : ℕ
  expectedGuests : ℕ

/-- Calculates the number of days without clean towels -/
def daysWithoutCleanTowels (s : TowelSituation) : ℕ :=
  daysInWeek

/-- Theorem stating that Barney will not have clean towels for 7 days -/
theorem barney_towel_shortage (s : TowelSituation)
  (h1 : s.totalTowels = 18)
  (h2 : s.towelsPerDay = 2)
  (h3 : s.extraTowelsUsed = 5)
  (h4 : s.expectedGuests = 3) :
  daysWithoutCleanTowels s = daysInWeek :=
by sorry

#check barney_towel_shortage

end barney_towel_shortage_l2645_264582


namespace telethon_total_money_telethon_specific_case_l2645_264527

/-- Calculates the total money raised in a telethon with varying hourly rates --/
theorem telethon_total_money (first_period_hours : ℕ) (second_period_hours : ℕ) 
  (first_period_rate : ℕ) (rate_increase_percent : ℕ) : ℕ :=
  let total_hours := first_period_hours + second_period_hours
  let first_period_total := first_period_hours * first_period_rate
  let second_period_rate := first_period_rate + (first_period_rate * rate_increase_percent / 100)
  let second_period_total := second_period_hours * second_period_rate
  first_period_total + second_period_total

/-- Proves that the telethon raises $144,000 given the specific conditions --/
theorem telethon_specific_case : 
  telethon_total_money 12 14 5000 20 = 144000 := by
  sorry

end telethon_total_money_telethon_specific_case_l2645_264527


namespace waiter_problem_l2645_264571

/-- Given an initial number of customers and two groups of customers leaving,
    calculate the final number of customers remaining. -/
def remaining_customers (initial : ℝ) (first_group : ℝ) (second_group : ℝ) : ℝ :=
  initial - first_group - second_group

/-- Theorem stating that for the given problem, the number of remaining customers is 3.0 -/
theorem waiter_problem (initial : ℝ) (first_group : ℝ) (second_group : ℝ)
    (h1 : initial = 36.0)
    (h2 : first_group = 19.0)
    (h3 : second_group = 14.0) :
    remaining_customers initial first_group second_group = 3.0 := by
  sorry

end waiter_problem_l2645_264571


namespace sum_of_constants_l2645_264537

/-- Given constants a, b, and c satisfying the conditions, prove that a + 2b + 3c = 65 -/
theorem sum_of_constants (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ x < -4 ∨ |x - 25| ≤ 2)
  (h2 : a < b) : 
  a + 2*b + 3*c = 65 := by
  sorry

end sum_of_constants_l2645_264537


namespace pairwise_products_equal_differences_impossible_l2645_264509

theorem pairwise_products_equal_differences_impossible
  (a b c d : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_order : a < b ∧ b < c ∧ c < d)
  (h_product_order : a * b < a * c ∧ a * c < a * d ∧ a * d < b * c ∧ b * c < b * d ∧ b * d < c * d) :
  ¬∃ k : ℝ, k > 0 ∧
    a * c - a * b = k ∧
    a * d - a * c = k ∧
    b * c - a * d = k ∧
    b * d - b * c = k ∧
    c * d - b * d = k :=
by sorry

end pairwise_products_equal_differences_impossible_l2645_264509


namespace half_squared_is_quarter_l2645_264534

theorem half_squared_is_quarter : (1/2)^2 = 0.25 := by
  sorry

end half_squared_is_quarter_l2645_264534


namespace divisibility_equivalence_l2645_264569

theorem divisibility_equivalence (a b c : ℕ) (h : c ≥ 1) :
  a ∣ b ↔ a^c ∣ b^c := by sorry

end divisibility_equivalence_l2645_264569


namespace percentage_of_employed_females_l2645_264531

-- Define the given percentages
def total_employed_percent : ℝ := 64
def employed_males_percent : ℝ := 46

-- Define the theorem
theorem percentage_of_employed_females :
  (total_employed_percent - employed_males_percent) / total_employed_percent * 100 = 28.125 :=
by sorry

end percentage_of_employed_females_l2645_264531


namespace circle_equation_l2645_264596

/-- The equation (x - 3)^2 + (y + 4)^2 = 9 represents a circle centered at (3, -4) with radius 3 -/
theorem circle_equation (x y : ℝ) : 
  (x - 3)^2 + (y + 4)^2 = 9 ↔ 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (3, -4) ∧ 
    radius = 3 ∧ 
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_equation_l2645_264596


namespace monkey_climb_l2645_264505

/-- The height of the tree climbed by the monkey -/
def tree_height : ℕ := 21

/-- The net progress of the monkey per hour -/
def net_progress_per_hour : ℕ := 1

/-- The time taken by the monkey to reach the top of the tree -/
def total_hours : ℕ := 19

/-- The distance the monkey hops up in the last hour -/
def last_hop : ℕ := 3

theorem monkey_climb :
  tree_height = net_progress_per_hour * (total_hours - 1) + last_hop := by
  sorry

end monkey_climb_l2645_264505


namespace laptop_price_exceeds_savings_l2645_264533

/-- Proves that for any initial laptop price greater than 0, 
    after 2 years of 6% annual price increase, 
    the laptop price will exceed 56358 rubles -/
theorem laptop_price_exceeds_savings (P₀ : ℝ) (h : P₀ > 0) : 
  P₀ * (1 + 0.06)^2 > 56358 := by
  sorry

#check laptop_price_exceeds_savings

end laptop_price_exceeds_savings_l2645_264533


namespace integer_roots_of_polynomial_l2645_264544

def f (x : ℤ) : ℤ := x^3 - 4*x^2 - 14*x + 24

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, f x = 0 ↔ x = -1 ∨ x = 3 ∨ x = 4 :=
by sorry

end integer_roots_of_polynomial_l2645_264544


namespace function_upper_bound_implies_parameter_range_l2645_264558

theorem function_upper_bound_implies_parameter_range 
  (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1) →
  (∀ x, f x = Real.sin x ^ 2 + a * Real.cos x + a) →
  a ∈ Set.Iic 0 := by
  sorry

end function_upper_bound_implies_parameter_range_l2645_264558


namespace average_expenditure_for_week_l2645_264576

/-- The average expenditure for a week given the average expenditures for two parts of the week -/
theorem average_expenditure_for_week 
  (avg_first_3_days : ℝ) 
  (avg_next_4_days : ℝ) 
  (h1 : avg_first_3_days = 350)
  (h2 : avg_next_4_days = 420) :
  (3 * avg_first_3_days + 4 * avg_next_4_days) / 7 = 390 := by
  sorry

#check average_expenditure_for_week

end average_expenditure_for_week_l2645_264576


namespace rackets_sold_l2645_264568

/-- Given the total sales and average price of ping pong rackets, 
    prove the number of pairs sold. -/
theorem rackets_sold (total_sales : ℝ) (avg_price : ℝ) 
  (h1 : total_sales = 686)
  (h2 : avg_price = 9.8) : 
  total_sales / avg_price = 70 := by
  sorry

end rackets_sold_l2645_264568


namespace solution_set_implies_k_empty_solution_set_implies_k_range_l2645_264510

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2*x + 3*k

-- Part 1
theorem solution_set_implies_k (k : ℝ) :
  (∀ x, f k x < 0 ↔ x < -3 ∨ x > -1) → k = -1/2 := by sorry

-- Part 2
theorem empty_solution_set_implies_k_range (k : ℝ) :
  (∀ x, ¬(f k x < 0)) → 0 < k ∧ k ≤ Real.sqrt 3 / 3 := by sorry

end solution_set_implies_k_empty_solution_set_implies_k_range_l2645_264510


namespace only_four_solutions_l2645_264516

/-- A digit is a natural number from 0 to 9. -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- Convert a repeating decimal 0.aaaaa... to a fraction a/9. -/
def repeatingDecimalToFraction (a : Digit) : ℚ := a.val / 9

/-- The property that a pair of digits (a,b) satisfies √(0.aaaaa...) = 0.bbbbb... -/
def SatisfiesEquation (a b : Digit) : Prop :=
  (repeatingDecimalToFraction b) ^ 2 = repeatingDecimalToFraction a

/-- The theorem stating that only four specific digit pairs satisfy the equation. -/
theorem only_four_solutions :
  ∀ a b : Digit, SatisfiesEquation a b ↔ 
    ((a.val = 0 ∧ b.val = 0) ∨
     (a.val = 1 ∧ b.val = 3) ∨
     (a.val = 4 ∧ b.val = 6) ∨
     (a.val = 9 ∧ b.val = 9)) :=
by sorry

end only_four_solutions_l2645_264516


namespace five_digit_divisible_by_36_l2645_264535

def is_divisible_by_36 (n : ℕ) : Prop := n % 36 = 0

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def form_number (a b : ℕ) : ℕ := 90000 + 1000 * a + 650 + b

theorem five_digit_divisible_by_36 :
  ∀ a b : ℕ,
    is_single_digit a →
    is_single_digit b →
    is_divisible_by_36 (form_number a b) →
    ((a = 5 ∧ b = 2) ∨ (a = 1 ∧ b = 6)) :=
sorry

end five_digit_divisible_by_36_l2645_264535


namespace parabola_intersection_l2645_264530

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 18
def g (x : ℝ) : ℝ := x^2 - 2 * x + 4

-- Define the intersection points
def p₁ : ℝ × ℝ := (-2, 12)
def p₂ : ℝ × ℝ := (5.5, 23.25)

-- Theorem statement
theorem parabola_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x, y) = p₁ ∨ (x, y) = p₂) := by
  sorry

end parabola_intersection_l2645_264530


namespace solve_system_l2645_264507

/-- Given a system of equations, prove that y and z have specific values -/
theorem solve_system (x y z : ℚ) 
  (eq1 : (x + y) / (z - x) = 9/2)
  (eq2 : (y + z) / (y - x) = 5)
  (eq3 : x = 43/4) :
  y = 305/17 ∧ z = 1165/68 := by
  sorry

end solve_system_l2645_264507


namespace range_of_f_l2645_264514

def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

theorem range_of_f :
  Set.range f = Set.Icc (-8) 8 :=
sorry

end range_of_f_l2645_264514


namespace max_above_average_students_l2645_264557

theorem max_above_average_students (n : ℕ) (h : n = 150) :
  ∃ (scores : Fin n → ℚ),
    (∃ (count : ℕ), count = (Finset.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n) Finset.univ).card ∧
                    count = n - 1) ∧
    ∀ (count : ℕ),
      count = (Finset.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n) Finset.univ).card →
      count ≤ n - 1 :=
by sorry

end max_above_average_students_l2645_264557


namespace exponential_decreasing_range_l2645_264597

/-- A function f: ℝ → ℝ is strictly decreasing -/
def StrictlyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem exponential_decreasing_range (a : ℝ) :
  StrictlyDecreasing (fun x ↦ (a - 1) ^ x) → 1 < a ∧ a < 2 := by
  sorry

end exponential_decreasing_range_l2645_264597


namespace limit_P_div_B_l2645_264519

/-- The number of ways to make n cents using quarters, dimes, nickels, and pennies -/
def P (n : ℕ) : ℕ := sorry

/-- The number of ways to make n cents using dollar bills, quarters, dimes, and nickels -/
def B (n : ℕ) : ℕ := sorry

/-- The value of a penny in cents -/
def penny : ℕ := 1

/-- The value of a nickel in cents -/
def nickel : ℕ := 5

/-- The value of a dime in cents -/
def dime : ℕ := 10

/-- The value of a quarter in cents -/
def quarter : ℕ := 25

/-- The value of a dollar bill in cents -/
def dollar : ℕ := 100

/-- The limit of P_n / B_n as n approaches infinity -/
theorem limit_P_div_B :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((P n : ℝ) / (B n : ℝ)) - (1 / 20)| < ε :=
sorry

end limit_P_div_B_l2645_264519


namespace marbles_cost_calculation_l2645_264521

/-- The amount spent on marbles, given the total spent on toys and the cost of a football -/
def marbles_cost (total_spent : ℚ) (football_cost : ℚ) : ℚ :=
  total_spent - football_cost

/-- Theorem stating that the amount spent on marbles is $6.59 -/
theorem marbles_cost_calculation :
  marbles_cost 12.30 5.71 = 6.59 := by
  sorry

end marbles_cost_calculation_l2645_264521


namespace increasing_function_a_range_l2645_264501

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4*a*x else (2*a+3)*x - 4*a + 5

theorem increasing_function_a_range (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) → a ≥ 1/2 := by
  sorry

end increasing_function_a_range_l2645_264501


namespace simplify_expression_l2645_264559

theorem simplify_expression (a b : ℝ) : 4 * (a - 2 * b) - 2 * (2 * a + 3 * b) = -14 * b := by
  sorry

end simplify_expression_l2645_264559


namespace hyperbola_equation_l2645_264561

/-- The equation of a hyperbola with given conditions -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (∃ c : ℝ, c / a = Real.sqrt 3) →
  (∃ k : ℝ, ∀ x : ℝ, k = -1 ∧ k = a^2 / (a * Real.sqrt 3)) →
  (x^2 / 3 - y^2 / 6 = 1) :=
sorry

end hyperbola_equation_l2645_264561


namespace age_difference_l2645_264584

theorem age_difference (younger_age elder_age : ℕ) 
  (h1 : younger_age = 33)
  (h2 : elder_age = 53) : 
  elder_age - younger_age = 20 := by
sorry

end age_difference_l2645_264584


namespace grocery_store_inventory_l2645_264546

theorem grocery_store_inventory (apples regular_soda diet_soda : ℕ) 
  (h1 : apples = 36)
  (h2 : regular_soda = 80)
  (h3 : diet_soda = 54) :
  regular_soda + diet_soda - apples = 98 := by
  sorry

end grocery_store_inventory_l2645_264546


namespace problem_solution_l2645_264512

theorem problem_solution (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : a ^ b = b ^ a) (h4 : b = 27 * a) : a = (27 : ℝ) ^ (1 / 26) := by
  sorry

end problem_solution_l2645_264512


namespace negative_three_plus_four_equals_one_l2645_264560

theorem negative_three_plus_four_equals_one : -3 + 4 = 1 := by
  sorry

end negative_three_plus_four_equals_one_l2645_264560


namespace student_count_proof_l2645_264524

theorem student_count_proof (initial_avg : ℝ) (new_student_weight : ℝ) (final_avg : ℝ) :
  initial_avg = 28 →
  new_student_weight = 7 →
  final_avg = 27.3 →
  (∃ n : ℕ, (n : ℝ) * initial_avg + new_student_weight = (n + 1 : ℝ) * final_avg ∧ n = 29) :=
by sorry

end student_count_proof_l2645_264524


namespace game_ends_after_63_rounds_l2645_264573

/-- Represents a player in the game -/
inductive Player : Type
| A | B | C | D

/-- Represents the state of the game -/
structure GameState :=
  (tokens : Player → ℕ)
  (round : ℕ)

/-- Initial state of the game -/
def initialState : GameState :=
  { tokens := λ p => match p with
    | Player.A => 20
    | Player.B => 18
    | Player.C => 16
    | Player.D => 14
  , round := 0 }

/-- Updates the game state for one round -/
def updateState (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Theorem stating that the game ends after 63 rounds -/
theorem game_ends_after_63_rounds :
  ∃ (finalState : GameState),
    finalState.round = 63 ∧
    isGameOver finalState ∧
    (∀ (prevState : GameState),
      prevState.round < 63 →
      ¬isGameOver prevState) :=
  sorry

end game_ends_after_63_rounds_l2645_264573


namespace original_number_is_ten_l2645_264525

theorem original_number_is_ten : 
  ∃ x : ℝ, (2 * x + 5 = x / 2 + 20) ∧ (x = 10) := by
  sorry

end original_number_is_ten_l2645_264525


namespace max_gcd_of_product_7200_l2645_264540

theorem max_gcd_of_product_7200 :
  ∃ (a b : ℕ), a * b = 7200 ∧ 
  ∀ (c d : ℕ), c * d = 7200 → Nat.gcd c d ≤ 60 ∧
  Nat.gcd a b = 60 :=
sorry

end max_gcd_of_product_7200_l2645_264540


namespace quadratic_uniqueness_l2645_264504

/-- A quadratic function is uniquely determined by three distinct points -/
theorem quadratic_uniqueness (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) :
  ∃! (a b c : ℝ), 
    y₁ = a * x₁^2 + b * x₁ + c ∧
    y₂ = a * x₂^2 + b * x₂ + c ∧
    y₃ = a * x₃^2 + b * x₃ + c := by
  sorry

#check quadratic_uniqueness

end quadratic_uniqueness_l2645_264504


namespace diamond_ratio_equals_three_fifths_l2645_264541

-- Define the ♢ operation
def diamond (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem diamond_ratio_equals_three_fifths :
  (diamond 5 3) / (diamond 3 5 : ℚ) = 3/5 := by sorry

end diamond_ratio_equals_three_fifths_l2645_264541


namespace sqrt_sum_given_diff_l2645_264523

theorem sqrt_sum_given_diff (y : ℝ) : 
  Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4 → 
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
sorry

end sqrt_sum_given_diff_l2645_264523


namespace total_pencils_and_crayons_l2645_264587

theorem total_pencils_and_crayons (rows : ℕ) (pencils_per_row : ℕ) (crayons_per_row : ℕ)
  (h_rows : rows = 11)
  (h_pencils : pencils_per_row = 31)
  (h_crayons : crayons_per_row = 27) :
  rows * pencils_per_row + rows * crayons_per_row = 638 := by
  sorry

end total_pencils_and_crayons_l2645_264587


namespace total_cost_is_2000_l2645_264551

/-- The cost of buying two laptops, where the first laptop costs $500 and the second laptop is 3 times as costly as the first laptop. -/
def total_cost (first_laptop_cost : ℕ) (cost_multiplier : ℕ) : ℕ :=
  first_laptop_cost + (cost_multiplier * first_laptop_cost)

/-- Theorem stating that the total cost of buying both laptops is $2000. -/
theorem total_cost_is_2000 : total_cost 500 3 = 2000 := by
  sorry

end total_cost_is_2000_l2645_264551


namespace functional_equation_solution_l2645_264564

-- Define the property that f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x + y) * (f x - f y)

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end functional_equation_solution_l2645_264564


namespace second_smallest_sum_of_two_cubes_l2645_264578

-- Define a function to check if a number is the sum of two cubes
def isSumOfTwoCubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ a^3 + b^3 = n

-- Define a function to check if a number can be written as the sum of two cubes in two different ways
def hasTwoDifferentCubeRepresentations (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a^3 + b^3 = n ∧ c^3 + d^3 = n ∧
    (a ≠ c ∨ b ≠ d) ∧ (a ≠ d ∨ b ≠ c)

-- Define the theorem
theorem second_smallest_sum_of_two_cubes : 
  (∃ m : ℕ, m < 4104 ∧ hasTwoDifferentCubeRepresentations m) ∧
  (∀ k : ℕ, k < 4104 → k ≠ 1729 → ¬hasTwoDifferentCubeRepresentations k) ∧
  hasTwoDifferentCubeRepresentations 4104 :=
sorry

end second_smallest_sum_of_two_cubes_l2645_264578


namespace william_has_45_napkins_l2645_264555

/-- The number of napkins William has now -/
def williams_napkins (original : ℕ) (from_olivia : ℕ) (amelia_multiplier : ℕ) : ℕ :=
  original + from_olivia + amelia_multiplier * from_olivia

/-- Proof that William has 45 napkins given the conditions -/
theorem william_has_45_napkins :
  williams_napkins 15 10 2 = 45 := by
  sorry

end william_has_45_napkins_l2645_264555


namespace line_intersects_ellipse_at_midpoint_l2645_264542

theorem line_intersects_ellipse_at_midpoint (x y : ℝ) :
  let P : ℝ × ℝ := (1, 1)
  let ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1
  let line (x y : ℝ) : Prop := 4*x + 9*y = 13
  (∀ x y, line x y → (x, y) = P ∨ ellipse x y) ∧
  (∃ A B : ℝ × ℝ, A ≠ B ∧ line A.1 A.2 ∧ line B.1 B.2 ∧ 
    ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
    ((A.1 + B.1)/2, (A.2 + B.2)/2) = P) :=
by sorry

end line_intersects_ellipse_at_midpoint_l2645_264542


namespace molecular_weight_c4h10_l2645_264506

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The number of Carbon atoms in C4H10 -/
def carbon_count : ℕ := 4

/-- The number of Hydrogen atoms in C4H10 -/
def hydrogen_count : ℕ := 10

/-- The number of moles of C4H10 -/
def mole_count : ℝ := 6

/-- Theorem: The molecular weight of 6 moles of C4H10 is 348.72 grams -/
theorem molecular_weight_c4h10 :
  (carbon_weight * carbon_count + hydrogen_weight * hydrogen_count) * mole_count = 348.72 := by
  sorry

end molecular_weight_c4h10_l2645_264506


namespace painted_cube_theorem_l2645_264580

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  side_count : ℕ
  total_cubes : ℕ
  inner_cubes : ℕ

/-- The number of smaller cubes with no faces colored in a painted cube cut into 64 equal parts -/
def painted_cube_inner_count : ℕ := 8

/-- Theorem: In a cube cut into 64 equal smaller cubes, 
    the number of smaller cubes with no faces touching the original cube's surface is 8 -/
theorem painted_cube_theorem (c : CutCube) 
  (h1 : c.side_count = 4)
  (h2 : c.total_cubes = 64)
  (h3 : c.inner_cubes = (c.side_count - 2)^3) :
  c.inner_cubes = painted_cube_inner_count := by sorry

end painted_cube_theorem_l2645_264580

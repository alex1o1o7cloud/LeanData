import Mathlib

namespace a_formula_l3746_374659

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ+) : ℤ := n^2 - n

/-- The nth term of the sequence a_n -/
def a (n : ℕ+) : ℤ := 
  if n = 1 then S 1
  else S n - S (n - 1)

theorem a_formula (n : ℕ+) : a n = 2*n - 2 := by
  sorry

end a_formula_l3746_374659


namespace perfect_square_condition_l3746_374600

theorem perfect_square_condition (a b k : ℝ) :
  (∃ (c : ℝ), 4 * a^2 + k * a * b + 9 * b^2 = c^2) →
  k = 12 ∨ k = -12 := by
sorry

end perfect_square_condition_l3746_374600


namespace set_operations_and_intersection_condition_l3746_374671

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}

theorem set_operations_and_intersection_condition (a : ℝ) :
  (A ∪ B = {x | 1 < x ∧ x ≤ 8}) ∧
  ((Set.univ \ A) ∩ B = {x | 1 < x ∧ x < 2}) ∧
  (A ∩ C a ≠ ∅ → a < 8) := by
  sorry

end set_operations_and_intersection_condition_l3746_374671


namespace cos_double_angle_with_tan_l3746_374649

theorem cos_double_angle_with_tan (θ : ℝ) (h : Real.tan θ = 3) : Real.cos (2 * θ) = -4/5 := by
  sorry

end cos_double_angle_with_tan_l3746_374649


namespace negation_of_existence_geq_l3746_374608

theorem negation_of_existence_geq (p : Prop) :
  (¬ (∃ x : ℝ, x^2 ≥ x)) ↔ (∀ x : ℝ, x^2 < x) := by
  sorry

end negation_of_existence_geq_l3746_374608


namespace quadratic_inequality_coefficients_sum_l3746_374629

/-- Given a quadratic inequality x² - ax + b < 0 with solution set {x | 1 < x < 2},
    prove that a + b = 5 -/
theorem quadratic_inequality_coefficients_sum (a b : ℝ) : 
  (∀ x, x^2 - a*x + b < 0 ↔ 1 < x ∧ x < 2) → a + b = 5 := by
  sorry

end quadratic_inequality_coefficients_sum_l3746_374629


namespace cookie_problem_l3746_374675

theorem cookie_problem (initial_cookies : ℕ) : 
  (initial_cookies : ℚ) * (1/4) * (1/2) = 8 → initial_cookies = 64 := by
  sorry

end cookie_problem_l3746_374675


namespace eve_can_discover_secret_number_l3746_374638

theorem eve_can_discover_secret_number :
  ∀ x : ℕ, ∃ (k : ℕ) (n : Fin k → ℕ),
    ∀ y : ℕ, (∀ i : Fin k, Prime (x + n i) ↔ Prime (y + n i)) → x = y :=
sorry

end eve_can_discover_secret_number_l3746_374638


namespace translation_of_quadratic_l3746_374626

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 4

/-- The translated quadratic function -/
def g (x : ℝ) : ℝ := (x - 1)^2 - 2

/-- Theorem stating that g is the result of translating f one unit left and two units up -/
theorem translation_of_quadratic :
  ∀ x : ℝ, g x = f (x - 1) + 2 := by sorry

end translation_of_quadratic_l3746_374626


namespace x_greater_than_x_squared_only_half_satisfies_l3746_374605

theorem x_greater_than_x_squared (x : ℝ) : x > x^2 ↔ x ∈ (Set.Ioo 0 1) := by sorry

theorem only_half_satisfies :
  ∀ x ∈ ({-2, -(1/2), 0, 1/2, 2} : Set ℝ), x > x^2 ↔ x = 1/2 := by sorry

end x_greater_than_x_squared_only_half_satisfies_l3746_374605


namespace mean_median_difference_l3746_374655

/-- Represents the score distribution in a class -/
structure ScoreDistribution where
  total_students : ℕ
  score_60_percent : ℚ
  score_75_percent : ℚ
  score_85_percent : ℚ
  score_90_percent : ℚ
  score_100_percent : ℚ

/-- Calculates the mean score given a score distribution -/
def mean_score (dist : ScoreDistribution) : ℚ :=
  (60 * dist.score_60_percent + 75 * dist.score_75_percent + 
   85 * dist.score_85_percent + 90 * dist.score_90_percent + 
   100 * dist.score_100_percent) / 1

/-- Calculates the median score given a score distribution -/
def median_score (dist : ScoreDistribution) : ℚ := 85

/-- Theorem stating the difference between mean and median scores -/
theorem mean_median_difference (dist : ScoreDistribution) : 
  dist.total_students = 25 ∧
  dist.score_60_percent = 15/100 ∧
  dist.score_75_percent = 20/100 ∧
  dist.score_85_percent = 30/100 ∧
  dist.score_90_percent = 20/100 ∧
  dist.score_100_percent = 15/100 →
  mean_score dist - median_score dist = 8/10 := by
  sorry

end mean_median_difference_l3746_374655


namespace two_x_value_l3746_374622

theorem two_x_value (x : ℚ) (h : 4 * x + 14 = 8 * x - 48) : 2 * x = 31 := by
  sorry

end two_x_value_l3746_374622


namespace car_speed_increase_l3746_374652

theorem car_speed_increase (original_speed : ℝ) (supercharge_percent : ℝ) (weight_reduction_increase : ℝ) : 
  original_speed = 150 → 
  supercharge_percent = 30 → 
  weight_reduction_increase = 10 → 
  original_speed * (1 + supercharge_percent / 100) + weight_reduction_increase = 205 :=
by
  sorry

#check car_speed_increase

end car_speed_increase_l3746_374652


namespace max_value_L_in_triangle_l3746_374612

/-- The function L(x, y) = -2x + y -/
def L (x y : ℝ) : ℝ := -2*x + y

/-- The triangle ABC with vertices A(-2, -1), B(0, 1), and C(2, -1) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {p | ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
       p.1 = -2*a + 0*b + 2*c ∧
       p.2 = -1*a + 1*b - 1*c}

theorem max_value_L_in_triangle :
  ∃ (max : ℝ), max = 3 ∧ 
  ∀ (x y : ℝ), (x, y) ∈ triangle_ABC → L x y ≤ max :=
sorry

end max_value_L_in_triangle_l3746_374612


namespace equidistant_point_on_y_axis_l3746_374643

theorem equidistant_point_on_y_axis : 
  ∃ y : ℝ, y > 0 ∧ 
  ((-3 - 0)^2 + (0 - y)^2 = (-2 - 0)^2 + (5 - y)^2) ∧ 
  y = 2 := by
  sorry

end equidistant_point_on_y_axis_l3746_374643


namespace sqrt_fraction_equality_l3746_374611

theorem sqrt_fraction_equality : 
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) = 
  (-2 * Real.sqrt 7 + Real.sqrt 3 + Real.sqrt 5) / 59 := by
  sorry

end sqrt_fraction_equality_l3746_374611


namespace committee_selection_l3746_374606

theorem committee_selection (n : ℕ) (k : ℕ) : n = 9 → k = 4 → Nat.choose n k = 126 := by
  sorry

end committee_selection_l3746_374606


namespace neither_necessary_nor_sufficient_condition_l3746_374693

/-- A geometric sequence with first term a and common ratio q -/
def GeometricSequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

/-- Predicate for an increasing sequence -/
def IsIncreasing (f : ℕ → ℝ) : Prop := ∀ n : ℕ, f n ≤ f (n + 1)

theorem neither_necessary_nor_sufficient_condition
  (a q : ℝ) :
  ¬(((a * q > 0) ↔ IsIncreasing (GeometricSequence a q))) :=
sorry

end neither_necessary_nor_sufficient_condition_l3746_374693


namespace garden_separation_possible_l3746_374663

/-- Represents the content of a garden plot -/
inductive PlotContent
  | Empty
  | Cabbage
  | Goat

/-- Represents a position in the garden -/
structure Position where
  x : Fin 4
  y : Fin 4

/-- Represents a fence in the garden -/
inductive Fence
  | Vertical (x : Fin 3) -- A vertical fence after column x
  | Horizontal (y : Fin 3) -- A horizontal fence after row y

/-- Represents the garden layout -/
def Garden := Position → PlotContent

/-- Checks if a fence separates two positions -/
def separates (f : Fence) (p1 p2 : Position) : Prop :=
  match f with
  | Fence.Vertical x => p1.x ≤ x ∧ x < p2.x
  | Fence.Horizontal y => p1.y ≤ y ∧ y < p2.y

/-- The theorem to be proved -/
theorem garden_separation_possible (g : Garden) :
  ∃ (f1 f2 f3 : Fence),
    (∀ p1 p2 : Position,
      g p1 = PlotContent.Goat →
      g p2 = PlotContent.Cabbage →
      separates f1 p1 p2 ∨ separates f2 p1 p2 ∨ separates f3 p1 p2) ∧
    (∀ f : Fence, f ∈ [f1, f2, f3] →
      ∀ p : Position,
        g p ≠ PlotContent.Empty →
        ¬(∃ p' : Position, g p' ≠ PlotContent.Empty ∧ separates f p p')) :=
by sorry

end garden_separation_possible_l3746_374663


namespace product_equals_sum_solution_l3746_374662

theorem product_equals_sum_solution :
  ∀ (a b c d e f : ℕ),
    a * b * c * d * e * f = a + b + c + d + e + f →
    ((a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 2 ∧ f = 6) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 6 ∧ f = 2) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 2 ∧ e = 1 ∧ f = 6) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 2 ∧ e = 6 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 6 ∧ e = 1 ∧ f = 2) ∨
     (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 6 ∧ e = 2 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 1 ∧ e = 1 ∧ f = 6) ∨
     (a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 1 ∧ e = 6 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 6 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 6 ∧ d = 1 ∧ e = 1 ∧ f = 2) ∨
     (a = 1 ∧ b = 1 ∧ c = 6 ∧ d = 1 ∧ e = 2 ∧ f = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 6 ∧ d = 2 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 6) ∨
     (a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 6 ∧ f = 1) ∨
     (a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 6 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 2 ∧ c = 6 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 6 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 2) ∨
     (a = 1 ∧ b = 6 ∧ c = 1 ∧ d = 1 ∧ e = 2 ∧ f = 1) ∨
     (a = 1 ∧ b = 6 ∧ c = 1 ∧ d = 2 ∧ e = 1 ∧ f = 1) ∨
     (a = 1 ∧ b = 6 ∧ c = 2 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 6) ∨
     (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 6 ∧ f = 1) ∨
     (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 6 ∧ e = 1 ∧ f = 1) ∨
     (a = 2 ∧ b = 1 ∧ c = 6 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 2 ∧ b = 6 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 6 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 2) ∨
     (a = 6 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 2 ∧ f = 1) ∨
     (a = 6 ∧ b = 1 ∧ c = 1 ∧ d = 2 ∧ e = 1 ∧ f = 1) ∨
     (a = 6 ∧ b = 1 ∧ c = 2 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 6 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1)) :=
by
  sorry


end product_equals_sum_solution_l3746_374662


namespace store_loss_percentage_l3746_374691

/-- Calculate the store's loss percentage on a radio sale -/
theorem store_loss_percentage
  (cost_price : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (actual_selling_price : ℝ)
  (h1 : cost_price = 25000)
  (h2 : discount_rate = 0.15)
  (h3 : tax_rate = 0.05)
  (h4 : actual_selling_price = 22000) :
  let discounted_price := cost_price * (1 - discount_rate)
  let final_selling_price := discounted_price * (1 + tax_rate)
  let loss := final_selling_price - actual_selling_price
  let loss_percentage := (loss / cost_price) * 100
  loss_percentage = 1.25 := by
sorry


end store_loss_percentage_l3746_374691


namespace apple_ratio_l3746_374699

def total_apples : ℕ := 496
def green_apples : ℕ := 124

theorem apple_ratio : 
  let red_apples := total_apples - green_apples
  (red_apples : ℚ) / green_apples = 93 / 31 := by
sorry

end apple_ratio_l3746_374699


namespace diagonal_intersection_probability_l3746_374639

/-- The probability that two randomly chosen diagonals intersect in a convex polygon with 2n+1 vertices -/
theorem diagonal_intersection_probability (n : ℕ) (h : n > 0) :
  let vertices := 2 * n + 1
  let total_diagonals := vertices.choose 2 - vertices
  let intersecting_pairs := vertices.choose 4
  let probability := intersecting_pairs / total_diagonals.choose 2
  probability = n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) :=
by sorry

end diagonal_intersection_probability_l3746_374639


namespace congruence_iff_divisible_l3746_374688

theorem congruence_iff_divisible (a b m : ℤ) : a ≡ b [ZMOD m] ↔ m ∣ (a - b) := by sorry

end congruence_iff_divisible_l3746_374688


namespace min_sum_at_five_l3746_374647

/-- An arithmetic sequence -/
def arithmetic_sequence : ℕ → ℝ := sorry

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The conditions given in the problem -/
axiom condition1 : S 10 = 0
axiom condition2 : S 15 = 25

/-- The theorem to prove -/
theorem min_sum_at_five :
  ∃ (n : ℕ), n = 5 ∧ ∀ (m : ℕ), S m ≥ S n :=
sorry

end min_sum_at_five_l3746_374647


namespace polynomial_factor_theorem_l3746_374607

theorem polynomial_factor_theorem (c q k : ℝ) : 
  (∀ x, 3 * x^3 + c * x + 8 = (x^2 + q * x + 2) * (3 * x + k)) →
  c = 4 := by
sorry

end polynomial_factor_theorem_l3746_374607


namespace x_value_l3746_374623

theorem x_value (x : ℝ) : x = 40 * (1 + 0.2) → x = 48 := by
  sorry

end x_value_l3746_374623


namespace box_volume_increase_l3746_374664

/-- Given a rectangular box with length l, width w, and height h satisfying certain conditions,
    prove that increasing each dimension by 2 results in a volume of 7208 cubic inches. -/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5400)
  (surface_area : 2 * (l * w + w * h + h * l) = 1560)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7208 := by
  sorry

end box_volume_increase_l3746_374664


namespace order_of_t_squared_t_neg_t_l3746_374692

theorem order_of_t_squared_t_neg_t (t : ℝ) (h : t^2 + t < 0) : t < t^2 ∧ t^2 < -t := by
  sorry

end order_of_t_squared_t_neg_t_l3746_374692


namespace standing_arrangements_eq_210_l3746_374615

/-- The number of ways to arrange n distinct objects in k positions --/
def arrangement (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose k objects from n distinct objects --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways 3 people can stand on 6 steps with given conditions --/
def standing_arrangements : ℕ :=
  arrangement 6 3 + choose 3 1 * arrangement 6 2

theorem standing_arrangements_eq_210 : standing_arrangements = 210 := by sorry

end standing_arrangements_eq_210_l3746_374615


namespace total_combinations_l3746_374689

/-- The number of color options available -/
def num_colors : ℕ := 5

/-- The number of painting method options available -/
def num_methods : ℕ := 4

/-- Theorem: The total number of combinations of color and painting method is 20 -/
theorem total_combinations : num_colors * num_methods = 20 := by
  sorry

end total_combinations_l3746_374689


namespace abdul_binh_age_difference_l3746_374651

/- Define Susie's age -/
variable (S : ℤ)

/- Define Abdul's age in terms of Susie's -/
def A : ℤ := S + 9

/- Define Binh's age in terms of Susie's -/
def B : ℤ := S + 2

/- Theorem statement -/
theorem abdul_binh_age_difference : A - B = 7 := by
  sorry

end abdul_binh_age_difference_l3746_374651


namespace third_number_proof_l3746_374632

/-- The largest five-digit number with all even digits -/
def largest_even_five_digit : ℕ := 88888

/-- The smallest four-digit number with all odd digits -/
def smallest_odd_four_digit : ℕ := 1111

/-- The sum of the three numbers -/
def total_sum : ℕ := 121526

/-- The third number -/
def third_number : ℕ := total_sum - largest_even_five_digit - smallest_odd_four_digit

theorem third_number_proof :
  third_number = 31527 :=
by sorry

end third_number_proof_l3746_374632


namespace problem_solution_l3746_374610

def sum_of_integers (a b : ℕ) : ℕ :=
  ((b - a + 1) * (a + b)) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem problem_solution :
  let x := sum_of_integers 40 60
  let y := count_even_integers 40 60
  x + y = 1061 → y = 11 := by
sorry

end problem_solution_l3746_374610


namespace sin_alpha_for_point_l3746_374646

/-- If the terminal side of angle α passes through point (-1, 2), then sin α = 2√5/5 -/
theorem sin_alpha_for_point (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 2) → 
  Real.sin α = 2 * Real.sqrt 5 / 5 := by
sorry

end sin_alpha_for_point_l3746_374646


namespace arithmetic_progression_properties_l3746_374660

-- Define the arithmetic progression
def arithmeticProgression (n : ℕ) : ℕ := 36 * n + 3

-- Define the property of not being a sum of two squares
def notSumOfTwoSquares (k : ℕ) : Prop := ∀ a b : ℕ, k ≠ a^2 + b^2

-- Define the property of not being a sum of two cubes
def notSumOfTwoCubes (k : ℕ) : Prop := ∀ a b : ℕ, k ≠ a^3 + b^3

theorem arithmetic_progression_properties :
  (∀ n : ℕ, arithmeticProgression n > 0) ∧  -- Positive integers
  (∀ n m : ℕ, n ≠ m → arithmeticProgression n ≠ arithmeticProgression m) ∧  -- Non-constant
  (∀ n : ℕ, notSumOfTwoSquares (arithmeticProgression n)) ∧  -- Not sum of two squares
  (∀ n : ℕ, notSumOfTwoCubes (arithmeticProgression n)) :=
by sorry

end arithmetic_progression_properties_l3746_374660


namespace sum_x_y_equals_six_l3746_374661

theorem sum_x_y_equals_six (x y : ℝ) : 
  (|x| + x + y = 16) → (x + |y| - y = 18) → (x + y = 6) := by
  sorry

end sum_x_y_equals_six_l3746_374661


namespace smallest_integer_satisfying_inequality_l3746_374657

theorem smallest_integer_satisfying_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 13*m + 36 ≤ 0 → n ≤ m) ∧ (n^2 - 13*n + 36 ≤ 0) ∧ n = 4 := by
  sorry

end smallest_integer_satisfying_inequality_l3746_374657


namespace least_common_multiple_4_5_6_9_l3746_374636

theorem least_common_multiple_4_5_6_9 : ∃ (n : ℕ), n > 0 ∧ 
  4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n ∧ 9 ∣ n ∧ 
  ∀ (m : ℕ), m > 0 → 4 ∣ m → 5 ∣ m → 6 ∣ m → 9 ∣ m → n ≤ m :=
by
  use 180
  sorry

end least_common_multiple_4_5_6_9_l3746_374636


namespace max_a_for_monotonic_increasing_l3746_374617

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem max_a_for_monotonic_increasing (a : ℝ) : 
  (∀ x ≥ 1, ∀ y ≥ x, f a y ≥ f a x) → a ≤ 3 :=
by sorry

end max_a_for_monotonic_increasing_l3746_374617


namespace negation_of_conjunction_l3746_374676

theorem negation_of_conjunction (x y : ℝ) : 
  ¬(x = 2 ∧ y = 3) ↔ (x ≠ 2 ∨ y ≠ 3) := by
  sorry

end negation_of_conjunction_l3746_374676


namespace problem_statement_l3746_374624

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 1

/-- The statement to be proved -/
theorem problem_statement (m : ℝ) : 
  (∀ x : ℝ, f m x > 0) ∧ 
  (∃ x : ℝ, x^2 < 9 - m^2) ↔ 
  (m > -3 ∧ m ≤ -2) ∨ (m ≥ 2 ∧ m < 3) :=
sorry

end problem_statement_l3746_374624


namespace celsius_to_fahrenheit_55_l3746_374631

/-- Converts Celsius to Fahrenheit -/
def celsius_to_fahrenheit (c : ℚ) : ℚ := (c * 9 / 5) + 32

/-- Water boiling point in Fahrenheit -/
def water_boiling_f : ℚ := 212

/-- Water boiling point in Celsius -/
def water_boiling_c : ℚ := 100

/-- Ice melting point in Fahrenheit -/
def ice_melting_f : ℚ := 32

/-- Ice melting point in Celsius -/
def ice_melting_c : ℚ := 0

/-- The temperature of the pot of water in Celsius -/
def pot_temp_c : ℚ := 55

/-- The temperature of the pot of water in Fahrenheit -/
def pot_temp_f : ℚ := 131

theorem celsius_to_fahrenheit_55 :
  celsius_to_fahrenheit pot_temp_c = pot_temp_f := by sorry

end celsius_to_fahrenheit_55_l3746_374631


namespace selection_methods_count_l3746_374620

def total_students : ℕ := 9
def selected_students : ℕ := 4
def specific_students : ℕ := 3

def selection_methods : ℕ := 
  Nat.choose specific_students 2 * Nat.choose (total_students - specific_students) 2 +
  Nat.choose specific_students 3 * Nat.choose (total_students - specific_students) 1

theorem selection_methods_count : selection_methods = 51 := by
  sorry

end selection_methods_count_l3746_374620


namespace alice_painted_six_cuboids_l3746_374658

/-- The number of outer faces on a cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of faces Alice painted -/
def total_painted_faces : ℕ := 36

/-- The number of cuboids Alice painted -/
def num_cuboids : ℕ := total_painted_faces / faces_per_cuboid

theorem alice_painted_six_cuboids :
  num_cuboids = 6 :=
sorry

end alice_painted_six_cuboids_l3746_374658


namespace solution_set_for_decreasing_function_l3746_374616

/-- A function f is decreasing on its domain -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The set of x satisfying f(1/x) > f(1) for a decreasing function f -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | f (1/x) > f 1}

theorem solution_set_for_decreasing_function (f : ℝ → ℝ) (h : IsDecreasing f) :
    SolutionSet f = {x | x < 0 ∨ x > 1} := by
  sorry

end solution_set_for_decreasing_function_l3746_374616


namespace B_equals_A_l3746_374618

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x ∈ A}

theorem B_equals_A : B = {1, 2, 3} := by
  sorry

end B_equals_A_l3746_374618


namespace particular_solution_l3746_374674

noncomputable def y (x : ℝ) : ℝ := Real.sqrt (1 + 2 * Real.log ((1 + Real.exp x) / 2))

theorem particular_solution (x : ℝ) :
  (1 + Real.exp x) * y x * (deriv y x) = Real.exp x ∧ y 0 = 1 := by
  sorry

end particular_solution_l3746_374674


namespace remaining_soup_feeds_six_adults_l3746_374633

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Proves that given 5 cans of soup, where each can feeds 3 adults or 5 children,
    if 15 children are fed, the remaining soup will feed 6 adults -/
theorem remaining_soup_feeds_six_adults 
  (can : SoupCan) 
  (h1 : can.adults = 3) 
  (h2 : can.children = 5) 
  (total_cans : ℕ) 
  (h3 : total_cans = 5) 
  (children_fed : ℕ) 
  (h4 : children_fed = 15) : 
  (total_cans - (children_fed / can.children)) * can.adults = 6 := by
sorry

end remaining_soup_feeds_six_adults_l3746_374633


namespace washington_dc_trip_cost_l3746_374656

/-- Calculates the total cost per person for a group trip to Washington D.C. -/
theorem washington_dc_trip_cost 
  (num_friends : ℕ)
  (airfare_hotel_cost : ℚ)
  (food_expenses : ℚ)
  (transportation_expenses : ℚ)
  (smithsonian_tour_cost : ℚ)
  (zoo_entry_fee : ℚ)
  (zoo_spending_allowance : ℚ)
  (river_cruise_cost : ℚ)
  (h1 : num_friends = 15)
  (h2 : airfare_hotel_cost = 13500)
  (h3 : food_expenses = 4500)
  (h4 : transportation_expenses = 3000)
  (h5 : smithsonian_tour_cost = 50)
  (h6 : zoo_entry_fee = 75)
  (h7 : zoo_spending_allowance = 15)
  (h8 : river_cruise_cost = 100) :
  (airfare_hotel_cost + food_expenses + transportation_expenses + 
   num_friends * (smithsonian_tour_cost + zoo_entry_fee + zoo_spending_allowance + river_cruise_cost)) / num_friends = 1640 := by
sorry

end washington_dc_trip_cost_l3746_374656


namespace field_area_l3746_374613

/-- Given a rectangular field with one side of 34 feet and three sides fenced with a total of 74 feet of fencing, the area of the field is 680 square feet. -/
theorem field_area (L W : ℝ) (h1 : L = 34) (h2 : 2 * W + L = 74) : L * W = 680 := by
  sorry

end field_area_l3746_374613


namespace jerry_one_way_time_15_minutes_l3746_374642

-- Define the distance to school in miles
def distance_to_school : ℝ := 4

-- Define Carson's speed in miles per hour
def carson_speed : ℝ := 8

-- Define the relationship between Jerry's round trip and Carson's one-way trip
axiom jerry_carson_time_relation : 
  ∀ (jerry_round_trip_time carson_one_way_time : ℝ), 
    jerry_round_trip_time = carson_one_way_time

-- Theorem: Jerry's one-way trip time to school is 15 minutes
theorem jerry_one_way_time_15_minutes : 
  ∃ (jerry_one_way_time : ℝ), 
    jerry_one_way_time = 15 := by sorry

end jerry_one_way_time_15_minutes_l3746_374642


namespace nesbitt_inequality_l3746_374682

theorem nesbitt_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
  sorry

end nesbitt_inequality_l3746_374682


namespace actual_distance_traveled_l3746_374621

/-- Proves that the actual distance traveled is 60 km given the conditions of the problem -/
theorem actual_distance_traveled (speed_slow speed_fast distance_difference : ℝ) 
  (h1 : speed_slow = 15)
  (h2 : speed_fast = 30)
  (h3 : distance_difference = 60)
  (h4 : speed_slow > 0)
  (h5 : speed_fast > speed_slow) :
  ∃ (actual_distance : ℝ), 
    actual_distance / speed_slow = (actual_distance + distance_difference) / speed_fast ∧ 
    actual_distance = 60 := by
  sorry

end actual_distance_traveled_l3746_374621


namespace marble_probability_l3746_374667

theorem marble_probability (total_marbles : ℕ) (p_white p_green : ℚ) :
  total_marbles = 90 →
  p_white = 1/3 →
  p_green = 1/5 →
  (1 : ℚ) - (p_white + p_green) = 7/15 := by
  sorry

end marble_probability_l3746_374667


namespace johns_mean_score_l3746_374695

def johns_scores : List ℝ := [89, 92, 95, 88, 90]

theorem johns_mean_score :
  (johns_scores.sum / johns_scores.length : ℝ) = 90.8 := by
  sorry

end johns_mean_score_l3746_374695


namespace fraction_equality_l3746_374627

theorem fraction_equality (a b : ℝ) (h : b / a = 1 / 2) : (a + b) / a = 3 / 2 := by
  sorry

end fraction_equality_l3746_374627


namespace elder_person_age_l3746_374650

theorem elder_person_age (y e : ℕ) : 
  e = y + 16 →                     -- The ages differ by 16 years
  e - 6 = 3 * (y - 6) →            -- 6 years ago, elder was 3 times younger's age
  e = 30                           -- Elder's present age is 30
  := by sorry

end elder_person_age_l3746_374650


namespace value_of_a_l3746_374665

theorem value_of_a (a b : ℚ) (h1 : b / a = 3) (h2 : b = 12 - 5 * a) : a = 3 / 2 := by
  sorry

end value_of_a_l3746_374665


namespace cloud_ratio_l3746_374645

theorem cloud_ratio : 
  let carson_clouds : ℕ := 6
  let total_clouds : ℕ := 24
  let brother_clouds : ℕ := total_clouds - carson_clouds
  (brother_clouds : ℚ) / carson_clouds = 3 := by
  sorry

end cloud_ratio_l3746_374645


namespace hyperbola_equation_l3746_374696

/-- A hyperbola with eccentricity √6/2 has the equation x²/4 - y²/2 = 1 -/
theorem hyperbola_equation (e : ℝ) (h : e = Real.sqrt 6 / 2) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), x^2 / (a^2) - y^2 / (b^2) = 1 ↔ 
    x^2 / 4 - y^2 / 2 = 1) :=
sorry

end hyperbola_equation_l3746_374696


namespace series_sum_l3746_374628

/-- The sum of the series Σ(3^(2^k) / (9^(2^k) - 1)) from k = 0 to infinity is 1/2 -/
theorem series_sum : 
  ∑' k, (3 ^ (2 ^ k) : ℝ) / ((9 : ℝ) ^ (2 ^ k) - 1) = 1 / 2 := by sorry

end series_sum_l3746_374628


namespace quadratic_monotonic_condition_l3746_374602

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- Define the property of monotonic interval starting at 1
def monotonic_from_one (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f a x < f a y

-- Theorem statement
theorem quadratic_monotonic_condition (a : ℝ) :
  monotonic_from_one a → a ≤ 2 := by
  sorry

end quadratic_monotonic_condition_l3746_374602


namespace ozone_experiment_properties_l3746_374604

/-- Represents the experimental setup and data for the ozone effect study on mice. -/
structure OzoneExperiment where
  total_mice : Nat
  control_group : Nat
  experimental_group : Nat
  weight_increases : List ℝ
  k_squared_threshold : ℝ

/-- Represents the distribution of X, where X is the number of specified two mice
    assigned to the control group. -/
def distribution_X (exp : OzoneExperiment) : Fin 3 → ℝ := sorry

/-- Calculates the expected value of X. -/
def expected_value_X (exp : OzoneExperiment) : ℝ := sorry

/-- Calculates the median of all mice weight increases. -/
def median_weight_increase (exp : OzoneExperiment) : ℝ := sorry

/-- Represents the contingency table based on the median. -/
structure ContingencyTable where
  control_less : Nat
  control_greater_equal : Nat
  experimental_less : Nat
  experimental_greater_equal : Nat

/-- Constructs the contingency table based on the median. -/
def create_contingency_table (exp : OzoneExperiment) : ContingencyTable := sorry

/-- Calculates the K^2 value based on the contingency table. -/
def calculate_k_squared (table : ContingencyTable) : ℝ := sorry

/-- The main theorem stating the properties of the ozone experiment. -/
theorem ozone_experiment_properties (exp : OzoneExperiment) :
  exp.total_mice = 40 ∧
  exp.control_group = 20 ∧
  exp.experimental_group = 20 ∧
  distribution_X exp 0 = 19/78 ∧
  distribution_X exp 1 = 20/39 ∧
  distribution_X exp 2 = 19/78 ∧
  expected_value_X exp = 1 ∧
  median_weight_increase exp = 23.4 ∧
  let table := create_contingency_table exp
  table.control_less = 6 ∧
  table.control_greater_equal = 14 ∧
  table.experimental_less = 14 ∧
  table.experimental_greater_equal = 6 ∧
  calculate_k_squared table = 6.4 ∧
  calculate_k_squared table > exp.k_squared_threshold := by
  sorry

end ozone_experiment_properties_l3746_374604


namespace smallest_valid_number_l3746_374601

def is_valid (n : ℕ) : Prop :=
  n ≥ 1000 ∧
  (n / 10) % 20 = 0 ∧
  (n % 1000) % 21 = 0 ∧
  (n / 100 % 10) ≠ 0

theorem smallest_valid_number :
  is_valid 1609 ∧ ∀ m < 1609, ¬(is_valid m) :=
sorry

end smallest_valid_number_l3746_374601


namespace min_value_line_circle_l3746_374637

/-- Given a line ax + by + c - 1 = 0 that passes through the center of the circle x^2 + y^2 - 2y - 5 = 0,
    prove that the minimum value of 4/b + 1/c is 9, where b > 0 and c > 0. -/
theorem min_value_line_circle (a b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (∀ x y : ℝ, a * x + b * y + c - 1 = 0 → x^2 + y^2 - 2*y - 5 = 0) →
  (∃ x y : ℝ, a * x + b * y + c - 1 = 0 ∧ x^2 + y^2 - 2*y - 5 = 0) →
  (∀ b' c' : ℝ, b' > 0 → c' > 0 → 4 / b' + 1 / c' ≥ 9) ∧
  (∃ b' c' : ℝ, b' > 0 ∧ c' > 0 ∧ 4 / b' + 1 / c' = 9) := by
  sorry

end min_value_line_circle_l3746_374637


namespace equation_satisfied_l3746_374668

theorem equation_satisfied (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
sorry

end equation_satisfied_l3746_374668


namespace four_people_five_chairs_middle_empty_l3746_374683

/-- The number of ways to arrange people in chairs. -/
def seating_arrangements (total_chairs : ℕ) (people : ℕ) (empty_chair : ℕ) : ℕ :=
  (total_chairs - 1).factorial / ((total_chairs - 1 - people).factorial)

/-- Theorem: There are 24 ways to arrange 4 people in 5 chairs with the middle chair empty. -/
theorem four_people_five_chairs_middle_empty :
  seating_arrangements 5 4 3 = 24 := by sorry

end four_people_five_chairs_middle_empty_l3746_374683


namespace max_value_M_l3746_374677

theorem max_value_M (x y z w : ℝ) (h : x + y + z + w = 1) :
  ∃ (max : ℝ), max = (3 : ℝ) / 2 ∧ 
  ∀ (a b c d : ℝ), a + b + c + d = 1 → 
  a * d + 2 * b * d + 3 * a * b + 3 * c * d + 4 * a * c + 5 * b * c ≤ max :=
sorry

end max_value_M_l3746_374677


namespace quadratic_no_real_roots_l3746_374609

theorem quadratic_no_real_roots : 
  ∀ x : ℝ, 7 * x^2 - 4 * x + 6 ≠ 0 := by
  sorry

end quadratic_no_real_roots_l3746_374609


namespace lowest_score_proof_l3746_374670

theorem lowest_score_proof (scores : List ℝ) (highest lowest : ℝ) : 
  scores.length = 12 →
  scores.sum / scores.length = 82 →
  highest ∈ scores →
  lowest ∈ scores →
  highest = 98 →
  (scores.filter (λ x => x ≠ highest ∧ x ≠ lowest)).sum / 10 = 84 →
  lowest = 46 := by
sorry

end lowest_score_proof_l3746_374670


namespace log_equation_solution_l3746_374690

theorem log_equation_solution (x : ℝ) (h1 : x > 0) (h2 : 2 * x ≠ 1) (h3 : 4 * x ≠ 1) :
  (Real.log (4 * x) / Real.log (2 * x)) + (Real.log (16 * x) / Real.log (4 * x)) = 4 ↔ 
  x = 1 ∨ x = 1 / (2 * Real.sqrt 2) := by
sorry

end log_equation_solution_l3746_374690


namespace jacket_price_proof_l3746_374697

/-- The original price of the jacket -/
def original_price : ℝ := 250

/-- The regular discount percentage -/
def regular_discount : ℝ := 0.4

/-- The weekend additional discount percentage -/
def weekend_discount : ℝ := 0.1

/-- The final price after both discounts -/
def final_price : ℝ := original_price * (1 - regular_discount) * (1 - weekend_discount)

theorem jacket_price_proof : final_price = 135 := by
  sorry

end jacket_price_proof_l3746_374697


namespace probability_Sa_before_Sb_l3746_374672

/-- Represents a three-letter string -/
structure ThreeLetterString :=
  (letters : Fin 3 → Char)

/-- The probability of a letter being received correctly -/
def correct_probability : ℚ := 2/3

/-- The probability of a letter being received incorrectly -/
def incorrect_probability : ℚ := 1/3

/-- The transmitted string aaa -/
def aaa : ThreeLetterString :=
  { letters := λ _ => 'a' }

/-- The transmitted string bbb -/
def bbb : ThreeLetterString :=
  { letters := λ _ => 'b' }

/-- The received string when aaa is transmitted -/
def Sa : ThreeLetterString :=
  sorry

/-- The received string when bbb is transmitted -/
def Sb : ThreeLetterString :=
  sorry

/-- The probability that Sa comes before Sb in alphabetical order -/
def p : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem probability_Sa_before_Sb : p = 532/729 :=
  sorry

end probability_Sa_before_Sb_l3746_374672


namespace bob_gardening_project_cost_l3746_374687

/-- The total cost of Bob's gardening project --/
def gardening_project_cost 
  (num_rose_bushes : ℕ) 
  (cost_per_rose_bush : ℕ) 
  (gardener_hourly_rate : ℕ) 
  (gardener_hours_per_day : ℕ) 
  (gardener_work_days : ℕ) 
  (soil_volume : ℕ) 
  (soil_cost_per_unit : ℕ) : ℕ :=
  num_rose_bushes * cost_per_rose_bush + 
  gardener_hourly_rate * gardener_hours_per_day * gardener_work_days + 
  soil_volume * soil_cost_per_unit

/-- Theorem stating that the total cost of Bob's gardening project is $4100 --/
theorem bob_gardening_project_cost : 
  gardening_project_cost 20 150 30 5 4 100 5 = 4100 := by
  sorry

end bob_gardening_project_cost_l3746_374687


namespace chip_cost_is_fifty_cents_l3746_374678

/-- The cost of a bag of chips given the conditions in the problem -/
def chip_cost : ℚ :=
  let candy_cost : ℚ := 2
  let student_count : ℕ := 5
  let total_cost : ℚ := 15
  let candy_per_student : ℕ := 1
  let chips_per_student : ℕ := 2
  (total_cost - student_count * candy_cost) / (student_count * chips_per_student)

/-- Theorem stating that the cost of a bag of chips is $0.50 -/
theorem chip_cost_is_fifty_cents : chip_cost = 1/2 := by
  sorry

end chip_cost_is_fifty_cents_l3746_374678


namespace study_session_duration_in_minutes_l3746_374666

-- Define the duration of the study session
def study_session_hours : ℕ := 8
def study_session_minutes : ℕ := 45

-- Define the conversion factor from hours to minutes
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem study_session_duration_in_minutes :
  study_session_hours * minutes_per_hour + study_session_minutes = 525 :=
by sorry

end study_session_duration_in_minutes_l3746_374666


namespace parallelogram_height_l3746_374694

/-- Proves that the height of a parallelogram is 18 cm given its area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (h1 : area = 648) (h2 : base = 36) :
  area / base = 18 := by
  sorry

end parallelogram_height_l3746_374694


namespace line_parameterization_l3746_374684

def is_valid_parameterization (x₀ y₀ dx dy : ℝ) : Prop :=
  y₀ = 3 * x₀ + 5 ∧ ∃ (k : ℝ), dx = k * 1 ∧ dy = k * 3

theorem line_parameterization 
  (x₀ y₀ dx dy t : ℝ) :
  is_valid_parameterization x₀ y₀ dx dy ↔ 
  ∀ t, (3 * (x₀ + t * dx) + 5 = y₀ + t * dy) :=
by sorry

end line_parameterization_l3746_374684


namespace max_coins_distribution_l3746_374640

theorem max_coins_distribution (k : ℕ) : 
  (∀ n : ℕ, n < 100 ∧ ∃ k : ℕ, n = 13 * k + 3) → 
  (∀ m : ℕ, m < 100 ∧ ∃ k : ℕ, m = 13 * k + 3 → m ≤ 91) ∧
  (∃ k : ℕ, 91 = 13 * k + 3) ∧ 
  91 < 100 :=
by sorry

end max_coins_distribution_l3746_374640


namespace handshakes_in_social_event_l3746_374619

/-- Represents a social event with two groups of people -/
structure SocialEvent where
  totalPeople : Nat
  group1Size : Nat
  group2Size : Nat
  knownInGroup1 : Nat
  knownInGroup2 : Nat

/-- Calculates the number of handshakes in a social event -/
def calculateHandshakes (event : SocialEvent) : Nat :=
  let group1Handshakes := event.group1Size * (event.totalPeople - event.group1Size + event.knownInGroup1)
  let group2Handshakes := event.group2Size * (event.totalPeople - event.group2Size + event.knownInGroup2)
  (group1Handshakes + group2Handshakes) / 2

/-- Theorem stating that the number of handshakes in the given social event is 630 -/
theorem handshakes_in_social_event :
  let event : SocialEvent := {
    totalPeople := 40,
    group1Size := 25,
    group2Size := 15,
    knownInGroup1 := 18,
    knownInGroup2 := 4
  }
  calculateHandshakes event = 630 := by
  sorry


end handshakes_in_social_event_l3746_374619


namespace p_minus_q_empty_iff_a_nonneg_l3746_374698

/-- The set P as defined in the problem -/
def P : Set ℝ :=
  {y | ∃ x, 1 - Real.sqrt 2 / 2 < x ∧ x < 3/2 ∧ y = -x^2 + 2*x - 1/2}

/-- The set Q as defined in the problem -/
def Q (a : ℝ) : Set ℝ :=
  {x | x^2 + (a-1)*x - a < 0}

/-- The main theorem stating the equivalence between P - Q being empty and a being in [0, +∞) -/
theorem p_minus_q_empty_iff_a_nonneg (a : ℝ) :
  P \ Q a = ∅ ↔ a ∈ Set.Ici 0 := by sorry

end p_minus_q_empty_iff_a_nonneg_l3746_374698


namespace sin_alpha_plus_seven_pi_sixth_l3746_374634

theorem sin_alpha_plus_seven_pi_sixth (α : ℝ) 
  (h : Real.sin α + Real.cos (α - π / 6) = Real.sqrt 3 / 3) : 
  Real.sin (α + 7 * π / 6) = -1 / 3 := by
  sorry

end sin_alpha_plus_seven_pi_sixth_l3746_374634


namespace largest_angle_in_special_triangle_l3746_374685

theorem largest_angle_in_special_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →  -- angles are positive
    a + b + c = 180 →        -- sum of angles is 180°
    b = 3 * a →              -- ratio condition
    c = 4 * a →              -- ratio condition
    c = 90 :=                -- largest angle is 90°
by sorry

end largest_angle_in_special_triangle_l3746_374685


namespace distances_to_other_vertices_l3746_374673

/-- A circle with radius 5 and an inscribed square -/
structure CircleSquare where
  center : ℝ × ℝ
  radius : ℝ
  square_vertices : Fin 4 → ℝ × ℝ

/-- A point on the circle -/
def PointOnCircle (cs : CircleSquare) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the distances to other vertices -/
theorem distances_to_other_vertices (cs : CircleSquare) 
  (h_radius : cs.radius = 5)
  (h_inscribed : ∀ v, distance cs.center (cs.square_vertices v) = cs.radius)
  (h_on_circle : distance cs.center (PointOnCircle cs) = cs.radius)
  (h_distance_to_one : ∃ v, distance (PointOnCircle cs) (cs.square_vertices v) = 6) :
  ∃ (v1 v2 v3 : Fin 4), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
    distance (PointOnCircle cs) (cs.square_vertices v1) = Real.sqrt 2 ∧
    distance (PointOnCircle cs) (cs.square_vertices v2) = 8 ∧
    distance (PointOnCircle cs) (cs.square_vertices v3) = 7 * Real.sqrt 2 :=
  sorry

end distances_to_other_vertices_l3746_374673


namespace tangent_line_determines_n_l3746_374603

/-- A curve defined by a cubic function -/
structure CubicCurve where
  m : ℝ
  n : ℝ

/-- A line defined by a linear function -/
structure Line where
  k : ℝ

/-- Checks if a line is tangent to a cubic curve at a given point -/
def is_tangent_at (c : CubicCurve) (l : Line) (x₀ y₀ : ℝ) : Prop :=
  y₀ = x₀^3 + c.m * x₀ + c.n ∧
  y₀ = l.k * x₀ + 2 ∧
  3 * x₀^2 + c.m = l.k

theorem tangent_line_determines_n (c : CubicCurve) (l : Line) :
  is_tangent_at c l 1 4 → c.n = 4 := by sorry

end tangent_line_determines_n_l3746_374603


namespace ricks_sisters_cards_l3746_374686

/-- The number of cards Rick's sisters receive -/
def cards_per_sister (total_cards : ℕ) (kept_cards : ℕ) (miguel_cards : ℕ) 
  (num_friends : ℕ) (cards_per_friend : ℕ) (num_sisters : ℕ) : ℕ :=
  let remaining_cards := total_cards - kept_cards - miguel_cards - (num_friends * cards_per_friend)
  remaining_cards / num_sisters

/-- Proof that each of Rick's sisters received 3 cards -/
theorem ricks_sisters_cards : 
  cards_per_sister 130 15 13 8 12 2 = 3 := by
  sorry

end ricks_sisters_cards_l3746_374686


namespace election_result_l3746_374641

theorem election_result (total_votes : ℕ) (winner_votes first_opponent_votes second_opponent_votes third_opponent_votes : ℕ)
  (h1 : total_votes = 963)
  (h2 : winner_votes = 195)
  (h3 : first_opponent_votes = 142)
  (h4 : second_opponent_votes = 116)
  (h5 : third_opponent_votes = 90)
  (h6 : total_votes = winner_votes + first_opponent_votes + second_opponent_votes + third_opponent_votes) :
  winner_votes - first_opponent_votes = 53 := by
  sorry

end election_result_l3746_374641


namespace sin_cos_sum_17_13_l3746_374635

theorem sin_cos_sum_17_13 : 
  Real.sin (17 * π / 180) * Real.cos (13 * π / 180) + 
  Real.cos (17 * π / 180) * Real.sin (13 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_sum_17_13_l3746_374635


namespace negation_of_existence_l3746_374654

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x, x^2 - 1 < 0) ↔ (∀ x, x^2 - 1 ≥ 0) :=
by sorry

end negation_of_existence_l3746_374654


namespace flagpole_breaking_point_l3746_374648

theorem flagpole_breaking_point (h : ℝ) (b : ℝ) (t : ℝ) :
  h = 12 ∧ t = 2 ∧ b > 0 →
  b^2 + (h - t)^2 = h^2 →
  b = 2 * Real.sqrt 11 :=
by sorry

end flagpole_breaking_point_l3746_374648


namespace unique_cookie_distribution_l3746_374614

/-- Represents the number of cookies eaten by each sibling -/
structure CookieDistribution where
  ben : ℕ
  mia : ℕ
  leo : ℕ

/-- Checks if a cookie distribution satisfies the problem conditions -/
def isValidDistribution (d : CookieDistribution) : Prop :=
  d.ben + d.mia + d.leo = 30 ∧
  d.mia = 2 * d.ben ∧
  d.leo = d.ben + d.mia

/-- The correct cookie distribution -/
def correctDistribution : CookieDistribution :=
  { ben := 5, mia := 10, leo := 15 }

/-- Theorem stating that the correct distribution is the only valid one -/
theorem unique_cookie_distribution :
  isValidDistribution correctDistribution ∧
  ∀ d : CookieDistribution, isValidDistribution d → d = correctDistribution :=
sorry

end unique_cookie_distribution_l3746_374614


namespace clock_chime_theorem_l3746_374681

/-- Represents the number of chimes at a given time -/
def num_chimes (hour : ℕ) (minute : ℕ) : ℕ :=
  if minute = 0 then hour % 12
  else if minute = 30 then 1
  else 0

/-- Represents a sequence of four consecutive chimes -/
def chime_sequence (start_hour : ℕ) (start_minute : ℕ) : Prop :=
  num_chimes start_hour start_minute = 1 ∧
  num_chimes ((start_hour + (start_minute + 30) / 60) % 24) ((start_minute + 30) % 60) = 1 ∧
  num_chimes ((start_hour + (start_minute + 60) / 60) % 24) ((start_minute + 60) % 60) = 1 ∧
  num_chimes ((start_hour + (start_minute + 90) / 60) % 24) ((start_minute + 90) % 60) = 1

theorem clock_chime_theorem :
  ∀ (start_hour : ℕ) (start_minute : ℕ),
    chime_sequence start_hour start_minute →
    start_hour = 12 ∧ start_minute = 0 :=
by sorry

end clock_chime_theorem_l3746_374681


namespace overlapping_squares_area_l3746_374625

def rotation_angle (α : Real) : Prop :=
  0 < α ∧ α < Real.pi / 2 ∧ Real.cos α = 4 / 5

def overlapping_area (α : Real) : Real :=
  -- Definition of the overlapping area function
  sorry

theorem overlapping_squares_area (α : Real) 
  (h : rotation_angle α) : overlapping_area α = 1 / 2 := by
  sorry

end overlapping_squares_area_l3746_374625


namespace correct_parentheses_removal_l3746_374630

theorem correct_parentheses_removal (x : ℝ) : -0.5 * (1 - 2 * x) = -0.5 + x := by
  sorry

end correct_parentheses_removal_l3746_374630


namespace overlapping_area_is_75_over_8_l3746_374679

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ

/-- The area of the overlapping region formed by two 30-60-90 triangles -/
def overlapping_area (t1 t2 : Triangle30_60_90) : ℝ :=
  sorry

/-- The theorem stating the area of the overlapping region -/
theorem overlapping_area_is_75_over_8 (t1 t2 : Triangle30_60_90) 
  (h1 : t1.hypotenuse = 10)
  (h2 : t2.hypotenuse = 10)
  (h3 : overlapping_area t1 t2 ≠ 0) : 
  overlapping_area t1 t2 = 75 / 8 := by
  sorry

end overlapping_area_is_75_over_8_l3746_374679


namespace largest_divisor_of_n4_minus_n2_l3746_374669

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- The theorem stating that 6n^2 is the largest divisor of n^4 - n^2 for all composite n -/
theorem largest_divisor_of_n4_minus_n2 (n : ℕ) (h : IsComposite n) :
  (∃ (k : ℕ), (n^4 - n^2) % (6 * n^2) = 0 ∧
    ∀ (m : ℕ), (n^4 - n^2) % m = 0 → m ≤ 6 * n^2) :=
sorry

end largest_divisor_of_n4_minus_n2_l3746_374669


namespace triangle_side_less_than_half_perimeter_l3746_374653

theorem triangle_side_less_than_half_perimeter (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a < (a + b + c) / 2 ∧ b < (a + b + c) / 2 ∧ c < (a + b + c) / 2 := by
  sorry

end triangle_side_less_than_half_perimeter_l3746_374653


namespace absolute_value_simplification_l3746_374680

theorem absolute_value_simplification : |(-4^2 + 6)| = 10 := by sorry

end absolute_value_simplification_l3746_374680


namespace marbles_given_to_juan_l3746_374644

theorem marbles_given_to_juan (initial_marbles : ℕ) (remaining_marbles : ℕ) 
  (h1 : initial_marbles = 73)
  (h2 : remaining_marbles = 3) :
  initial_marbles - remaining_marbles = 70 := by
  sorry

end marbles_given_to_juan_l3746_374644

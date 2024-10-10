import Mathlib

namespace intersection_line_of_circles_l970_97064

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line passing through the intersection points of two circles -/
def intersectionLine (c1 c2 : Circle) : ℝ → ℝ → Prop :=
  fun x y => x + y = 6.5

theorem intersection_line_of_circles :
  let c1 : Circle := { center := (5, -2), radius := 7 }
  let c2 : Circle := { center := (-1, 5), radius := 5 }
  ∃ (p1 p2 : ℝ × ℝ),
    (p1.1 - c1.center.1)^2 + (p1.2 - c1.center.2)^2 = c1.radius^2 ∧
    (p1.1 - c2.center.1)^2 + (p1.2 - c2.center.2)^2 = c2.radius^2 ∧
    (p2.1 - c1.center.1)^2 + (p2.2 - c1.center.2)^2 = c1.radius^2 ∧
    (p2.1 - c2.center.1)^2 + (p2.2 - c2.center.2)^2 = c2.radius^2 ∧
    p1 ≠ p2 ∧
    intersectionLine c1 c2 p1.1 p1.2 ∧
    intersectionLine c1 c2 p2.1 p2.2 :=
by
  sorry

end intersection_line_of_circles_l970_97064


namespace min_value_of_f_l970_97038

-- Define the function
def f (y : ℝ) : ℝ := 3 * y^2 - 18 * y + 11

-- State the theorem
theorem min_value_of_f :
  ∃ (y_min : ℝ), ∀ (y : ℝ), f y ≥ f y_min ∧ f y_min = -16 :=
sorry

end min_value_of_f_l970_97038


namespace file_storage_problem_l970_97035

/-- Represents the minimum number of disks required to store files -/
def min_disks (total_files : ℕ) (disk_capacity : ℚ) 
  (files_size_1 : ℕ) (size_1 : ℚ)
  (files_size_2 : ℕ) (size_2 : ℚ)
  (size_3 : ℚ) : ℕ :=
  sorry

theorem file_storage_problem :
  let total_files : ℕ := 33
  let disk_capacity : ℚ := 1.44
  let files_size_1 : ℕ := 3
  let size_1 : ℚ := 1.1
  let files_size_2 : ℕ := 15
  let size_2 : ℚ := 0.6
  let size_3 : ℚ := 0.5
  let remaining_files : ℕ := total_files - files_size_1 - files_size_2
  min_disks total_files disk_capacity files_size_1 size_1 files_size_2 size_2 size_3 = 17 :=
by sorry

end file_storage_problem_l970_97035


namespace quadratic_roots_range_l970_97085

theorem quadratic_roots_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) →
  a < 1 ∧ a ≠ 0 :=
by sorry

end quadratic_roots_range_l970_97085


namespace all_pairs_product_48_l970_97000

theorem all_pairs_product_48 : 
  ((-6) * (-8) = 48) ∧
  ((-4) * (-12) = 48) ∧
  ((3/2 : ℚ) * 32 = 48) ∧
  (2 * 24 = 48) ∧
  ((4/3 : ℚ) * 36 = 48) := by
  sorry

end all_pairs_product_48_l970_97000


namespace reflection_theorem_l970_97090

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a given line --/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Checks if two points are symmetric with respect to the line x + y = 0 --/
def symmetricPoints (p1 p2 : Point) : Prop :=
  p2.x = -p1.y ∧ p2.y = -p1.x

/-- The main theorem to prove --/
theorem reflection_theorem :
  ∀ (b : ℝ),
  let incident_ray : Line := { slope := -3, intercept := b }
  let reflected_ray : Line := { slope := -1/3, intercept := 2 }
  let incident_point : Point := { x := 1, y := b - 3 }
  let reflected_point : Point := { x := -b + 3, y := -1 }
  pointOnLine incident_point incident_ray ∧
  pointOnLine reflected_point reflected_ray ∧
  symmetricPoints incident_point reflected_point →
  b = -6 := by
sorry

end reflection_theorem_l970_97090


namespace doraemon_toys_count_l970_97033

theorem doraemon_toys_count : ∃! n : ℕ, 40 ≤ n ∧ n ≤ 55 ∧ (n - 3) % 5 = 0 ∧ (n + 2) % 3 = 0 := by
  sorry

end doraemon_toys_count_l970_97033


namespace interior_perimeter_is_14_l970_97099

/-- Represents a rectangular picture frame -/
structure PictureFrame where
  outerWidth : ℝ
  outerHeight : ℝ
  frameWidth : ℝ

/-- Calculates the area of just the frame -/
def frameArea (frame : PictureFrame) : ℝ :=
  frame.outerWidth * frame.outerHeight - (frame.outerWidth - 2 * frame.frameWidth) * (frame.outerHeight - 2 * frame.frameWidth)

/-- Calculates the sum of the lengths of the four interior edges -/
def interiorPerimeter (frame : PictureFrame) : ℝ :=
  2 * (frame.outerWidth - 2 * frame.frameWidth) + 2 * (frame.outerHeight - 2 * frame.frameWidth)

/-- Theorem: Given the conditions, the sum of interior edges is 14 inches -/
theorem interior_perimeter_is_14 (frame : PictureFrame) 
  (h1 : frame.frameWidth = 1)
  (h2 : frameArea frame = 18)
  (h3 : frame.outerWidth = 5) :
  interiorPerimeter frame = 14 := by
  sorry

end interior_perimeter_is_14_l970_97099


namespace magic_8_ball_probability_l970_97008

theorem magic_8_ball_probability :
  let n : ℕ := 7  -- total number of questions
  let k : ℕ := 4  -- number of positive answers
  let p : ℚ := 3/7  -- probability of a positive answer
  Nat.choose n k * p^k * (1-p)^(n-k) = 181440/823543 :=
by sorry

end magic_8_ball_probability_l970_97008


namespace point_on_line_between_l970_97020

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Check if a point is between two other points -/
def between (p q r : Point) : Prop :=
  collinear p q r ∧
  min p.x r.x ≤ q.x ∧ q.x ≤ max p.x r.x ∧
  min p.y r.y ≤ q.y ∧ q.y ≤ max p.y r.y

theorem point_on_line_between (p₁ p₂ q : Point) 
  (h₁ : p₁ = ⟨8, 16⟩) 
  (h₂ : p₂ = ⟨2, 6⟩)
  (h₃ : q = ⟨5, 11⟩) : 
  between p₁ q p₂ := by
  sorry

end point_on_line_between_l970_97020


namespace inverse_variation_problem_l970_97019

theorem inverse_variation_problem (y x : ℝ) (k : ℝ) :
  (∀ x y, y * x^2 = k) →  -- y varies inversely as x^2
  (1 * 4^2 = k) →         -- when x = 4, y = 1
  (0.25 * x^2 = k) →      -- condition for y = 0.25
  x = 8 :=                -- prove x = 8
by
  sorry

#check inverse_variation_problem

end inverse_variation_problem_l970_97019


namespace special_pizza_all_toppings_l970_97001

/-- Represents a pizza with various toppings -/
structure Pizza where
  total_slices : ℕ
  pepperoni_slices : ℕ
  mushroom_slices : ℕ
  olive_slices : ℕ
  all_toppings_slices : ℕ

/-- Conditions for our specific pizza -/
def special_pizza : Pizza := {
  total_slices := 24,
  pepperoni_slices := 15,
  mushroom_slices := 16,
  olive_slices := 10,
  all_toppings_slices := 2
}

/-- Every slice has at least one topping -/
def has_at_least_one_topping (p : Pizza) : Prop :=
  p.pepperoni_slices + p.mushroom_slices + p.olive_slices - p.all_toppings_slices ≥ p.total_slices

/-- The theorem to prove -/
theorem special_pizza_all_toppings :
  has_at_least_one_topping special_pizza ∧
  special_pizza.all_toppings_slices = 2 :=
sorry


end special_pizza_all_toppings_l970_97001


namespace x_varies_as_z_l970_97087

-- Define the variables and constants
variable (x y z : ℝ)
variable (k j : ℝ)

-- Define the conditions
axiom x_varies_as_y : ∃ k, x = k * y^3
axiom y_varies_as_z : ∃ j, y = j * z^(1/4)

-- Define the theorem to prove
theorem x_varies_as_z : ∃ m, x = m * z^(3/4) := by
  sorry

end x_varies_as_z_l970_97087


namespace restaurant_menu_combinations_l970_97029

theorem restaurant_menu_combinations (n : ℕ) (h : n = 12) :
  n * (n - 1) = 132 := by
  sorry

end restaurant_menu_combinations_l970_97029


namespace coin_flip_probability_l970_97047

/-- The number of coins being flipped -/
def num_coins : ℕ := 5

/-- The number of coins that need to match -/
def num_matching : ℕ := 3

/-- The number of possible outcomes for each coin -/
def outcomes_per_coin : ℕ := 2

/-- The total number of possible outcomes when flipping the coins -/
def total_outcomes : ℕ := outcomes_per_coin ^ num_coins

/-- The number of successful outcomes where the specified coins match -/
def successful_outcomes : ℕ := outcomes_per_coin * outcomes_per_coin ^ (num_coins - num_matching)

/-- The probability of the specified coins matching -/
def probability : ℚ := successful_outcomes / total_outcomes

theorem coin_flip_probability : probability = 1 / 4 := by sorry

end coin_flip_probability_l970_97047


namespace tree_distribution_l970_97022

/-- The number of ways to distribute n indistinguishable objects into k distinct groups,
    with each group containing at least one object -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 10 trees over 3 days with at least one tree per day -/
theorem tree_distribution : distribute 10 3 = 36 := by
  sorry

end tree_distribution_l970_97022


namespace batsman_score_difference_l970_97026

/-- Given a batsman's statistics, prove the difference between highest and lowest scores -/
theorem batsman_score_difference
  (total_innings : ℕ)
  (total_runs : ℕ)
  (excluded_innings : ℕ)
  (excluded_runs : ℕ)
  (highest_score : ℕ)
  (h_total_innings : total_innings = 46)
  (h_excluded_innings : excluded_innings = 44)
  (h_total_runs : total_runs = 60 * total_innings)
  (h_excluded_runs : excluded_runs = 58 * excluded_innings)
  (h_highest_score : highest_score = 174) :
  highest_score - (total_runs - excluded_runs - highest_score) = 140 :=
by sorry

end batsman_score_difference_l970_97026


namespace polynomial_factorization_l970_97052

theorem polynomial_factorization (x : ℝ) : x^2 - x = x * (x - 1) := by
  sorry

end polynomial_factorization_l970_97052


namespace circle_equation_l970_97095

theorem circle_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (4, 6)
  let center_line (x y : ℝ) := x - 2*y - 1 = 0
  ∃ (h k : ℝ), 
    center_line h k ∧ 
    (h - A.1)^2 + (k - A.2)^2 = (h - B.1)^2 + (k - B.2)^2 ∧
    (x - h)^2 + (y - k)^2 = 17 := by
  sorry

end circle_equation_l970_97095


namespace blue_to_red_ratio_is_four_to_one_l970_97079

/-- Represents the number of pencils of each color and the total number of pencils. -/
structure PencilCounts where
  total : ℕ
  red : ℕ
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- Theorem stating that under given conditions, the ratio of blue to red pencils is 4:1. -/
theorem blue_to_red_ratio_is_four_to_one (p : PencilCounts)
    (h_total : p.total = 160)
    (h_red : p.red = 20)
    (h_yellow : p.yellow = 40)
    (h_green : p.green = p.red + p.blue) :
    p.blue / p.red = 4 := by
  sorry

end blue_to_red_ratio_is_four_to_one_l970_97079


namespace day_50_of_previous_year_is_thursday_l970_97044

/-- Represents the days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year -/
structure Year where
  number : ℕ

/-- Returns the day of the week for a given day in a year -/
def dayOfWeek (y : Year) (day : ℕ) : DayOfWeek :=
  sorry

/-- Returns the number of days in a year -/
def daysInYear (y : Year) : ℕ :=
  sorry

theorem day_50_of_previous_year_is_thursday
  (N : Year)
  (h1 : dayOfWeek N 250 = DayOfWeek.Friday)
  (h2 : dayOfWeek (Year.mk (N.number + 1)) 150 = DayOfWeek.Friday) :
  dayOfWeek (Year.mk (N.number - 1)) 50 = DayOfWeek.Thursday :=
sorry

end day_50_of_previous_year_is_thursday_l970_97044


namespace total_poultry_count_l970_97058

def poultry_farm (num_hens num_ducks num_geese : ℕ) 
                 (male_female_ratio : ℚ) 
                 (chicks_per_hen ducklings_per_duck goslings_per_goose : ℕ) : ℕ :=
  let female_hens := (num_hens * 4) / 5
  let female_ducks := (num_ducks * 4) / 5
  let female_geese := (num_geese * 4) / 5
  let total_chicks := female_hens * chicks_per_hen
  let total_ducklings := female_ducks * ducklings_per_duck
  let total_goslings := female_geese * goslings_per_goose
  num_hens + num_ducks + num_geese + total_chicks + total_ducklings + total_goslings

theorem total_poultry_count : 
  poultry_farm 25 10 5 (1/4) 6 8 3 = 236 := by
  sorry

end total_poultry_count_l970_97058


namespace batsman_highest_score_l970_97010

def batting_problem (total_innings : ℕ) (overall_average : ℚ) (score_difference : ℕ) (average_excluding_extremes : ℚ) : Prop :=
  let total_runs := total_innings * overall_average
  let runs_excluding_extremes := (total_innings - 2) * average_excluding_extremes
  let sum_of_extremes := total_runs - runs_excluding_extremes
  let highest_score := (sum_of_extremes + score_difference) / 2
  highest_score = 199

theorem batsman_highest_score :
  batting_problem 46 60 190 58 := by sorry

end batsman_highest_score_l970_97010


namespace outfit_combinations_l970_97063

/-- Represents the number of shirts Li Fang has -/
def num_shirts : ℕ := 4

/-- Represents the number of skirts Li Fang has -/
def num_skirts : ℕ := 3

/-- Represents the number of dresses Li Fang has -/
def num_dresses : ℕ := 2

/-- Calculates the total number of outfit combinations -/
def total_outfits : ℕ := num_shirts * num_skirts + num_dresses

/-- Theorem stating that the total number of outfit combinations is 14 -/
theorem outfit_combinations : total_outfits = 14 := by
  sorry

end outfit_combinations_l970_97063


namespace roots_are_imaginary_l970_97042

theorem roots_are_imaginary (m : ℝ) : 
  (∃ x y : ℂ, x^2 - 4*m*x + 5*m^2 + 2 = 0 ∧ y^2 - 4*m*y + 5*m^2 + 2 = 0 ∧ x*y = 9) →
  (∃ a b : ℝ, a ≠ 0 ∧ (∀ z : ℂ, z^2 - 4*m*z + 5*m^2 + 2 = 0 → ∃ r : ℝ, z = Complex.mk r (a*r + b) ∨ z = Complex.mk r (-a*r - b))) :=
by sorry

end roots_are_imaginary_l970_97042


namespace function_inequality_implies_a_range_l970_97073

/-- The function f(x) = x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The function g(x) = e^x -/
noncomputable def g (x : ℝ) : ℝ := Real.exp x

/-- The theorem statement -/
theorem function_inequality_implies_a_range :
  ∀ a : ℝ, 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → x₁ ≠ x₂ → 
    |f a x₁ - f a x₂| < |g x₁ - g x₂|) →
  a ∈ Set.Icc (-1) (2 - 2 * Real.log 2) :=
by sorry

end function_inequality_implies_a_range_l970_97073


namespace x_2023_minus_1_values_l970_97014

theorem x_2023_minus_1_values (x : ℝ) : 
  (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 → 
  x^2023 - 1 = 0 ∨ x^2023 - 1 = -2 := by
sorry

end x_2023_minus_1_values_l970_97014


namespace smallest_lcm_with_gcd_4_l970_97068

theorem smallest_lcm_with_gcd_4 :
  ∃ (m n : ℕ),
    1000 ≤ m ∧ m < 10000 ∧
    1000 ≤ n ∧ n < 10000 ∧
    Nat.gcd m n = 4 ∧
    Nat.lcm m n = 252912 ∧
    ∀ (a b : ℕ),
      1000 ≤ a ∧ a < 10000 ∧
      1000 ≤ b ∧ b < 10000 ∧
      Nat.gcd a b = 4 →
      Nat.lcm a b ≥ 252912 :=
sorry

end smallest_lcm_with_gcd_4_l970_97068


namespace largest_value_is_2_pow_35_l970_97055

theorem largest_value_is_2_pow_35 : 
  (2 ^ 35 : ℕ) > 26 ∧ (2 ^ 35 : ℕ) > 1 := by
  sorry

end largest_value_is_2_pow_35_l970_97055


namespace parallelogram_with_equal_vector_sums_is_rectangle_l970_97069

/-- A parallelogram ABCD with vertices A, B, C, and D. -/
structure Parallelogram (V : Type*) [NormedAddCommGroup V] :=
  (A B C D : V)
  (is_parallelogram : (B - A) = (C - D) ∧ (D - A) = (C - B))

/-- Definition of a rectangle as a parallelogram with equal diagonals. -/
def is_rectangle {V : Type*} [NormedAddCommGroup V] (p : Parallelogram V) : Prop :=
  ‖p.C - p.A‖ = ‖p.D - p.B‖

theorem parallelogram_with_equal_vector_sums_is_rectangle
  {V : Type*} [NormedAddCommGroup V] (p : Parallelogram V) :
  ‖p.B - p.A + (p.D - p.A)‖ = ‖p.B - p.A - (p.D - p.A)‖ →
  is_rectangle p :=
sorry

end parallelogram_with_equal_vector_sums_is_rectangle_l970_97069


namespace magic_square_b_plus_c_l970_97084

/-- Represents a 3x3 magic square with the given layout -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  S : ℕ
  row1_sum : 30 + b + 18 = S
  row2_sum : 15 + c + d = S
  row3_sum : a + 33 + e = S
  col1_sum : 30 + 15 + a = S
  col2_sum : b + c + 33 = S
  col3_sum : 18 + d + e = S
  diag1_sum : 30 + c + e = S
  diag2_sum : 18 + c + a = S

/-- The sum of b and c in a magic square is 33 -/
theorem magic_square_b_plus_c (ms : MagicSquare) : ms.b + ms.c = 33 := by
  sorry

end magic_square_b_plus_c_l970_97084


namespace complex_power_difference_l970_97067

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^40 - (1 - i)^40 = 0 := by
  sorry

end complex_power_difference_l970_97067


namespace smallest_positive_integer_3003m_55555n_specific_solution_3003m_55555n_l970_97049

theorem smallest_positive_integer_3003m_55555n :
  ∃ (m n : ℤ), 3003 * m + 55555 * n = 1 ∧
  ∀ (k l : ℤ), 3003 * k + 55555 * l > 0 → 3003 * k + 55555 * l ≥ 1 :=
by sorry

theorem specific_solution_3003m_55555n :
  3003 * 37 + 55555 * (-2) = 1 :=
by sorry

end smallest_positive_integer_3003m_55555n_specific_solution_3003m_55555n_l970_97049


namespace percentage_calculation_l970_97088

theorem percentage_calculation (P : ℝ) : 
  (0.47 * 1442 - P / 100 * 1412) + 65 = 5 → P = 52.24 := by
  sorry

end percentage_calculation_l970_97088


namespace petya_friends_count_l970_97078

/-- The number of friends Petya has -/
def num_friends : ℕ := 19

/-- The number of stickers Petya has -/
def total_stickers : ℕ := num_friends * 5 + 8

theorem petya_friends_count :
  (total_stickers = num_friends * 5 + 8) ∧
  (total_stickers = num_friends * 6 - 11) →
  num_friends = 19 := by
sorry

end petya_friends_count_l970_97078


namespace magician_trick_exists_strategy_l970_97030

/-- Represents a card placement strategy for the magician's trick -/
structure CardPlacementStrategy (n : ℕ) :=
  (place_cards : Fin n → Fin n)
  (deduce_card1 : Fin n → Fin n → Fin n)
  (deduce_card2 : Fin n → Fin n → Fin n)

/-- The main theorem stating that a successful strategy exists for all n ≥ 3 -/
theorem magician_trick_exists_strategy (n : ℕ) (h : n ≥ 3) :
  ∃ (strategy : CardPlacementStrategy n),
    ∀ (card1_pos card2_pos : Fin n),
      card1_pos ≠ card2_pos →
      ∀ (magician_reveal spectator_reveal : Fin n),
        magician_reveal ≠ spectator_reveal →
        strategy.deduce_card1 magician_reveal spectator_reveal = card1_pos ∧
        strategy.deduce_card2 magician_reveal spectator_reveal = card2_pos :=
sorry

end magician_trick_exists_strategy_l970_97030


namespace original_price_of_discounted_items_l970_97007

theorem original_price_of_discounted_items 
  (num_items : ℕ) 
  (discount_rate : ℚ) 
  (total_paid : ℚ) 
  (h1 : num_items = 6)
  (h2 : discount_rate = 1/2)
  (h3 : total_paid = 60) :
  (total_paid / (1 - discount_rate)) / num_items = 20 := by
sorry

end original_price_of_discounted_items_l970_97007


namespace penny_bakery_revenue_l970_97048

/-- Calculates the total money made from selling cheesecakes -/
def total_money_made (price_per_slice : ℕ) (slices_per_cake : ℕ) (cakes_sold : ℕ) : ℕ :=
  price_per_slice * slices_per_cake * cakes_sold

/-- Theorem: Penny's bakery makes $294 from selling 7 cheesecakes -/
theorem penny_bakery_revenue : total_money_made 7 6 7 = 294 := by
  sorry

end penny_bakery_revenue_l970_97048


namespace range_of_a_l970_97054

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 - x + a

-- Define the range of f as set B
def B (a : ℝ) : Set ℝ := {y : ℝ | ∃ x ∈ A, f a x = y}

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ A, f a x ∈ A) → a = -1 :=
sorry

end range_of_a_l970_97054


namespace problem_solution_l970_97028

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 3*a

-- Define the conditions
def condition1 (a : ℝ) (m : ℝ) : Prop :=
  ∀ x, f a x < 0 ↔ 1 < x ∧ x < m

def condition2 (a : ℝ) : Prop :=
  ∀ x, f a x > 0

def condition3 (a : ℝ) (k : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc 0 1 → a^(k+3) < a^(x^2-k*x) ∧ a^(x^2-k*x) < a^(k-3)

-- State the theorem
theorem problem_solution (a m k : ℝ) :
  condition1 a m →
  condition2 a →
  condition3 a k →
  (a = 1 ∧ m = 3) ∧
  (-1 < k ∧ k < -2 + Real.sqrt 7) :=
sorry

end problem_solution_l970_97028


namespace sum_of_powers_l970_97096

theorem sum_of_powers (ω : ℂ) (h1 : ω^11 = 1) (h2 : ω ≠ 1) :
  ω^10 + ω^14 + ω^18 + ω^22 + ω^26 + ω^30 + ω^34 + ω^38 + ω^42 + ω^46 + ω^50 + ω^54 + ω^58 = 1 := by
  sorry

end sum_of_powers_l970_97096


namespace pet_store_ratio_l970_97092

theorem pet_store_ratio (dogs : ℕ) (total : ℕ) : 
  dogs = 6 → 
  total = 39 → 
  (dogs + dogs / 2 + 2 * dogs + (total - (dogs + dogs / 2 + 2 * dogs))) / dogs = 3 := by
  sorry

end pet_store_ratio_l970_97092


namespace xyz_equals_five_l970_97080

theorem xyz_equals_five
  (a b c x y z : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (eq_a : a = (b + c) / (x - 2))
  (eq_b : b = (a + c) / (y - 2))
  (eq_c : c = (a + b) / (z - 2))
  (sum_xy_xz_yz : x * y + x * z + y * z = 5)
  (sum_x_y_z : x + y + z = 3) :
  x * y * z = 5 := by
sorry

end xyz_equals_five_l970_97080


namespace factor_condition_l970_97024

theorem factor_condition (x t : ℝ) : 
  (∃ k : ℝ, 6 * x^2 + 13 * x - 5 = (x - t) * k) ↔ (t = -5/2 ∨ t = 1/3) := by
  sorry

end factor_condition_l970_97024


namespace sum_first_seven_odd_numbers_l970_97050

def sum_odd_numbers (n : ℕ) : ℕ := (2 * n - 1) * n

theorem sum_first_seven_odd_numbers :
  (sum_odd_numbers 2 = 2^2) →
  (sum_odd_numbers 5 = 5^2) →
  (sum_odd_numbers 7 = 7^2) :=
by
  sorry

end sum_first_seven_odd_numbers_l970_97050


namespace shaded_percentage_of_grid_l970_97051

theorem shaded_percentage_of_grid (total_squares : Nat) (shaded_squares : Nat) :
  total_squares = 25 →
  shaded_squares = 13 →
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100 = 52 := by
  sorry

end shaded_percentage_of_grid_l970_97051


namespace sticks_in_yard_l970_97059

theorem sticks_in_yard (picked_up left : ℕ) 
  (h1 : picked_up = 38) 
  (h2 : left = 61) : 
  picked_up + left = 99 := by
  sorry

end sticks_in_yard_l970_97059


namespace area_at_stage_6_l970_97034

/-- The side length of each square -/
def square_side : ℕ := 3

/-- The number of stages -/
def num_stages : ℕ := 6

/-- The area of the rectangle at a given stage -/
def rectangle_area (stage : ℕ) : ℕ :=
  stage * square_side * square_side

/-- Theorem: The area of the rectangle at Stage 6 is 54 square inches -/
theorem area_at_stage_6 : rectangle_area num_stages = 54 := by
  sorry

end area_at_stage_6_l970_97034


namespace point_division_theorem_l970_97097

/-- Given a line segment AB and a point P on AB such that AP:PB = 3:4,
    prove that P = (4/7)*A + (3/7)*B -/
theorem point_division_theorem (A B P : ℝ × ℝ) :
  (P.1 - A.1) / (B.1 - P.1) = 3 / 4 ∧
  (P.2 - A.2) / (B.2 - P.2) = 3 / 4 →
  P = ((4:ℝ)/7) • A + ((3:ℝ)/7) • B :=
sorry

end point_division_theorem_l970_97097


namespace set_equality_l970_97093

def M : Set ℝ := {x | ∃ n : ℤ, x = n}
def N : Set ℝ := {x | ∃ n : ℤ, x = n / 2}
def P : Set ℝ := {x | ∃ n : ℤ, x = n + 1 / 2}

theorem set_equality : N = M ∪ P := by sorry

end set_equality_l970_97093


namespace andrew_cookie_expenditure_l970_97057

/-- The number of days in May -/
def days_in_may : ℕ := 31

/-- The number of cookies Andrew purchases each day -/
def cookies_per_day : ℕ := 3

/-- The cost of each cookie in dollars -/
def cost_per_cookie : ℕ := 15

/-- The total amount Andrew spent on cookies in May -/
def total_spent : ℕ := days_in_may * cookies_per_day * cost_per_cookie

/-- Theorem stating that Andrew spent 1395 dollars on cookies in May -/
theorem andrew_cookie_expenditure : total_spent = 1395 := by
  sorry

end andrew_cookie_expenditure_l970_97057


namespace right_triangles_2012_characterization_l970_97027

/-- A right triangle with natural number side lengths where one leg is 2012 -/
structure RightTriangle2012 where
  other_leg : ℕ
  hypotenuse : ℕ
  is_right_triangle : other_leg ^ 2 + 2012 ^ 2 = hypotenuse ^ 2

/-- The set of all valid RightTriangle2012 -/
def all_right_triangles_2012 : Set RightTriangle2012 :=
  { t | t.other_leg > 0 ∧ t.hypotenuse > 0 }

/-- The four specific triangles mentioned in the problem -/
def specific_triangles : Set RightTriangle2012 :=
  { ⟨253005, 253013, by sorry⟩,
    ⟨506016, 506020, by sorry⟩,
    ⟨1012035, 1012037, by sorry⟩,
    ⟨1509, 2515, by sorry⟩ }

/-- The main theorem stating that the set of all valid right triangles with one leg 2012
    is equal to the set of four specific triangles -/
theorem right_triangles_2012_characterization :
  all_right_triangles_2012 = specific_triangles :=
sorry

end right_triangles_2012_characterization_l970_97027


namespace wendy_sales_l970_97037

/-- Represents the sales data for a fruit vendor --/
structure FruitSales where
  apple_price : ℝ
  orange_price : ℝ
  morning_apples : ℕ
  morning_oranges : ℕ
  afternoon_apples : ℕ
  afternoon_oranges : ℕ

/-- Calculates the total sales for a given FruitSales instance --/
def total_sales (sales : FruitSales) : ℝ :=
  let total_apples := sales.morning_apples + sales.afternoon_apples
  let total_oranges := sales.morning_oranges + sales.afternoon_oranges
  (total_apples : ℝ) * sales.apple_price + (total_oranges : ℝ) * sales.orange_price

/-- Theorem stating that the total sales for the given conditions equal $205 --/
theorem wendy_sales : 
  let sales := FruitSales.mk 1.5 1 40 30 50 40
  total_sales sales = 205 := by
  sorry


end wendy_sales_l970_97037


namespace simplify_and_evaluate_l970_97025

theorem simplify_and_evaluate (a : ℤ) (h : a = 2023) :
  a * (1 - 2 * a) + 2 * (a + 1) * (a - 1) = 2021 := by
  sorry

end simplify_and_evaluate_l970_97025


namespace senior_tickets_sold_l970_97098

/-- Proves the number of senior citizen tickets sold given the total tickets,
    ticket prices, and total receipts -/
theorem senior_tickets_sold
  (total_tickets : ℕ)
  (adult_price senior_price : ℕ)
  (total_receipts : ℕ)
  (h1 : total_tickets = 510)
  (h2 : adult_price = 21)
  (h3 : senior_price = 15)
  (h4 : total_receipts = 8748) :
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 327 :=
by sorry

end senior_tickets_sold_l970_97098


namespace corner_subset_exists_l970_97074

/-- A corner is a finite set of n-tuples of positive integers with a specific property. -/
def Corner (n : ℕ) : Type :=
  {S : Set (Fin n → ℕ+) // S.Finite ∧
    ∀ a b : Fin n → ℕ+, a ∈ S → (∀ k, b k ≤ a k) → b ∈ S}

/-- The theorem states that in any infinite collection of corners,
    there exist two corners where one is a subset of the other. -/
theorem corner_subset_exists {n : ℕ} (h : n > 0) (S : Set (Corner n)) (hS : Set.Infinite S) :
  ∃ C₁ C₂ : Corner n, C₁ ∈ S ∧ C₂ ∈ S ∧ C₁.1 ⊆ C₂.1 :=
sorry

end corner_subset_exists_l970_97074


namespace largest_n_for_factorization_l970_97041

theorem largest_n_for_factorization : ∃ (n : ℤ),
  (∀ m : ℤ, (∃ (a b c d : ℤ), 7 * X^2 + m * X + 56 = (a * X + b) * (c * X + d)) → m ≤ n) ∧
  (∃ (a b c d : ℤ), 7 * X^2 + n * X + 56 = (a * X + b) * (c * X + d)) ∧
  n = 393 :=
by sorry

end largest_n_for_factorization_l970_97041


namespace correct_num_spiders_l970_97018

/-- The number of spiders introduced to control pests in a garden --/
def num_spiders : ℕ := 12

/-- The initial number of bugs in the garden --/
def initial_bugs : ℕ := 400

/-- The number of bugs each spider eats --/
def bugs_per_spider : ℕ := 7

/-- The fraction of bugs remaining after spraying --/
def spray_factor : ℚ := 4/5

/-- The number of bugs remaining after pest control measures --/
def remaining_bugs : ℕ := 236

/-- Theorem stating that the number of spiders introduced is correct --/
theorem correct_num_spiders :
  (initial_bugs : ℚ) * spray_factor - (num_spiders : ℚ) * bugs_per_spider = remaining_bugs := by
  sorry

end correct_num_spiders_l970_97018


namespace log_equation_solution_l970_97072

theorem log_equation_solution (x : ℝ) :
  x > 0 →
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 →
  x = 4 ∨ x = 8 :=
by sorry

end log_equation_solution_l970_97072


namespace special_quadrilateral_is_square_l970_97031

/-- A quadrilateral with equal length diagonals that are perpendicular to each other -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has diagonals of equal length -/
  equal_diagonals : Bool
  /-- The diagonals are perpendicular to each other -/
  perpendicular_diagonals : Bool

/-- Definition of a square -/
def is_square (q : SpecialQuadrilateral) : Prop :=
  q.equal_diagonals ∧ q.perpendicular_diagonals

/-- Theorem stating that a quadrilateral with equal length diagonals that are perpendicular to each other is a square -/
theorem special_quadrilateral_is_square (q : SpecialQuadrilateral) 
  (h1 : q.equal_diagonals = true) 
  (h2 : q.perpendicular_diagonals = true) : 
  is_square q := by
  sorry

end special_quadrilateral_is_square_l970_97031


namespace tangent_circle_radius_l970_97070

/-- A circle tangent to the y-axis and a line, passing through a specific point -/
structure TangentCircle where
  -- The slope of the line the circle is tangent to
  slope : ℝ
  -- The point the circle passes through
  point : ℝ × ℝ

/-- The radii of a circle satisfying the given conditions -/
def circle_radii (c : TangentCircle) : Set ℝ :=
  {r : ℝ | r = 1 ∨ r = 7/3}

/-- Theorem stating that a circle satisfying the given conditions has radius 1 or 7/3 -/
theorem tangent_circle_radius 
  (c : TangentCircle) 
  (h1 : c.slope = Real.sqrt 3 / 3) 
  (h2 : c.point = (2, Real.sqrt 3)) : 
  ∀ r ∈ circle_radii c, r = 1 ∨ r = 7/3 := by
  sorry

end tangent_circle_radius_l970_97070


namespace divisibility_condition_l970_97040

theorem divisibility_condition (n : ℕ) : 
  (2^n + n) ∣ (8^n + n) ↔ n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 6 :=
sorry

end divisibility_condition_l970_97040


namespace initial_bacteria_count_l970_97016

def tripling_time : ℕ := 30  -- seconds
def total_time : ℕ := 300    -- seconds (5 minutes)
def final_count : ℕ := 1239220
def halfway_time : ℕ := 150  -- seconds (2.5 minutes)

def tripling_events (t : ℕ) : ℕ := t / tripling_time

theorem initial_bacteria_count :
  ∃ (n : ℕ),
    n * (3 ^ (tripling_events total_time)) / 2 = final_count ∧
    (n * (3 ^ (tripling_events halfway_time))) / 2 * (3 ^ (tripling_events halfway_time)) = final_count ∧
    n = 42 :=
by sorry

end initial_bacteria_count_l970_97016


namespace oliver_socks_l970_97004

/-- The number of socks Oliver initially had -/
def initial_socks : ℕ := 11

/-- The number of socks Oliver threw away -/
def thrown_away_socks : ℕ := 4

/-- The number of new socks Oliver bought -/
def new_socks : ℕ := 26

/-- The number of socks Oliver has now -/
def current_socks : ℕ := 33

theorem oliver_socks : 
  initial_socks - thrown_away_socks + new_socks = current_socks := by
  sorry


end oliver_socks_l970_97004


namespace sum_of_three_consecutive_even_numbers_l970_97061

theorem sum_of_three_consecutive_even_numbers (m : ℤ) : 
  m % 2 = 0 → (m + (m + 2) + (m + 4)) = 3 * m + 6 := by
sorry

end sum_of_three_consecutive_even_numbers_l970_97061


namespace trajectory_theorem_l970_97076

/-- The trajectory of point M -/
def trajectory_M (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 4 * ((x - 4)^2 + y^2)

/-- The trajectory of point P -/
def trajectory_P (x y : ℝ) : Prop :=
  (x - 1/2)^2 + y^2 = 1

/-- The main theorem -/
theorem trajectory_theorem :
  (∀ x y : ℝ, trajectory_M x y ↔ x^2 + y^2 = 4) ∧
  (∀ x y : ℝ, (∃ a b : ℝ, trajectory_M a b ∧ x = (a + 1) / 2 ∧ y = b / 2) → trajectory_P x y) :=
sorry

end trajectory_theorem_l970_97076


namespace jose_weekly_earnings_l970_97083

/-- Calculates Jose's weekly earnings from his swimming pool. -/
theorem jose_weekly_earnings :
  let kid_price : ℕ := 3
  let adult_price : ℕ := 2 * kid_price
  let kids_per_day : ℕ := 8
  let adults_per_day : ℕ := 10
  let days_per_week : ℕ := 7
  
  (kid_price * kids_per_day + adult_price * adults_per_day) * days_per_week = 588 :=
by sorry

end jose_weekly_earnings_l970_97083


namespace carrot_weight_problem_l970_97065

/-- Prove that given 20 carrots weighing 3.64 kg in total, and 4 carrots with an average weight of 190 grams are removed, the average weight of the remaining 16 carrots is 180 grams. -/
theorem carrot_weight_problem (total_weight : ℝ) (removed_avg : ℝ) :
  total_weight = 3.64 →
  removed_avg = 190 →
  (total_weight * 1000 - 4 * removed_avg) / 16 = 180 := by
sorry

end carrot_weight_problem_l970_97065


namespace quadratic_root_relation_l970_97077

theorem quadratic_root_relation (a b : ℝ) : 
  (3 : ℝ)^2 + 2*a*3 + 3*b = 0 → 2*a + b = -3 := by
  sorry

end quadratic_root_relation_l970_97077


namespace problem_solution_l970_97017

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem problem_solution :
  arithmetic_sequence 2 5 150 = 747 := by
  sorry

end problem_solution_l970_97017


namespace perfect_cube_units_digits_l970_97011

theorem perfect_cube_units_digits : 
  ∃! (S : Finset ℕ), 
    (∀ n : ℕ, n ∈ S ↔ ∃ m : ℕ, m^3 % 10 = n) ∧ 
    S.card = 10 :=
sorry

end perfect_cube_units_digits_l970_97011


namespace expected_participants_2008_l970_97021

/-- The number of participants in the school festival after n years, given an initial number of participants and an annual increase rate. -/
def participants_after_n_years (initial : ℝ) (rate : ℝ) (n : ℕ) : ℝ :=
  initial * (1 + rate) ^ n

/-- The expected number of participants in 2008, given the initial number in 2005 and the annual increase rate. -/
theorem expected_participants_2008 :
  participants_after_n_years 1000 0.25 3 = 1953.125 := by
  sorry

#eval participants_after_n_years 1000 0.25 3

end expected_participants_2008_l970_97021


namespace evaluate_expression_l970_97094

theorem evaluate_expression : -(18 / 3 * 8 - 40 + 5^2) = -33 := by
  sorry

end evaluate_expression_l970_97094


namespace minAbsNegativeFractions_minAbsNegativeTwo_solveMinAbsEquation_l970_97089

-- Define the min|a,b| operation for rational numbers
def minAbs (a b : ℚ) : ℚ := min a b

-- Theorem 1
theorem minAbsNegativeFractions : minAbs (-5/2) (-4/3) = -5/2 := by sorry

-- Theorem 2
theorem minAbsNegativeTwo (y : ℚ) (h : y < -2) : minAbs (-2) y = y := by sorry

-- Theorem 3
theorem solveMinAbsEquation : 
  ∃ x : ℚ, (minAbs (-x) 0 = -5 + 2*x) ∧ (x = 5/3) := by sorry

end minAbsNegativeFractions_minAbsNegativeTwo_solveMinAbsEquation_l970_97089


namespace dad_steps_count_l970_97015

theorem dad_steps_count (dad_masha_ratio : ℕ → ℕ → Prop)
                        (masha_yasha_ratio : ℕ → ℕ → Prop)
                        (masha_yasha_total : ℕ) :
  dad_masha_ratio 3 5 →
  masha_yasha_ratio 3 5 →
  masha_yasha_total = 400 →
  ∃ (dad_steps : ℕ), dad_steps = 90 := by
  sorry

end dad_steps_count_l970_97015


namespace collinear_vectors_m_values_l970_97003

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def are_collinear (u v : V) : Prop := ∃ (k : ℝ), u = k • v

theorem collinear_vectors_m_values
  (a b : V)
  (h1 : ¬ are_collinear a b)
  (h2 : ∃ (k : ℝ), (m : ℝ) • a - 3 • b = k • (a + (2 - m) • b)) :
  m = -1 ∨ m = 3 :=
sorry

end collinear_vectors_m_values_l970_97003


namespace smallest_n_is_three_l970_97005

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define x and y
noncomputable def x : ℂ := (-1 + i * Real.sqrt 3) / 2
noncomputable def y : ℂ := (-1 - i * Real.sqrt 3) / 2

-- Define the property we want to prove
def is_smallest_n (n : ℕ) : Prop :=
  n > 0 ∧ x^n + y^n = 2 ∧ ∀ m : ℕ, 0 < m ∧ m < n → x^m + y^m ≠ 2

-- The theorem we want to prove
theorem smallest_n_is_three : is_smallest_n 3 := by sorry

end smallest_n_is_three_l970_97005


namespace apartment_cost_difference_l970_97013

def apartment_cost (rent : ℕ) (utilities : ℕ) (daily_miles : ℕ) : ℕ :=
  rent + utilities + (daily_miles * 58 * 20) / 100

theorem apartment_cost_difference : 
  apartment_cost 800 260 31 - apartment_cost 900 200 21 = 76 := by sorry

end apartment_cost_difference_l970_97013


namespace roller_derby_teams_l970_97081

/-- The number of teams competing in a roller derby --/
def number_of_teams (members_per_team : ℕ) (skates_per_member : ℕ) (laces_per_skate : ℕ) (total_laces : ℕ) : ℕ :=
  total_laces / (members_per_team * skates_per_member * laces_per_skate)

/-- Theorem stating that the number of teams competing is 4 --/
theorem roller_derby_teams : number_of_teams 10 2 3 240 = 4 := by
  sorry

end roller_derby_teams_l970_97081


namespace min_value_fraction_l970_97043

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 2*y^2 + z^2) / (x*y + 3*y*z) ≥ 2*Real.sqrt 5 / 5 := by
  sorry

end min_value_fraction_l970_97043


namespace rational_roots_of_p_l970_97009

def p (x : ℚ) : ℚ := x^4 - 3*x^3 - 8*x^2 + 12*x + 16

theorem rational_roots_of_p :
  {x : ℚ | p x = 0} = {-1, -2, 2, 4} := by sorry

end rational_roots_of_p_l970_97009


namespace cos_equality_proof_l970_97066

theorem cos_equality_proof (n : ℤ) : 
  n = 43 ∧ -180 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (317 * π / 180) := by
  sorry

end cos_equality_proof_l970_97066


namespace circle_plus_solution_l970_97023

def circle_plus (a b : ℝ) : ℝ := a * b - 2 * b + 3 * a

theorem circle_plus_solution :
  ∃ x : ℝ, circle_plus 7 x = 61 ∧ x = 8 := by
  sorry

end circle_plus_solution_l970_97023


namespace integer_coloring_theorem_l970_97012

/-- A color type with four colors -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- A coloring function that assigns a color to each integer -/
def coloring : ℤ → Color := sorry

theorem integer_coloring_theorem 
  (m n : ℤ) 
  (h_odd_m : Odd m) 
  (h_odd_n : Odd n) 
  (h_distinct : m ≠ n) 
  (h_sum_nonzero : m + n ≠ 0) :
  ∃ (a b : ℤ), 
    coloring a = coloring b ∧ 
    (a - b = m ∨ a - b = n ∨ a - b = m + n ∨ a - b = m - n) := by
  sorry

end integer_coloring_theorem_l970_97012


namespace complex_equation_solution_l970_97071

theorem complex_equation_solution (z : ℂ) : (z - Complex.I) * (2 - Complex.I) = 5 → z = 2 + 2 * Complex.I := by
  sorry

end complex_equation_solution_l970_97071


namespace surface_area_of_sliced_solid_l970_97056

/-- Represents a right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- Represents the sliced-off solid CPQR -/
structure SlicedSolid where
  prism : RightPrism

/-- Calculates the surface area of the sliced-off solid CPQR -/
def surface_area (solid : SlicedSolid) : ℝ :=
  sorry

/-- Theorem stating the surface area of the sliced-off solid CPQR -/
theorem surface_area_of_sliced_solid (solid : SlicedSolid) 
  (h1 : solid.prism.height = 18)
  (h2 : solid.prism.base_side = 14) :
  surface_area solid = 63 + (49 * Real.sqrt 3 + Real.sqrt 521) / 4 :=
sorry

end surface_area_of_sliced_solid_l970_97056


namespace monotonic_f_implies_a_in_range_l970_97091

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) * (x - a)

-- State the theorem
theorem monotonic_f_implies_a_in_range (a : ℝ) :
  (∀ x y, x ≤ y ∧ y ≤ -2 → f a x ≤ f a y) ∧
  (∀ x y, 2 ≤ x ∧ x ≤ y → f a x ≤ f a y) →
  -2 ≤ a ∧ a ≤ 2 :=
by sorry

end monotonic_f_implies_a_in_range_l970_97091


namespace cinema_seating_l970_97036

/-- The number of chairs occupied in a cinema row --/
def occupied_chairs (chairs_between : ℕ) : ℕ :=
  chairs_between + 2

theorem cinema_seating (chairs_between : ℕ) 
  (h : chairs_between = 30) : occupied_chairs chairs_between = 32 := by
  sorry

end cinema_seating_l970_97036


namespace broadway_ticket_sales_l970_97075

theorem broadway_ticket_sales
  (num_adults : ℕ)
  (num_children : ℕ)
  (adult_ticket_price : ℝ)
  (h1 : num_adults = 400)
  (h2 : num_children = 200)
  (h3 : adult_ticket_price = 32)
  (h4 : adult_ticket_price = 2 * (adult_ticket_price / 2)) :
  num_adults * adult_ticket_price + num_children * (adult_ticket_price / 2) = 16000 := by
sorry

end broadway_ticket_sales_l970_97075


namespace satisfying_function_characterization_l970_97086

/-- A function from positive reals to reals satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 →
    (f x + f y ≤ f (x + y) / 2) ∧
    (f x / x + f y / y ≥ f (x + y) / (x + y))

/-- The theorem stating that any satisfying function must be of the form f(x) = ax² where a ≤ 0 -/
theorem satisfying_function_characterization (f : ℝ → ℝ) :
  SatisfyingFunction f →
  ∃ a : ℝ, a ≤ 0 ∧ ∀ x : ℝ, x > 0 → f x = a * x^2 :=
sorry

end satisfying_function_characterization_l970_97086


namespace people_born_in_country_l970_97082

/-- The number of people who immigrated to the country last year -/
def immigrants : ℕ := 16320

/-- The total number of new people who began living in the country last year -/
def new_residents : ℕ := 106491

/-- The number of people born in the country last year -/
def births : ℕ := new_residents - immigrants

theorem people_born_in_country : births = 90171 := by
  sorry

end people_born_in_country_l970_97082


namespace limit_f_at_zero_l970_97060

open Real
open Filter
open Topology

noncomputable def f (x : ℝ) : ℝ := Real.log ((Real.exp (x^2) - Real.cos x) * Real.cos (1/x) + Real.tan (x + π/3))

theorem limit_f_at_zero : 
  Tendsto f (𝓝 0) (𝓝 ((1/2) * Real.log 3)) := by sorry

end limit_f_at_zero_l970_97060


namespace tree_planting_l970_97062

theorem tree_planting (road_length : ℕ) (tree_spacing : ℕ) (h1 : road_length = 42) (h2 : tree_spacing = 7) : 
  road_length / tree_spacing + 1 = 7 := by
  sorry

end tree_planting_l970_97062


namespace flour_for_dozen_cookies_l970_97046

/-- Given information about cookie production and consumption, calculate the amount of flour needed for a dozen cookies -/
theorem flour_for_dozen_cookies 
  (bags : ℕ) 
  (weight_per_bag : ℕ) 
  (cookies_eaten : ℕ) 
  (cookies_left : ℕ) 
  (h1 : bags = 4) 
  (h2 : weight_per_bag = 5) 
  (h3 : cookies_eaten = 15) 
  (h4 : cookies_left = 105) : 
  (12 : ℝ) * (bags * weight_per_bag : ℝ) / ((cookies_left + cookies_eaten) : ℝ) = 2 := by
  sorry

end flour_for_dozen_cookies_l970_97046


namespace shopkeeper_percentage_gain_l970_97039

/-- The percentage gain of a shopkeeper using a false weight --/
theorem shopkeeper_percentage_gain :
  let actual_weight : ℝ := 970
  let claimed_weight : ℝ := 1000
  let gain : ℝ := claimed_weight - actual_weight
  let percentage_gain : ℝ := (gain / actual_weight) * 100
  ∃ ε > 0, abs (percentage_gain - 3.09) < ε :=
by sorry

end shopkeeper_percentage_gain_l970_97039


namespace fixed_points_of_f_composition_l970_97002

def f (x : ℝ) : ℝ := x^2 - 4*x

theorem fixed_points_of_f_composition (x : ℝ) : 
  f (f x) = f x ↔ x ∈ ({-1, 0, 4, 5} : Set ℝ) := by
  sorry

end fixed_points_of_f_composition_l970_97002


namespace smallest_prime_above_50_l970_97032

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem smallest_prime_above_50 :
  ∃ p : ℕ, is_prime p ∧ p > 50 ∧ ∀ q : ℕ, is_prime q ∧ q > 50 → p ≤ q :=
by sorry

end smallest_prime_above_50_l970_97032


namespace cyclic_trio_exists_l970_97045

/-- Represents the result of a match between two players -/
inductive MatchResult
| Win
| Loss

/-- A tournament with a fixed number of players -/
structure Tournament where
  numPlayers : Nat
  results : Fin numPlayers → Fin numPlayers → MatchResult

/-- Predicate to check if player i defeated player j -/
def defeated (t : Tournament) (i j : Fin t.numPlayers) : Prop :=
  t.results i j = MatchResult.Win

theorem cyclic_trio_exists (t : Tournament) 
  (h1 : t.numPlayers = 12)
  (h2 : ∀ i j : Fin t.numPlayers, i ≠ j → (defeated t i j ∨ defeated t j i))
  (h3 : ∀ i : Fin t.numPlayers, ∃ j : Fin t.numPlayers, defeated t i j) :
  ∃ a b c : Fin t.numPlayers, 
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    defeated t a b ∧ defeated t b c ∧ defeated t c a :=
sorry

end cyclic_trio_exists_l970_97045


namespace class_average_score_l970_97053

theorem class_average_score (total_students : ℕ) 
  (score_95_count score_0_count score_65_count score_80_count : ℕ)
  (remaining_avg : ℚ) :
  total_students = 40 →
  score_95_count = 5 →
  score_0_count = 3 →
  score_65_count = 6 →
  score_80_count = 8 →
  remaining_avg = 45 →
  (2000 : ℚ) ≤ (score_95_count * 95 + score_0_count * 0 + score_65_count * 65 + 
    score_80_count * 80 + (total_students - score_95_count - score_0_count - 
    score_65_count - score_80_count) * remaining_avg) →
  (score_95_count * 95 + score_0_count * 0 + score_65_count * 65 + 
    score_80_count * 80 + (total_students - score_95_count - score_0_count - 
    score_65_count - score_80_count) * remaining_avg) ≤ (2400 : ℚ) →
  (score_95_count * 95 + score_0_count * 0 + score_65_count * 65 + 
    score_80_count * 80 + (total_students - score_95_count - score_0_count - 
    score_65_count - score_80_count) * remaining_avg) / total_students = (57875 : ℚ) / 1000 :=
by sorry

end class_average_score_l970_97053


namespace xy_value_l970_97006

theorem xy_value (x y : ℝ) (h : Real.sqrt (2 * x - 4) + |y - 1| = 0) : x * y = 2 := by
  sorry

end xy_value_l970_97006

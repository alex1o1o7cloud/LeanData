import Mathlib

namespace NUMINAMATH_CALUDE_coin_order_correct_l1222_122216

-- Define the type for coins
inductive Coin : Type
  | A | B | C | D | E | F

-- Define a relation for one coin being above another
def IsAbove : Coin → Coin → Prop := sorry

-- Define the correct order of coins
def CorrectOrder : List Coin := [Coin.F, Coin.C, Coin.E, Coin.D, Coin.A, Coin.B]

-- State the theorem
theorem coin_order_correct (coins : List Coin) 
  (h1 : IsAbove Coin.F Coin.C)
  (h2 : IsAbove Coin.F Coin.E)
  (h3 : IsAbove Coin.F Coin.A)
  (h4 : IsAbove Coin.F Coin.D)
  (h5 : IsAbove Coin.F Coin.B)
  (h6 : IsAbove Coin.C Coin.A)
  (h7 : IsAbove Coin.C Coin.D)
  (h8 : IsAbove Coin.C Coin.B)
  (h9 : IsAbove Coin.E Coin.A)
  (h10 : IsAbove Coin.E Coin.B)
  (h11 : IsAbove Coin.D Coin.B)
  (h12 : coins.length = 6)
  (h13 : coins.Nodup)
  (h14 : ∀ c, c ∈ coins ↔ c ∈ [Coin.A, Coin.B, Coin.C, Coin.D, Coin.E, Coin.F]) :
  coins = CorrectOrder := by sorry


end NUMINAMATH_CALUDE_coin_order_correct_l1222_122216


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l1222_122245

theorem quadratic_root_sum (m n : ℝ) : 
  (∀ x, m * x^2 - n * x - 2023 = 0 → x = -1 ∨ x ≠ -1) →
  (m * (-1)^2 - n * (-1) - 2023 = 0) →
  m + n = 2023 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l1222_122245


namespace NUMINAMATH_CALUDE_cades_marbles_l1222_122243

theorem cades_marbles (initial_marbles : ℕ) (marbles_given : ℕ) : 
  initial_marbles = 87 → marbles_given = 8 → initial_marbles - marbles_given = 79 := by
  sorry

end NUMINAMATH_CALUDE_cades_marbles_l1222_122243


namespace NUMINAMATH_CALUDE_rectangular_plot_poles_l1222_122225

/-- The number of poles needed to enclose a rectangular plot -/
def num_poles (length width pole_distance : ℕ) : ℕ :=
  2 * (length + width) / pole_distance + 4

theorem rectangular_plot_poles :
  num_poles 90 50 4 = 74 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_poles_l1222_122225


namespace NUMINAMATH_CALUDE_probability_of_car_Z_winning_l1222_122224

/-- Given a race with 15 cars, prove that the probability of car Z winning is 1/12 -/
theorem probability_of_car_Z_winning (total_cars : ℕ) (prob_X prob_Y prob_XYZ : ℚ) :
  total_cars = 15 →
  prob_X = 1/4 →
  prob_Y = 1/8 →
  prob_XYZ = 458333333333333333/1000000000000000000 →
  prob_XYZ = prob_X + prob_Y + (1/12) :=
by sorry

end NUMINAMATH_CALUDE_probability_of_car_Z_winning_l1222_122224


namespace NUMINAMATH_CALUDE_adam_ella_equation_l1222_122289

theorem adam_ella_equation (d e : ℝ) : 
  (∀ x, |x - 8| = 3 ↔ x^2 + d*x + e = 0) → 
  d = -16 ∧ e = 55 := by
sorry

end NUMINAMATH_CALUDE_adam_ella_equation_l1222_122289


namespace NUMINAMATH_CALUDE_champion_is_team_d_l1222_122268

-- Define the teams
inductive Team : Type
| A | B | C | D

-- Define the rankings
structure Ranking :=
(first : Team)
(second : Team)
(third : Team)
(fourth : Team)

-- Define the predictions
structure Prediction :=
(first : Option Team)
(second : Option Team)
(third : Option Team)
(fourth : Option Team)

-- Define the function to check if a prediction is half correct
def isHalfCorrect (pred : Prediction) (actual : Ranking) : Prop :=
  (pred.first = some actual.first ∨ pred.second = some actual.second ∨ 
   pred.third = some actual.third ∨ pred.fourth = some actual.fourth) ∧
  (pred.first ≠ some actual.first ∨ pred.second ≠ some actual.second ∨
   pred.third ≠ some actual.third ∨ pred.fourth ≠ some actual.fourth)

-- Theorem statement
theorem champion_is_team_d :
  ∀ (actual : Ranking),
    let wang_pred : Prediction := ⟨some Team.D, some Team.B, none, none⟩
    let li_pred : Prediction := ⟨none, some Team.A, none, some Team.C⟩
    let zhang_pred : Prediction := ⟨none, some Team.D, some Team.C, none⟩
    isHalfCorrect wang_pred actual ∧
    isHalfCorrect li_pred actual ∧
    isHalfCorrect zhang_pred actual →
    actual.first = Team.D :=
by sorry


end NUMINAMATH_CALUDE_champion_is_team_d_l1222_122268


namespace NUMINAMATH_CALUDE_ball_probability_comparison_l1222_122200

theorem ball_probability_comparison :
  let total_balls : ℕ := 3
  let red_balls : ℕ := 2
  let white_balls : ℕ := 1
  let p_red : ℚ := red_balls / total_balls
  let p_white : ℚ := white_balls / total_balls
  p_red > p_white :=
by
  sorry

end NUMINAMATH_CALUDE_ball_probability_comparison_l1222_122200


namespace NUMINAMATH_CALUDE_problem_solution_l1222_122214

theorem problem_solution (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a * b = 2) 
  (h2 : a / (a + b^2) + b / (b + a^2) = 7/8) : 
  a^6 + b^6 = 128 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1222_122214


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_160_l1222_122226

/-- The sum of the digits in the binary representation of 160 is 2. -/
theorem sum_of_binary_digits_160 : 
  (Nat.digits 2 160).sum = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_160_l1222_122226


namespace NUMINAMATH_CALUDE_sixth_finger_is_one_l1222_122274

def f : ℕ → ℕ
| 2 => 1
| 1 => 8
| 8 => 7
| 7 => 2
| _ => 0  -- Default case for other inputs

def finger_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 2  -- Start with 2 on the first finger (index 0)
  | n + 1 => f (finger_sequence n)

theorem sixth_finger_is_one : finger_sequence 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sixth_finger_is_one_l1222_122274


namespace NUMINAMATH_CALUDE_shoes_sold_l1222_122211

theorem shoes_sold (large medium small left : ℕ) 
  (h1 : large = 22)
  (h2 : medium = 50)
  (h3 : small = 24)
  (h4 : left = 13) : 
  large + medium + small - left = 83 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sold_l1222_122211


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l1222_122246

/-- Represents a bag of marbles with two colors -/
structure Bag where
  color1 : ℕ
  color2 : ℕ

/-- Calculate the probability of drawing a specific color from a bag -/
def probColor (bag : Bag) (color : ℕ) : ℚ :=
  color / (bag.color1 + bag.color2)

/-- The probability of drawing a yellow marble as the second marble -/
def probYellowSecond (bagX bagY bagZ : Bag) : ℚ :=
  probColor bagX bagX.color1 * probColor bagY bagY.color1 +
  probColor bagX bagX.color2 * probColor bagZ bagZ.color1

theorem yellow_marble_probability :
  let bagX : Bag := ⟨4, 5⟩  -- 4 white, 5 black
  let bagY : Bag := ⟨7, 3⟩  -- 7 yellow, 3 blue
  let bagZ : Bag := ⟨3, 6⟩  -- 3 yellow, 6 blue
  probYellowSecond bagX bagY bagZ = 67 / 135 := by
  sorry


end NUMINAMATH_CALUDE_yellow_marble_probability_l1222_122246


namespace NUMINAMATH_CALUDE_origin_inside_ellipse_k_range_l1222_122259

/-- The ellipse equation -/
def ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0

/-- A point (x,y) is inside the ellipse if the left side of the equation is negative -/
def inside_ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 < 0

/-- The theorem stating the range of k for which the origin (0,0) is inside the ellipse -/
theorem origin_inside_ellipse_k_range :
  ∀ k : ℝ, (inside_ellipse k 0 0) ↔ (0 < |k| ∧ |k| < 1) :=
by sorry

end NUMINAMATH_CALUDE_origin_inside_ellipse_k_range_l1222_122259


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1222_122294

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := -8/21
  let a₃ : ℚ := 16/63
  let r : ℚ := a₂ / a₁
  (r = -2/3) ∧ (a₃ / a₂ = r) := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1222_122294


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l1222_122291

/-- Represents a sequence of 5 integers -/
def Sequence := Fin 5 → ℕ

/-- Checks if a sequence is valid for systematic sampling -/
def isValidSample (s : Sequence) (totalBags : ℕ) (sampleSize : ℕ) : Prop :=
  ∃ (start : ℕ) (interval : ℕ),
    (∀ i : Fin 5, s i = start + i.val * interval) ∧
    (∀ i : Fin 5, 1 ≤ s i ∧ s i ≤ totalBags) ∧
    interval = totalBags / sampleSize

theorem systematic_sampling_proof :
  let s : Sequence := fun i => [7, 17, 27, 37, 47][i]
  let totalBags := 50
  let sampleSize := 5
  isValidSample s totalBags sampleSize :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l1222_122291


namespace NUMINAMATH_CALUDE_complex_fraction_equals_negative_i_l1222_122231

theorem complex_fraction_equals_negative_i :
  ∀ (i : ℂ), i * i = -1 →
  (1 - i) / (1 + i) = -i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_negative_i_l1222_122231


namespace NUMINAMATH_CALUDE_least_multiple_of_15_greater_than_500_l1222_122256

theorem least_multiple_of_15_greater_than_500 : 
  (∃ (n : ℕ), 15 * n = 510 ∧ 
   510 > 500 ∧ 
   ∀ (m : ℕ), 15 * m > 500 → 15 * m ≥ 510) := by
sorry

end NUMINAMATH_CALUDE_least_multiple_of_15_greater_than_500_l1222_122256


namespace NUMINAMATH_CALUDE_fermat_like_prime_condition_l1222_122273

theorem fermat_like_prime_condition (a n : ℕ) (ha : a ≥ 2) (hn : n ≥ 2) 
  (h_prime : Nat.Prime (a^n - 1)) : a = 2 ∧ Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_fermat_like_prime_condition_l1222_122273


namespace NUMINAMATH_CALUDE_power_mod_twenty_l1222_122234

theorem power_mod_twenty : 17^2037 % 20 = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_twenty_l1222_122234


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1222_122206

theorem inequality_solution_set : 
  {x : ℝ | (3 : ℝ) / (5 - 3 * x) > 1} = {x : ℝ | 2/3 < x ∧ x < 5/3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1222_122206


namespace NUMINAMATH_CALUDE_sisters_gift_l1222_122229

def trip_cost : ℕ := 5000
def hourly_wage : ℕ := 20
def hours_worked : ℕ := 10
def cookie_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_ticket_cost : ℕ := 10
def lottery_winnings : ℕ := 500
def additional_money_needed : ℕ := 3214

def job_earnings : ℕ := hourly_wage * hours_worked
def cookie_earnings : ℕ := cookie_price * cookies_sold
def total_earnings : ℕ := job_earnings + cookie_earnings - lottery_ticket_cost + lottery_winnings

theorem sisters_gift (sisters_gift : ℕ) : sisters_gift = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_sisters_gift_l1222_122229


namespace NUMINAMATH_CALUDE_seven_point_four_five_repeating_equals_82_11_l1222_122202

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def repeatingDecimalToRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + (x.repeatingPart : ℚ) / (99 : ℚ)

/-- The repeating decimal 7.45̄ -/
def seven_point_four_five_repeating : RepeatingDecimal :=
  { integerPart := 7, repeatingPart := 45 }

theorem seven_point_four_five_repeating_equals_82_11 :
  repeatingDecimalToRational seven_point_four_five_repeating = 82 / 11 := by
  sorry

end NUMINAMATH_CALUDE_seven_point_four_five_repeating_equals_82_11_l1222_122202


namespace NUMINAMATH_CALUDE_betty_bead_ratio_l1222_122237

/-- Given that Betty has 30 red beads and 20 blue beads, prove that the ratio of red beads to blue beads is 3:2 -/
theorem betty_bead_ratio :
  let red_beads : ℕ := 30
  let blue_beads : ℕ := 20
  (red_beads : ℚ) / blue_beads = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_betty_bead_ratio_l1222_122237


namespace NUMINAMATH_CALUDE_lunch_price_with_gratuity_l1222_122209

theorem lunch_price_with_gratuity 
  (num_people : ℕ) 
  (avg_price : ℝ) 
  (gratuity_rate : ℝ) : 
  num_people = 15 →
  avg_price = 12 →
  gratuity_rate = 0.15 →
  num_people * avg_price * (1 + gratuity_rate) = 207 := by
  sorry

end NUMINAMATH_CALUDE_lunch_price_with_gratuity_l1222_122209


namespace NUMINAMATH_CALUDE_oil_depth_theorem_l1222_122261

/-- Represents a horizontal cylindrical tank with oil -/
structure OilTank where
  length : ℝ
  diameter : ℝ
  oil_surface_area : ℝ

/-- Calculates the possible depths of oil in the tank -/
def oil_depths (tank : OilTank) : Set ℝ :=
  { h | h = 3 - Real.sqrt 5 ∨ h = 3 + Real.sqrt 5 }

theorem oil_depth_theorem (tank : OilTank) 
  (h_length : tank.length = 10)
  (h_diameter : tank.diameter = 6)
  (h_area : tank.oil_surface_area = 40) :
  ∀ h ∈ oil_depths tank, 
    ∃ c : ℝ, 
      c = tank.oil_surface_area / tank.length ∧ 
      c ^ 2 = 2 * (tank.diameter / 2) * h - h ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_oil_depth_theorem_l1222_122261


namespace NUMINAMATH_CALUDE_equation_solution_l1222_122204

theorem equation_solution : ∃ x : ℝ, 35 - (5 + 3) = 7 + x ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1222_122204


namespace NUMINAMATH_CALUDE_talia_total_distance_l1222_122218

/-- Represents the total distance Talia drives in a day -/
def total_distance (home_to_park park_to_grocery grocery_to_friend friend_to_home : ℕ) : ℕ :=
  home_to_park + park_to_grocery + grocery_to_friend + friend_to_home

/-- Theorem stating that Talia drives 18 miles in total -/
theorem talia_total_distance :
  ∃ (home_to_park park_to_grocery grocery_to_friend friend_to_home : ℕ),
    home_to_park = 5 ∧
    park_to_grocery = 3 ∧
    grocery_to_friend = 6 ∧
    friend_to_home = 4 ∧
    total_distance home_to_park park_to_grocery grocery_to_friend friend_to_home = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_talia_total_distance_l1222_122218


namespace NUMINAMATH_CALUDE_min_product_abc_l1222_122265

theorem min_product_abc (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b + c = 3)
  (h5 : a ≤ 3 * b ∧ a ≤ 3 * c)
  (h6 : b ≤ 3 * a ∧ b ≤ 3 * c)
  (h7 : c ≤ 3 * a ∧ c ≤ 3 * b) :
  81 / 125 ≤ a * b * c :=
by sorry

end NUMINAMATH_CALUDE_min_product_abc_l1222_122265


namespace NUMINAMATH_CALUDE_simplify_expression_l1222_122286

theorem simplify_expression (x : ℝ) : (3*x)^5 - (4*x)*(x^4) = 239*(x^5) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1222_122286


namespace NUMINAMATH_CALUDE_motorcyclist_problem_l1222_122239

/-- The time taken by the first motorcyclist to travel the distance AB -/
def time_first : ℝ := 80

/-- The time taken by the second motorcyclist to travel the distance AB -/
def time_second : ℝ := 60

/-- The time taken by the third motorcyclist to travel the distance AB -/
def time_third : ℝ := 3240

/-- The head start of the first motorcyclist -/
def head_start : ℝ := 5

/-- The time difference between the third and second motorcyclist overtaking the first -/
def overtake_diff : ℝ := 10

/-- The distance between points A and B -/
def distance : ℝ := 1  -- We can set this to any positive real number

theorem motorcyclist_problem :
  ∃ (speed_first speed_second speed_third : ℝ),
    speed_first > 0 ∧ speed_second > 0 ∧ speed_third > 0 ∧
    speed_first ≠ speed_second ∧ speed_first ≠ speed_third ∧ speed_second ≠ speed_third ∧
    speed_first = distance / time_first ∧
    speed_second = distance / time_second ∧
    speed_third = distance / time_third ∧
    (time_third - head_start) * speed_third = time_first * speed_first ∧
    (time_second - head_start) * speed_second = (time_first + overtake_diff) * speed_first :=
by sorry

end NUMINAMATH_CALUDE_motorcyclist_problem_l1222_122239


namespace NUMINAMATH_CALUDE_union_of_sets_l1222_122281

theorem union_of_sets : 
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | 0 < x ∧ x < 2}
  A ∪ B = {x : ℝ | -2 < x ∧ x < 2} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1222_122281


namespace NUMINAMATH_CALUDE_egyptian_fraction_equation_solutions_l1222_122292

theorem egyptian_fraction_equation_solutions :
  ∀ x y z : ℕ+,
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = 4 / 5 →
  ((x = 2 ∧ y = 4 ∧ z = 20) ∨ (x = 2 ∧ y = 5 ∧ z = 10)) :=
by sorry

end NUMINAMATH_CALUDE_egyptian_fraction_equation_solutions_l1222_122292


namespace NUMINAMATH_CALUDE_altitude_sum_of_triangle_l1222_122208

/-- The line equation --/
def line_equation (x y : ℝ) : Prop := 15 * x + 6 * y = 90

/-- A point is on the x-axis if its y-coordinate is 0 --/
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- A point is on the y-axis if its x-coordinate is 0 --/
def on_y_axis (p : ℝ × ℝ) : Prop := p.1 = 0

/-- The triangle vertices --/
def triangle_vertices : Set (ℝ × ℝ) := {(0, 0), (6, 0), (0, 15)}

/-- The sum of altitudes of the triangle --/
noncomputable def altitude_sum : ℝ := 21 + 10 * Real.sqrt (1 / 29)

/-- The main theorem --/
theorem altitude_sum_of_triangle :
  ∀ (p : ℝ × ℝ), p ∈ triangle_vertices →
  (on_x_axis p ∨ on_y_axis p ∨ line_equation p.1 p.2) →
  altitude_sum = 21 + 10 * Real.sqrt (1 / 29) :=
sorry

end NUMINAMATH_CALUDE_altitude_sum_of_triangle_l1222_122208


namespace NUMINAMATH_CALUDE_find_b_find_a_range_l1222_122248

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - b * x

-- Define the derivative of f
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := a / x + (1 - a) * x - b

-- Theorem 1: Find the value of b
theorem find_b (a : ℝ) (h : a ≠ 1) :
  (∃ b : ℝ, f_deriv a b 1 = 0) → (∃ b : ℝ, b = 1) :=
sorry

-- Theorem 2: Find the range of values for a
theorem find_a_range (a : ℝ) (h : a ≠ 1) :
  (∃ x : ℝ, x ≥ 1 ∧ f a 1 x < a / (a - 1)) →
  (a ∈ Set.Ioo (- Real.sqrt 2 - 1) (Real.sqrt 2 - 1) ∪ Set.Ioi 1) :=
sorry

end

end NUMINAMATH_CALUDE_find_b_find_a_range_l1222_122248


namespace NUMINAMATH_CALUDE_smallest_nonnegative_solution_congruence_l1222_122233

theorem smallest_nonnegative_solution_congruence :
  ∃ (x : ℕ), x < 15 ∧ (7 * x + 3) % 15 = 6 % 15 ∧
  ∀ (y : ℕ), y < x → (7 * y + 3) % 15 ≠ 6 % 15 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_nonnegative_solution_congruence_l1222_122233


namespace NUMINAMATH_CALUDE_parabola_focus_l1222_122217

/-- A parabola is defined by the equation y = x^2 -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- The focus of a parabola is a point with specific properties -/
def IsFocus (f : ℝ × ℝ) (p : Set (ℝ × ℝ)) : Prop :=
  ∃ (a : ℝ), p = {point : ℝ × ℝ | point.2 = point.1^2} ∧ f = (0, 1/(4*a))

/-- The theorem states that the focus of the parabola y = x^2 is at (0, 1/4) -/
theorem parabola_focus : IsFocus (0, 1/4) Parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l1222_122217


namespace NUMINAMATH_CALUDE_domain_of_f_l1222_122285

noncomputable def f (x : ℝ) := (2 * x - 3) / (x^2 - 5 * x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 2 ∨ (2 < x ∧ x < 3) ∨ 3 < x} := by
  sorry

end NUMINAMATH_CALUDE_domain_of_f_l1222_122285


namespace NUMINAMATH_CALUDE_remaining_terms_geometric_l1222_122207

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q^(n - 1)

theorem remaining_terms_geometric (a q : ℝ) (k : ℕ) :
  let original_seq := geometric_sequence a q
  let remaining_seq := fun n => original_seq (n + k)
  ∃ a', remaining_seq = geometric_sequence a' q :=
sorry

end NUMINAMATH_CALUDE_remaining_terms_geometric_l1222_122207


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1222_122257

/-- 
Given a quadratic function y = 3x^2 + px + q, 
prove that the value of q that makes the minimum value of y equal to 1 is 1 + p^2/18
-/
theorem quadratic_minimum (p : ℝ) : 
  ∃ (q : ℝ), (∀ (x : ℝ), 3 * x^2 + p * x + q ≥ 1) ∧ 
  (∃ (x : ℝ), 3 * x^2 + p * x + q = 1) → 
  q = 1 + p^2 / 18 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1222_122257


namespace NUMINAMATH_CALUDE_count_lambs_l1222_122280

def farmer_cunningham_lambs : Nat → Nat → Prop
  | white_lambs, black_lambs =>
    ∀ (total_lambs : Nat),
      (white_lambs = 193) →
      (black_lambs = 5855) →
      (total_lambs = white_lambs + black_lambs) →
      (total_lambs = 6048)

theorem count_lambs :
  farmer_cunningham_lambs 193 5855 := by
  sorry

end NUMINAMATH_CALUDE_count_lambs_l1222_122280


namespace NUMINAMATH_CALUDE_parallel_transitivity_perpendicular_plane_implies_parallel_l1222_122212

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelToPlane : Line → Plane → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)

-- Axiom for transitivity of parallel lines
axiom parallel_trans (a b c : Line) : parallel a b → parallel b c → parallel a c

-- Axiom for perpendicular lines to the same plane being parallel
axiom perpendicular_plane_parallel (a b : Line) (γ : Plane) : 
  perpendicularToPlane a γ → perpendicularToPlane b γ → parallel a b

-- Theorem 1: If two lines are parallel to a third line, then they are parallel to each other
theorem parallel_transitivity (a b c : Line) : 
  parallel a b → parallel b c → parallel a c :=
sorry

-- Theorem 2: Two lines perpendicular to the same plane are parallel to each other
theorem perpendicular_plane_implies_parallel (a b : Line) (γ : Plane) :
  perpendicularToPlane a γ → perpendicularToPlane b γ → parallel a b :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_perpendicular_plane_implies_parallel_l1222_122212


namespace NUMINAMATH_CALUDE_probability_at_least_one_boy_one_girl_l1222_122275

theorem probability_at_least_one_boy_one_girl (p : ℝ) : 
  p = 1/2 → (1 - 2 * p^4) = 7/8 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_boy_one_girl_l1222_122275


namespace NUMINAMATH_CALUDE_log_eight_x_three_halves_l1222_122298

theorem log_eight_x_three_halves (x : ℝ) :
  Real.log x / Real.log 8 = 3/2 → x = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_eight_x_three_halves_l1222_122298


namespace NUMINAMATH_CALUDE_money_ratio_problem_l1222_122271

/-- Proves that given the ratios between Ravi, Giri, and Kiran's money, and the fact that Ravi has $36, Kiran must have $105. -/
theorem money_ratio_problem (ravi giri kiran : ℕ) : 
  (ravi : ℚ) / giri = 6 / 7 → 
  (giri : ℚ) / kiran = 6 / 15 → 
  ravi = 36 → 
  kiran = 105 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_problem_l1222_122271


namespace NUMINAMATH_CALUDE_line_equation_from_intercept_and_angle_l1222_122262

/-- The equation of a line with given x-intercept and inclination angle -/
theorem line_equation_from_intercept_and_angle (x_intercept : ℝ) (angle : ℝ) :
  x_intercept = 2 ∧ angle = 135 * π / 180 →
  ∀ x y : ℝ, (x + y - 2 = 0) ↔ (y = (x - x_intercept) * Real.tan angle) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_intercept_and_angle_l1222_122262


namespace NUMINAMATH_CALUDE_inequality_proof_l1222_122251

theorem inequality_proof (x : ℝ) : 
  (4 * x^2 / (1 - Real.sqrt (1 + 2*x))^2 < 2*x + 9) → 
  (-1/2 ≤ x ∧ x < 45/8 ∧ x ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1222_122251


namespace NUMINAMATH_CALUDE_paint_area_is_134_l1222_122242

/-- The area to be painted on a wall with a window and door -/
def areaToPaint (wallHeight wallLength windowSide doorWidth doorHeight : ℝ) : ℝ :=
  wallHeight * wallLength - windowSide * windowSide - doorWidth * doorHeight

/-- Theorem: The area to be painted is 134 square feet -/
theorem paint_area_is_134 :
  areaToPaint 10 15 3 1 7 = 134 := by
  sorry

end NUMINAMATH_CALUDE_paint_area_is_134_l1222_122242


namespace NUMINAMATH_CALUDE_jony_turnaround_block_l1222_122266

/-- Represents the walking scenario of Jony along Sunrise Boulevard -/
structure WalkingScenario where
  start_block : ℕ
  end_block : ℕ
  block_length : ℕ
  walking_speed : ℕ
  walking_time : ℕ

/-- Calculates the block where Jony turns around -/
def turnaround_block (scenario : WalkingScenario) : ℕ :=
  let total_distance := scenario.walking_speed * scenario.walking_time
  let start_to_end_distance := (scenario.end_block - scenario.start_block) * scenario.block_length
  let extra_distance := total_distance - start_to_end_distance
  let extra_blocks := extra_distance / scenario.block_length
  scenario.end_block + extra_blocks

/-- Theorem stating that Jony turns around at block 110 -/
theorem jony_turnaround_block :
  let scenario : WalkingScenario := {
    start_block := 10,
    end_block := 70,
    block_length := 40,
    walking_speed := 100,
    walking_time := 40
  }
  turnaround_block scenario = 110 := by
  sorry

end NUMINAMATH_CALUDE_jony_turnaround_block_l1222_122266


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1222_122283

-- Define the solution set based on the value of a
def solutionSet (a : ℝ) : Set ℝ :=
  if a > 0 then {x | x < -a/4 ∨ x > a/3}
  else if a = 0 then {x | x ≠ 0}
  else {x | x > -a/4 ∨ x < a/3}

-- Theorem statement
theorem inequality_solution_set (a : ℝ) :
  {x : ℝ | 12 * x^2 - a * x > a^2} = solutionSet a := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1222_122283


namespace NUMINAMATH_CALUDE_twenty_four_game_solvable_l1222_122222

/-- Represents the basic arithmetic operations -/
inductive Operation
  | Add
  | Subtract
  | Multiply
  | Divide

/-- Represents an expression in the 24 Game -/
inductive Expr
  | Num (n : ℕ)
  | Op (op : Operation) (e1 e2 : Expr)

/-- Evaluates an expression -/
def eval : Expr → ℚ
  | Expr.Num n => n
  | Expr.Op Operation.Add e1 e2 => eval e1 + eval e2
  | Expr.Op Operation.Subtract e1 e2 => eval e1 - eval e2
  | Expr.Op Operation.Multiply e1 e2 => eval e1 * eval e2
  | Expr.Op Operation.Divide e1 e2 => eval e1 / eval e2

/-- Checks if an expression uses all given numbers exactly once -/
def usesAllNumbers (e : Expr) (numbers : List ℕ) : Prop := sorry

/-- The 24 Game theorem -/
theorem twenty_four_game_solvable (numbers : List ℕ := [2, 5, 11, 12]) :
  ∃ e : Expr, usesAllNumbers e numbers ∧ eval e = 24 := by sorry

end NUMINAMATH_CALUDE_twenty_four_game_solvable_l1222_122222


namespace NUMINAMATH_CALUDE_intersection_condition_subset_complement_condition_l1222_122240

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 2 ≤ x ∧ x ≤ m + 2}

theorem intersection_condition (m : ℝ) :
  A ∩ B m = {x : ℝ | 0 ≤ x ∧ x ≤ 3} → m = 2 := by sorry

theorem subset_complement_condition (m : ℝ) :
  A ⊆ (Set.univ \ B m) → m < -3 ∨ m > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_subset_complement_condition_l1222_122240


namespace NUMINAMATH_CALUDE_inequality_proof_l1222_122223

theorem inequality_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (x^6 / y^2) + (y^6 / x^2) ≥ x^4 + y^4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1222_122223


namespace NUMINAMATH_CALUDE_competition_result_l1222_122276

/-- Represents the scores of contestants in a mathematics competition. -/
structure Scores where
  ann : ℝ
  bill : ℝ
  carol : ℝ
  dick : ℝ
  nonnegative : 0 ≤ ann ∧ 0 ≤ bill ∧ 0 ≤ carol ∧ 0 ≤ dick

/-- Conditions of the mathematics competition. -/
def CompetitionConditions (s : Scores) : Prop :=
  s.bill + s.dick = 2 * s.ann ∧
  s.ann + s.carol < s.bill + s.dick ∧
  s.ann < s.bill + s.carol

/-- The order of contestants from highest to lowest score. -/
def CorrectOrder (s : Scores) : Prop :=
  s.dick > s.bill ∧ s.bill > s.ann ∧ s.ann > s.carol

theorem competition_result (s : Scores) (h : CompetitionConditions s) : CorrectOrder s := by
  sorry

end NUMINAMATH_CALUDE_competition_result_l1222_122276


namespace NUMINAMATH_CALUDE_sequence_properties_l1222_122228

def sequence_a (n : ℕ) : ℝ := 2^n - 1

def S (n : ℕ) : ℝ := 2 * sequence_a n - n

theorem sequence_properties :
  (∀ n : ℕ, S n = 2 * sequence_a n - n) →
  (∀ n : ℕ, sequence_a (n + 1) + 1 = 2 * (sequence_a n + 1)) ∧
  (∀ n : ℕ, sequence_a n = 2^n - 1) ∧
  (∀ k : ℕ, 2 * sequence_a (k + 1) ≠ sequence_a k + sequence_a (k + 2)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1222_122228


namespace NUMINAMATH_CALUDE_size_relationship_l1222_122220

theorem size_relationship (a b c : ℝ) 
  (ha : a = 4^(1/2 : ℝ)) 
  (hb : b = 2^(1/3 : ℝ)) 
  (hc : c = 5^(1/2 : ℝ)) : 
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_size_relationship_l1222_122220


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1222_122284

/-- A point P with coordinates (m+3, m-1) lies on the x-axis if and only if its coordinates are (4, 0) -/
theorem point_on_x_axis (m : ℝ) : 
  (m - 1 = 0 ∧ (m + 3, m - 1) = (m + 3, 0)) ↔ (m + 3, m - 1) = (4, 0) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1222_122284


namespace NUMINAMATH_CALUDE_greatest_number_l1222_122238

theorem greatest_number : 
  8^85 > 5^100 ∧ 8^85 > 6^91 ∧ 8^85 > 7^90 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_l1222_122238


namespace NUMINAMATH_CALUDE_two_digit_number_pair_exists_l1222_122258

theorem two_digit_number_pair_exists : ∃ x y : ℤ,
  10 ≤ x ∧ x < 100 ∧
  10 ≤ y ∧ y < 100 ∧
  x + 15 < 100 ∧
  y - 20 ≥ 10 ∧
  (x + 15) * (y - 20) = x * y :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_pair_exists_l1222_122258


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_of_13_l1222_122219

theorem least_four_digit_multiple_of_13 : ∃ n : ℕ, 
  n % 13 = 0 ∧ 
  n ≥ 1000 ∧ 
  n < 10000 ∧ 
  (∀ m : ℕ, m % 13 = 0 ∧ m ≥ 1000 ∧ m < 10000 → n ≤ m) ∧
  n = 1001 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_of_13_l1222_122219


namespace NUMINAMATH_CALUDE_sample_capacity_proof_l1222_122282

/-- Given a sample divided into groups, prove that if one group has a frequency of 30
    and a frequency rate of 0.25, then the sample capacity is 120. -/
theorem sample_capacity_proof (n : ℕ) (frequency : ℕ) (frequency_rate : ℚ) 
    (h1 : frequency = 30)
    (h2 : frequency_rate = 1/4)
    (h3 : frequency_rate = frequency / n) : n = 120 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_proof_l1222_122282


namespace NUMINAMATH_CALUDE_group_size_proof_l1222_122215

theorem group_size_proof (W : ℝ) (N : ℕ) : 
  ((W + 35) / N = W / N + 3.5) → N = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l1222_122215


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_sums_l1222_122247

-- Define the polynomial
def p (x : ℝ) : ℝ := 10 * x^3 + 101 * x + 210

-- Define the roots
def roots_of_p (a b c : ℝ) : Prop := p a = 0 ∧ p b = 0 ∧ p c = 0

-- Theorem statement
theorem sum_of_cubes_of_sums (a b c : ℝ) :
  roots_of_p a b c → (a + b)^3 + (b + c)^3 + (c + a)^3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_sums_l1222_122247


namespace NUMINAMATH_CALUDE_power_division_rule_l1222_122254

theorem power_division_rule (n : ℕ) : 19^11 / 19^6 = 247609 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1222_122254


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1222_122203

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 9) 
  (h3 : S = a / (1 - r)) : a = 12 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1222_122203


namespace NUMINAMATH_CALUDE_midpoint_quadrilateral_area_l1222_122201

/-- Given a rectangle with length 15 and width 8, the quadrilateral formed by
    connecting the midpoints of its sides has an area of 30 square units. -/
theorem midpoint_quadrilateral_area (l w : ℝ) (hl : l = 15) (hw : w = 8) :
  let midpoint_quad_area := (l / 2) * (w / 2)
  midpoint_quad_area = 30 := by sorry

end NUMINAMATH_CALUDE_midpoint_quadrilateral_area_l1222_122201


namespace NUMINAMATH_CALUDE_illumination_theorem_l1222_122267

/-- Calculates the total number of nights a house can be illuminated given an initial number of candles. -/
def totalNights (initialCandles : ℕ) : ℕ :=
  let rec aux (candles stubs nights : ℕ) : ℕ :=
    if candles = 0 then
      nights + (stubs / 4)
    else
      aux (candles - 1) (stubs + 1) (nights + 1)
  aux initialCandles 0 0

/-- Theorem stating that 43 initial candles result in 57 nights of illumination. -/
theorem illumination_theorem :
  totalNights 43 = 57 := by
  sorry

end NUMINAMATH_CALUDE_illumination_theorem_l1222_122267


namespace NUMINAMATH_CALUDE_abcd_multiplication_l1222_122270

theorem abcd_multiplication (A B C D : ℕ) : 
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) →
  (1000 * A + 100 * B + 10 * C + D) * 9 = 1000 * D + 100 * C + 10 * B + A →
  A = 1 ∧ B = 0 ∧ C = 8 ∧ D = 9 := by
sorry

end NUMINAMATH_CALUDE_abcd_multiplication_l1222_122270


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l1222_122299

def line1 (t : ℝ) : ℝ × ℝ := (3 + 2*t, -4 - 5*t)
def line2 (s : ℝ) : ℝ × ℝ := (2 + 2*s, -6 - 5*s)

def direction : ℝ × ℝ := (2, -5)

theorem parallel_lines_distance :
  let v := (3 - 2, -4 - (-6))
  let projection := ((v.1 * direction.1 + v.2 * direction.2) / (direction.1^2 + direction.2^2)) • direction
  let perpendicular := (v.1 - projection.1, v.2 - projection.2)
  Real.sqrt (perpendicular.1^2 + perpendicular.2^2) = Real.sqrt 2349 / 29 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l1222_122299


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l1222_122293

/-- Represents a distribution of balls into boxes -/
def Distribution := List Nat

/-- Counts the number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def countDistributions (balls : Nat) (boxes : Nat) : Nat :=
  sorry

/-- Theorem: There are 7 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : countDistributions 6 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l1222_122293


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1222_122210

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + 3 * p.2 = 7}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 = -1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(1/2, 3/2)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1222_122210


namespace NUMINAMATH_CALUDE_notecard_problem_l1222_122272

theorem notecard_problem (N E : ℕ) : 
  N - E = 80 →  -- Bill used all envelopes and had 80 notecards left
  3 * E = N →   -- John used all notecards, each letter used 3 notecards
  N = 120       -- The number of notecards in each set is 120
:= by sorry

end NUMINAMATH_CALUDE_notecard_problem_l1222_122272


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1222_122290

theorem arithmetic_mean_problem (x : ℚ) : 
  (((x + 10) + 18 + 3*x + 16 + (x + 5) + (3*x + 6)) / 6 = 25) → x = 95/8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1222_122290


namespace NUMINAMATH_CALUDE_gcd_minus_twelve_equals_thirtysix_l1222_122296

theorem gcd_minus_twelve_equals_thirtysix :
  Nat.gcd 7344 48 - 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcd_minus_twelve_equals_thirtysix_l1222_122296


namespace NUMINAMATH_CALUDE_fraction_product_l1222_122252

theorem fraction_product : (2 : ℚ) / 5 * (3 : ℚ) / 5 = (6 : ℚ) / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l1222_122252


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1222_122263

theorem negation_of_proposition (p : Prop) : 
  (¬(∀ (x : ℝ), x ≥ 0 → Real.log (x^2 + 1) ≥ 0)) ↔ 
  (∃ (x : ℝ), x ≥ 0 ∧ Real.log (x^2 + 1) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1222_122263


namespace NUMINAMATH_CALUDE_officers_selection_count_l1222_122295

/-- Represents the number of ways to choose officers in a club -/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  total_members * girls * (boys - 1)

/-- Theorem: The number of ways to choose officers under given conditions is 6300 -/
theorem officers_selection_count :
  let total_members : ℕ := 30
  let boys : ℕ := 15
  let girls : ℕ := 15
  choose_officers total_members boys girls = 6300 := by
  sorry

#eval choose_officers 30 15 15

end NUMINAMATH_CALUDE_officers_selection_count_l1222_122295


namespace NUMINAMATH_CALUDE_volume_right_prism_isosceles_base_l1222_122278

/-- Volume of a right prism with isosceles triangular base -/
theorem volume_right_prism_isosceles_base 
  (a : ℝ) (α : ℝ) (S : ℝ) 
  (h_a : a > 0) 
  (h_α : 0 < α ∧ α < π) 
  (h_S : S > 0) : 
  ∃ V : ℝ, V = (a * S / 2) * Real.sin (α / 2) * Real.tan ((π - α) / 4) ∧ 
  V = (Real.sin α * a^2 / 2) * (S / (2 * a * (1 + Real.sin (α / 2)))) :=
sorry

end NUMINAMATH_CALUDE_volume_right_prism_isosceles_base_l1222_122278


namespace NUMINAMATH_CALUDE_chord_intersection_ratio_l1222_122253

theorem chord_intersection_ratio (EQ FQ GQ HQ : ℝ) :
  EQ = 5 →
  GQ = 12 →
  HQ = 3 →
  EQ * FQ = GQ * HQ →
  FQ / HQ = 12 / 5 := by
sorry

end NUMINAMATH_CALUDE_chord_intersection_ratio_l1222_122253


namespace NUMINAMATH_CALUDE_teaching_years_difference_l1222_122287

/-- The combined total of teaching years for Virginia, Adrienne, and Dennis -/
def total_years : ℕ := 102

/-- The number of years Dennis has taught -/
def dennis_years : ℕ := 43

/-- The number of years Virginia has taught -/
def virginia_years : ℕ := 34

/-- The number of years Adrienne has taught -/
def adrienne_years : ℕ := 25

theorem teaching_years_difference :
  total_years = virginia_years + adrienne_years + dennis_years ∧
  virginia_years = adrienne_years + 9 ∧
  virginia_years < dennis_years →
  dennis_years - virginia_years = 9 := by
sorry

end NUMINAMATH_CALUDE_teaching_years_difference_l1222_122287


namespace NUMINAMATH_CALUDE_negative_a_fourth_div_negative_a_l1222_122260

theorem negative_a_fourth_div_negative_a (a : ℝ) : (-a)^4 / (-a) = -a^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_fourth_div_negative_a_l1222_122260


namespace NUMINAMATH_CALUDE_find_other_number_l1222_122232

/-- Given two positive integers with known LCM, HCF, and one of the numbers, prove the value of the other number -/
theorem find_other_number (a b : ℕ+) (h1 : Nat.lcm a b = 76176) (h2 : Nat.gcd a b = 116) (h3 : a = 8128) : b = 1087 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l1222_122232


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1222_122244

theorem polynomial_factorization (x : ℝ) : 
  2 * x^3 - 4 * x^2 + 2 * x = 2 * x * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1222_122244


namespace NUMINAMATH_CALUDE_small_bottle_price_l1222_122250

/-- The price of a small bottle given the following conditions:
  * 1375 large bottles were purchased at $1.75 each
  * 690 small bottles were purchased
  * The average price of all bottles is approximately $1.6163438256658595
-/
theorem small_bottle_price :
  let large_bottles : ℕ := 1375
  let small_bottles : ℕ := 690
  let large_price : ℝ := 1.75
  let avg_price : ℝ := 1.6163438256658595
  let total_bottles : ℕ := large_bottles + small_bottles
  let small_price : ℝ := (avg_price * total_bottles - large_price * large_bottles) / small_bottles
  ∃ ε > 0, |small_price - 1.34988436247191| < ε := by
  sorry

end NUMINAMATH_CALUDE_small_bottle_price_l1222_122250


namespace NUMINAMATH_CALUDE_circle_radius_proof_l1222_122277

-- Define the circle and its properties
def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the chord length
def chord_length : ℝ := 10

-- Define the internal segment of the secant
def secant_internal : ℝ := 12

-- Theorem statement
theorem circle_radius_proof (r : ℝ) (h1 : r > 0) :
  ∃ (A B C : ℝ × ℝ),
    A ∈ Circle r ∧ 
    B ∈ Circle r ∧ 
    C ∈ Circle r ∧
    ‖A - B‖ = chord_length ∧
    ‖B - C‖ = secant_internal ∧
    (∃ (D : ℝ × ℝ), D ∈ Circle r ∧ (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0) →
    r = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l1222_122277


namespace NUMINAMATH_CALUDE_range_of_a_l1222_122297

theorem range_of_a (a : ℝ) : 
  (a > 0 ∧ ∀ x : ℝ, x > 0 → (Real.exp x / x + a * Real.log x - a * x + Real.exp 2) ≥ 0) →
  (0 < a ∧ a ≤ Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1222_122297


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l1222_122288

/-- A natural number that ends with 7 zeros and has exactly 72 divisors -/
def SeventyTwoDivisorNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10^7 * k ∧ (Nat.divisors n).card = 72

theorem sum_of_special_numbers :
  ∃ (a b : ℕ), a ≠ b ∧
    SeventyTwoDivisorNumber a ∧
    SeventyTwoDivisorNumber b ∧
    a + b = 70000000 :=
sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l1222_122288


namespace NUMINAMATH_CALUDE_lcm_of_16_27_35_l1222_122205

theorem lcm_of_16_27_35 : Nat.lcm (Nat.lcm 16 27) 35 = 15120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_16_27_35_l1222_122205


namespace NUMINAMATH_CALUDE_larger_number_in_ratio_l1222_122227

theorem larger_number_in_ratio (x y : ℝ) : 
  x > 0 → y > 0 → x + y = 40 → x / y = 3 → x = 30 :=
by sorry

end NUMINAMATH_CALUDE_larger_number_in_ratio_l1222_122227


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1222_122230

/-- Represents a hyperbola with its asymptotic equation coefficient -/
structure Hyperbola where
  k : ℝ
  asymptote_eq : ∀ (x y : ℝ), y = k * x ∨ y = -k * x

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) (h_asymptote : h.k = 1/2) :
  (eccentricity h = Real.sqrt 5 / 2) ∨ (eccentricity h = Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1222_122230


namespace NUMINAMATH_CALUDE_absolute_value_fraction_less_than_one_l1222_122221

theorem absolute_value_fraction_less_than_one (a b : ℝ) 
  (ha : |a| < 1) (hb : |b| < 1) : 
  |((a + b) / (1 + a * b))| < 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_less_than_one_l1222_122221


namespace NUMINAMATH_CALUDE_total_boxes_moved_l1222_122255

/-- The number of boxes a truck can hold -/
def boxes_per_truck : ℕ := 4

/-- The number of trips taken to move all boxes -/
def num_trips : ℕ := 218

/-- The total number of boxes moved -/
def total_boxes : ℕ := boxes_per_truck * num_trips

theorem total_boxes_moved :
  total_boxes = 872 := by sorry

end NUMINAMATH_CALUDE_total_boxes_moved_l1222_122255


namespace NUMINAMATH_CALUDE_max_distance_theorem_l1222_122213

/-- Given points in a 2D Cartesian coordinate system, prove the maximum distance -/
theorem max_distance_theorem (x y : ℝ) : 
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (0, Real.sqrt 3)
  let C : ℝ × ℝ := (3, 0)
  let D : ℝ × ℝ := (x, y)
  (x - 3)^2 + y^2 = 1 →
  (∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + y₀^2 = 1 ∧ 
    ∀ (x' y' : ℝ), (x' - 3)^2 + y'^2 = 1 → 
      ((x' - 1)^2 + (y' + Real.sqrt 3)^2) ≤ ((x₀ - 1)^2 + (y₀ + Real.sqrt 3)^2)) ∧
  ((x₀ - 1)^2 + (y₀ + Real.sqrt 3)^2) = (Real.sqrt 7 + 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_theorem_l1222_122213


namespace NUMINAMATH_CALUDE_class_average_age_l1222_122269

theorem class_average_age (original_students : ℕ) (new_students : ℕ) (new_average_age : ℕ) (average_decrease : ℕ) :
  original_students = 12 →
  new_students = 12 →
  new_average_age = 32 →
  average_decrease = 4 →
  ∃ (original_average : ℕ),
    (original_students * original_average + new_students * new_average_age) / (original_students + new_students) = original_average - average_decrease ∧
    original_average = 40 :=
by sorry

end NUMINAMATH_CALUDE_class_average_age_l1222_122269


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1222_122236

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  let z : ℂ := 5 * i / (1 - 2 * i)
  z = -2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1222_122236


namespace NUMINAMATH_CALUDE_line_direction_vector_b_l1222_122235

def point_1 : ℝ × ℝ := (-3, 1)
def point_2 : ℝ × ℝ := (1, 5)

def direction_vector (b : ℝ) : ℝ × ℝ := (3, b)

theorem line_direction_vector_b (b : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ direction_vector b = k • (point_2 - point_1)) → b = 3 :=
by sorry

end NUMINAMATH_CALUDE_line_direction_vector_b_l1222_122235


namespace NUMINAMATH_CALUDE_perfect_square_expression_l1222_122241

theorem perfect_square_expression : ∃ y : ℝ, (11.98 * 11.98 + 11.98 * 0.4792 + 0.02 * 0.02) = y^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_expression_l1222_122241


namespace NUMINAMATH_CALUDE_beth_bought_ten_cans_of_corn_l1222_122249

/-- The number of cans of corn Beth bought -/
def cans_of_corn : ℕ := sorry

/-- The number of cans of peas Beth bought -/
def cans_of_peas : ℕ := 35

/-- The relationship between cans of peas and cans of corn -/
axiom peas_corn_relation : cans_of_peas = 15 + 2 * cans_of_corn

theorem beth_bought_ten_cans_of_corn : cans_of_corn = 10 := by sorry

end NUMINAMATH_CALUDE_beth_bought_ten_cans_of_corn_l1222_122249


namespace NUMINAMATH_CALUDE_andrews_appetizers_l1222_122279

/-- The number of hotdogs on sticks Andrew brought -/
def hotdogs : ℕ := 30

/-- The number of bite-sized cheese pops Andrew brought -/
def cheese_pops : ℕ := 20

/-- The number of chicken nuggets Andrew brought -/
def chicken_nuggets : ℕ := 40

/-- The total number of appetizer portions Andrew brought -/
def total_appetizers : ℕ := hotdogs + cheese_pops + chicken_nuggets

theorem andrews_appetizers :
  total_appetizers = 90 :=
by sorry

end NUMINAMATH_CALUDE_andrews_appetizers_l1222_122279


namespace NUMINAMATH_CALUDE_line_passes_through_P_triangle_perimeter_l1222_122264

/-- The equation of line l is (a+1)x + y - 5 - 2a = 0, where a ∈ ℝ -/
def line_equation (a x y : ℝ) : Prop := (a + 1) * x + y - 5 - 2 * a = 0

/-- Point P that the line passes through -/
def point_P : ℝ × ℝ := (2, 3)

/-- The area of triangle AOB -/
def triangle_area : ℝ := 12

theorem line_passes_through_P (a : ℝ) : line_equation a (point_P.1) (point_P.2) := by sorry

theorem triangle_perimeter : 
  ∃ (a x_A y_B : ℝ), 
    line_equation a x_A 0 ∧ 
    line_equation a 0 y_B ∧ 
    x_A * y_B / 2 = triangle_area ∧ 
    x_A + y_B + Real.sqrt (x_A^2 + y_B^2) = 10 + 2 * Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_P_triangle_perimeter_l1222_122264

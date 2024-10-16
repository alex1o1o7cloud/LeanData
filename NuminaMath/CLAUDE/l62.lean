import Mathlib

namespace NUMINAMATH_CALUDE_game_probability_difference_l62_6299

def coin_prob_heads : ℚ := 2/3
def coin_prob_tails : ℚ := 1/3

def game_x_win_prob : ℚ :=
  3 * (coin_prob_heads^2 * coin_prob_tails) + coin_prob_heads^3

def game_y_win_prob : ℚ :=
  4 * (coin_prob_heads^3 * coin_prob_tails + coin_prob_tails^3 * coin_prob_heads) +
  coin_prob_heads^4 + coin_prob_tails^4

theorem game_probability_difference :
  game_x_win_prob - game_y_win_prob = 11/81 :=
sorry

end NUMINAMATH_CALUDE_game_probability_difference_l62_6299


namespace NUMINAMATH_CALUDE_no_real_solutions_quadratic_l62_6267

theorem no_real_solutions_quadratic (k : ℝ) :
  (∀ x : ℝ, x^2 - 3*x - k ≠ 0) ↔ k < -9/4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_quadratic_l62_6267


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l62_6206

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 4th term is 23 and the 9th term is 38, the 10th term is 41. -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) (h : ArithmeticSequence a) 
    (h4 : a 4 = 23) (h9 : a 9 = 38) : a 10 = 41 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l62_6206


namespace NUMINAMATH_CALUDE_rem_one_third_neg_three_fourths_l62_6289

-- Definition of the remainder function
def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

-- Theorem statement
theorem rem_one_third_neg_three_fourths :
  rem (1/3) (-3/4) = -5/12 := by sorry

end NUMINAMATH_CALUDE_rem_one_third_neg_three_fourths_l62_6289


namespace NUMINAMATH_CALUDE_no_primes_satisfy_equation_l62_6290

theorem no_primes_satisfy_equation :
  ∀ (p q : ℕ) (n : ℕ+), 
    Prime p → Prime q → p ≠ q → p^(q-1) - q^(p-1) ≠ 4*(n:ℕ)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_primes_satisfy_equation_l62_6290


namespace NUMINAMATH_CALUDE_geometric_sequence_max_value_l62_6225

/-- Given a geometric sequence {a_n} with common ratio √2, 
    T_n = (17S_n - S_{2n}) / a_{n+1} attains its maximum value when n = 4 -/
theorem geometric_sequence_max_value (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n * Real.sqrt 2) →
  (∀ n, S n = a 1 * (1 - (Real.sqrt 2)^n) / (1 - Real.sqrt 2)) →
  (∀ n, T n = (17 * S n - S (2 * n)) / a (n + 1)) →
  (∃ B : ℝ, ∀ n, T n ≤ B ∧ T 4 = B) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_max_value_l62_6225


namespace NUMINAMATH_CALUDE_circle_center_proof_l62_6269

theorem circle_center_proof (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) :
  -- The circle passes through (1,0)
  (1, 0) ∈ C →
  -- The circle is tangent to y = x^2 at (2,4)
  (2, 4) ∈ C →
  (∀ (x y : ℝ), (x, y) ∈ C → y ≠ x^2 ∨ (x = 2 ∧ y = 4)) →
  -- The circle is tangent to the x-axis
  (∃ (x : ℝ), (x, 0) ∈ C ∧ ∀ (y : ℝ), y ≠ 0 → (x, y) ∉ C) →
  -- C is a circle with center 'center'
  (∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = (1 - center.1)^2 + center.2^2) →
  -- The center is (178/15, 53/15)
  center = (178/15, 53/15) := by
sorry

end NUMINAMATH_CALUDE_circle_center_proof_l62_6269


namespace NUMINAMATH_CALUDE_platform_length_l62_6249

/-- Given a train of length 900 m that takes 39 sec to cross a platform and 18 sec to cross a signal pole, the length of the platform is 1050 m. -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 900)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 18) :
  train_length + (train_length / time_cross_pole * time_cross_platform) - train_length = 1050 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l62_6249


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l62_6259

theorem greatest_integer_radius (r : ℝ) (h : r > 0) (area_constraint : π * r^2 < 100 * π) :
  ⌊r⌋ ≤ 9 ∧ ∃ (r' : ℝ), π * r'^2 < 100 * π ∧ ⌊r'⌋ = 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l62_6259


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l62_6219

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    if the distance from one focus to an asymptote is √5/3 * c,
    where c is the semi-focal length, then the eccentricity is 3/2. -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  (c * b / Real.sqrt (a^2 + b^2) = c * Real.sqrt 5 / 3) →
  c^2 = a^2 + b^2 →
  c / a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l62_6219


namespace NUMINAMATH_CALUDE_exactly_one_statement_correct_l62_6246

-- Define rational and irrational numbers
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- Define the four statements
def Statement1 : Prop :=
  ∀ (r i : ℝ), IsRational r → IsIrrational i → IsIrrational (r + i)

def Statement2 : Prop :=
  ∀ (r i : ℝ), IsRational r → IsIrrational i → IsIrrational (r * i)

def Statement3 : Prop :=
  ∀ (i₁ i₂ : ℝ), IsIrrational i₁ → IsIrrational i₂ → IsIrrational (i₁ + i₂)

def Statement4 : Prop :=
  ∀ (i₁ i₂ : ℝ), IsIrrational i₁ → IsIrrational i₂ → IsIrrational (i₁ * i₂)

-- The main theorem
theorem exactly_one_statement_correct :
  (Statement1 ∧ ¬Statement2 ∧ ¬Statement3 ∧ ¬Statement4) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_statement_correct_l62_6246


namespace NUMINAMATH_CALUDE_problem_solution_l62_6272

theorem problem_solution : ∃ x : ℝ, 400 * x = 28000 * 100^1 ∧ x = 7000 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l62_6272


namespace NUMINAMATH_CALUDE_katie_earnings_l62_6209

/-- The number of bead necklaces sold -/
def bead_necklaces : ℕ := 4

/-- The number of gem stone necklaces sold -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 3

/-- The total money earned by Katie -/
def total_money : ℕ := (bead_necklaces + gem_necklaces) * necklace_cost

theorem katie_earnings : total_money = 21 := by
  sorry

end NUMINAMATH_CALUDE_katie_earnings_l62_6209


namespace NUMINAMATH_CALUDE_jacob_lunch_calories_l62_6291

theorem jacob_lunch_calories (planned : ℕ) (breakfast dinner extra : ℕ) 
  (h1 : planned < 1800)
  (h2 : breakfast = 400)
  (h3 : dinner = 1100)
  (h4 : extra = 600) :
  planned + extra - (breakfast + dinner) = 900 :=
by sorry

end NUMINAMATH_CALUDE_jacob_lunch_calories_l62_6291


namespace NUMINAMATH_CALUDE_simplify_polynomial_l62_6203

theorem simplify_polynomial (w : ℝ) : 
  3 * w + 5 - 6 * w^2 + 4 * w - 7 + 9 * w^2 = 3 * w^2 + 7 * w - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l62_6203


namespace NUMINAMATH_CALUDE_part_one_part_two_l62_6248

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem for part (I)
theorem part_one (a : ℝ) : p a → a ≤ 1 := by
  sorry

-- Theorem for part (II)
theorem part_two (a : ℝ) : ¬(p a ∧ q a) → a ∈ Set.union (Set.Ioo (-2) 1) (Set.Ioi 1) := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l62_6248


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l62_6288

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 - 6*p^3 + 11*p^2 - 6*p + 3 = 0) →
  (q^4 - 6*q^3 + 11*q^2 - 6*q + 3 = 0) →
  (r^4 - 6*r^3 + 11*r^2 - 6*r + 3 = 0) →
  (s^4 - 6*s^3 + 11*s^2 - 6*s + 3 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l62_6288


namespace NUMINAMATH_CALUDE_food_allowance_per_teacher_l62_6264

/-- Calculates the food allowance per teacher given the seminar details and total spent --/
theorem food_allowance_per_teacher
  (regular_fee : ℝ)
  (discount_rate : ℝ)
  (num_teachers : ℕ)
  (total_spent : ℝ)
  (h1 : regular_fee = 150)
  (h2 : discount_rate = 0.05)
  (h3 : num_teachers = 10)
  (h4 : total_spent = 1525)
  : (total_spent - num_teachers * (regular_fee * (1 - discount_rate))) / num_teachers = 10 := by
  sorry

#check food_allowance_per_teacher

end NUMINAMATH_CALUDE_food_allowance_per_teacher_l62_6264


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l62_6210

/-- The eccentricity of a hyperbola passing through the focus of a specific parabola -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) : 
  let parabola := fun x y : ℝ => y^2 = 8*x
  let hyperbola := fun x y : ℝ => x^2/a^2 - y^2 = 1
  let focus : ℝ × ℝ := (2, 0)
  (∀ x y, parabola x y → (x, y) = focus) →
  hyperbola (focus.1) (focus.2) →
  let e := Real.sqrt ((a^2 + a^2) / a^2)
  e = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l62_6210


namespace NUMINAMATH_CALUDE_num_solutions_is_four_l62_6265

/-- The number of distinct solutions to the system of equations:
    (x - 2y + 3)(4x + y - 5) = 0 and (x + 2y - 5)(3x - 4y + 6) = 0 -/
def num_solutions : ℕ :=
  let eq1 (x y : ℝ) := (x - 2*y + 3)*(4*x + y - 5) = 0
  let eq2 (x y : ℝ) := (x + 2*y - 5)*(3*x - 4*y + 6) = 0
  4  -- The actual number of solutions

theorem num_solutions_is_four :
  num_solutions = 4 := by sorry

end NUMINAMATH_CALUDE_num_solutions_is_four_l62_6265


namespace NUMINAMATH_CALUDE_outfit_combinations_l62_6207

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of pairs of pants available -/
def num_pants : ℕ := 5

/-- The number of ties available -/
def num_ties : ℕ := 6

/-- The number of jackets available -/
def num_jackets : ℕ := 2

/-- The number of different outfits that can be created -/
def num_outfits : ℕ := num_shirts * num_pants * (num_ties + 1) * (num_jackets + 1)

/-- Theorem stating that the number of different outfits is 840 -/
theorem outfit_combinations : num_outfits = 840 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l62_6207


namespace NUMINAMATH_CALUDE_ellipse_a_range_l62_6245

-- Define the ellipse equation
def ellipse_equation (x y a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (a + 6) = 1

-- Define the condition that the ellipse has foci on the x-axis
def foci_on_x_axis (a : ℝ) : Prop :=
  a^2 > a + 6 ∧ a + 6 > 0

-- Theorem stating the range of a
theorem ellipse_a_range :
  ∀ a : ℝ, (∃ x y : ℝ, ellipse_equation x y a ∧ foci_on_x_axis a) →
  (a > 3 ∨ (-6 < a ∧ a < -2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_a_range_l62_6245


namespace NUMINAMATH_CALUDE_sum_of_integers_l62_6298

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val - y.val = 4) 
  (h2 : x.val * y.val = 98) : 
  x.val + y.val = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l62_6298


namespace NUMINAMATH_CALUDE_total_players_count_l62_6235

theorem total_players_count (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ) : 
  kabadi = 10 → kho_kho_only = 30 → both = 5 → 
  kabadi + kho_kho_only - both = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_players_count_l62_6235


namespace NUMINAMATH_CALUDE_perpendicular_vectors_lambda_l62_6221

/-- Given two vectors a and b in ℝ², where a is perpendicular to (a - b), prove that the second component of b equals 4. -/
theorem perpendicular_vectors_lambda (a b : ℝ × ℝ) : 
  a = (-1, 3) → 
  b.1 = 2 → 
  a • (a - b) = 0 → 
  b.2 = 4 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_lambda_l62_6221


namespace NUMINAMATH_CALUDE_bobby_candy_total_l62_6263

theorem bobby_candy_total (initial_candy : ℕ) (more_candy : ℕ) (chocolate : ℕ)
  (h1 : initial_candy = 28)
  (h2 : more_candy = 42)
  (h3 : chocolate = 63) :
  initial_candy + more_candy + chocolate = 133 :=
by sorry

end NUMINAMATH_CALUDE_bobby_candy_total_l62_6263


namespace NUMINAMATH_CALUDE_parabola_intersection_l62_6228

/-- Given a parabola y = x^2 and four points on it, if two lines formed by these points
    intersect on the y-axis, then the x-coordinate of the fourth point is determined by
    the x-coordinates of the other three points. -/
theorem parabola_intersection (a b c d : ℝ) : 
  (∃ l : ℝ, (a^2 = (b + a)*a + l ∧ b^2 = (b + a)*b + l) ∧ 
             (c^2 = (d + c)*c + l ∧ d^2 = (d + c)*d + l)) →
  d = a * b / c :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_l62_6228


namespace NUMINAMATH_CALUDE_tank_capacity_l62_6200

/-- The capacity of a tank given specific inlet and outlet pipe rates --/
theorem tank_capacity 
  (outlet_time : ℝ) 
  (inlet_rate1 : ℝ) 
  (inlet_rate2 : ℝ) 
  (extended_time : ℝ) 
  (h1 : outlet_time = 10) 
  (h2 : inlet_rate1 = 4) 
  (h3 : inlet_rate2 = 6) 
  (h4 : extended_time = 8) : 
  ∃ (capacity : ℝ), 
    capacity = 13500 ∧ 
    capacity / outlet_time - (inlet_rate1 * 60 + inlet_rate2 * 60) = capacity / (outlet_time + extended_time) :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l62_6200


namespace NUMINAMATH_CALUDE_distance_between_A_and_B_l62_6277

/-- Represents a person traveling from point A to B -/
structure Traveler where
  departureTime : ℕ  -- departure time in minutes after 8:00
  speed : ℝ          -- speed in meters per minute

/-- The problem setup -/
def travelProblem (v : ℝ) : Prop :=
  let personA : Traveler := ⟨0, v⟩
  let personB : Traveler := ⟨20, v⟩
  let personC : Traveler := ⟨30, v⟩
  let totalDistance : ℝ := 60 * v
  
  -- At 8:40 (40 minutes after 8:00), A's remaining distance is half of B's
  (totalDistance - 40 * v) = (1/2) * (totalDistance - 20 * v) ∧
  -- At 8:40, C is 2015 meters away from B
  (totalDistance - 10 * v) = 2015

theorem distance_between_A_and_B :
  ∃ v : ℝ, travelProblem v → 60 * v = 2418 :=
sorry

end NUMINAMATH_CALUDE_distance_between_A_and_B_l62_6277


namespace NUMINAMATH_CALUDE_additional_spheres_in_cone_l62_6262

/-- Represents a truncated cone -/
structure TruncatedCone where
  height : ℝ
  lower_radius : ℝ
  upper_radius : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Function to check if a sphere is tangent to the cone's surfaces -/
def is_tangent_to_cone (s : Sphere) (c : TruncatedCone) : Prop :=
  sorry

/-- Function to check if two spheres are tangent -/
def are_spheres_tangent (s1 s2 : Sphere) : Prop :=
  sorry

/-- Function to calculate the maximum number of additional spheres -/
def max_additional_spheres (c : TruncatedCone) (s1 s2 : Sphere) : ℕ :=
  sorry

/-- Main theorem -/
theorem additional_spheres_in_cone 
  (c : TruncatedCone) 
  (s1 s2 : Sphere) :
  c.height = 8 ∧
  s1.radius = 2 ∧
  s2.radius = 3 ∧
  is_tangent_to_cone s1 c ∧
  is_tangent_to_cone s2 c ∧
  are_spheres_tangent s1 s2 →
  max_additional_spheres c s1 s2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_additional_spheres_in_cone_l62_6262


namespace NUMINAMATH_CALUDE_complex_magnitude_l62_6222

theorem complex_magnitude (z : ℂ) : z = (1 + Complex.I) / (2 - 2 * Complex.I) → Complex.abs z = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l62_6222


namespace NUMINAMATH_CALUDE_eric_return_time_l62_6260

def running_time : ℕ := 20
def jogging_time : ℕ := 10
def time_to_park : ℕ := running_time + jogging_time
def return_time_factor : ℕ := 3

theorem eric_return_time : time_to_park * return_time_factor = 90 := by
  sorry

end NUMINAMATH_CALUDE_eric_return_time_l62_6260


namespace NUMINAMATH_CALUDE_correct_package_cost_l62_6230

def packageCost (P : ℕ) : ℕ :=
  15 + 5 * (P - 1) - 8 * (if P ≥ 5 then 1 else 0)

theorem correct_package_cost (P : ℕ) (h : P ≥ 1) :
  packageCost P = 15 + 5 * (P - 1) - 8 * (if P ≥ 5 then 1 else 0) :=
by sorry

end NUMINAMATH_CALUDE_correct_package_cost_l62_6230


namespace NUMINAMATH_CALUDE_complex_power_four_l62_6242

theorem complex_power_four : 
  (3 * (Complex.cos (π / 6) + Complex.I * Complex.sin (π / 6)))^4 = 
  Complex.mk (-40.5) (40.5 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l62_6242


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l62_6257

/-- Given two adjacent points (1,2) and (4,6) on a square in a Cartesian coordinate plane,
    the area of the square is 25. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l62_6257


namespace NUMINAMATH_CALUDE_vasya_late_l62_6296

/-- Proves that Vasya did not arrive on time given the conditions of his journey -/
theorem vasya_late (v : ℝ) (h : v > 0) : 
  (10 / v + 16 / (v / 2.5) + 24 / (6 * v)) > (50 / v) := by
  sorry

#check vasya_late

end NUMINAMATH_CALUDE_vasya_late_l62_6296


namespace NUMINAMATH_CALUDE_soccer_league_female_fraction_l62_6233

theorem soccer_league_female_fraction :
  let last_year_males : ℕ := 30
  let male_increase_rate : ℚ := 11/10
  let female_increase_rate : ℚ := 5/4
  let total_increase_rate : ℚ := 23/20
  let this_year_males : ℚ := last_year_males * male_increase_rate
  let last_year_females : ℚ := (total_increase_rate * (last_year_males : ℚ) - this_year_males) / (female_increase_rate - total_increase_rate)
  let this_year_females : ℚ := last_year_females * female_increase_rate
  let this_year_total : ℚ := this_year_males + this_year_females
  
  (this_year_females / this_year_total) = 75/207 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_female_fraction_l62_6233


namespace NUMINAMATH_CALUDE_razorback_shop_tshirt_revenue_l62_6285

/-- The amount of money made from each t-shirt -/
def tshirt_price : ℕ := 62

/-- The number of t-shirts sold -/
def tshirts_sold : ℕ := 183

/-- The total money made from selling t-shirts -/
def total_tshirt_money : ℕ := tshirt_price * tshirts_sold

theorem razorback_shop_tshirt_revenue : total_tshirt_money = 11346 := by
  sorry

end NUMINAMATH_CALUDE_razorback_shop_tshirt_revenue_l62_6285


namespace NUMINAMATH_CALUDE_p_decreasing_zero_l62_6229

/-- The probability that |T-H| = k after a game with 4m coins, 
    where T is the number of tails and H is the number of heads. -/
def p (m : ℕ) (k : ℕ) : ℝ := sorry

/-- The optimal strategy for the coin flipping game -/
def optimal_strategy : sorry := sorry

axiom p_zero_zero : p 0 0 = 1

axiom p_zero_pos : ∀ k : ℕ, k ≥ 1 → p 0 k = 0

/-- The main theorem: p_m(0) ≥ p_m+1(0) for all nonnegative integers m -/
theorem p_decreasing_zero : ∀ m : ℕ, p m 0 ≥ p (m + 1) 0 := by sorry

end NUMINAMATH_CALUDE_p_decreasing_zero_l62_6229


namespace NUMINAMATH_CALUDE_estimate_three_plus_sqrt_ten_l62_6250

theorem estimate_three_plus_sqrt_ten : 6 < 3 + Real.sqrt 10 ∧ 3 + Real.sqrt 10 < 7 := by
  sorry

end NUMINAMATH_CALUDE_estimate_three_plus_sqrt_ten_l62_6250


namespace NUMINAMATH_CALUDE_pentagonal_tiles_count_l62_6270

theorem pentagonal_tiles_count (t s p : ℕ) : 
  t + s + p = 30 →
  3 * t + 4 * s + 5 * p = 128 →
  p = 10 :=
by sorry

end NUMINAMATH_CALUDE_pentagonal_tiles_count_l62_6270


namespace NUMINAMATH_CALUDE_angle_ADF_measure_l62_6258

-- Define the circle O and points A, B, C, D, E, F
variable (O : ℝ × ℝ) (A B C D E F : ℝ × ℝ)

-- Define the circle's radius
variable (r : ℝ)

-- Define the angle measure function
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

-- State the given conditions
axiom C_on_BE_extension : sorry
axiom CA_tangent : sorry
axiom DC_bisects_ACB : angle_measure C A D = angle_measure C B D
axiom DC_intersects_AE : sorry
axiom DC_intersects_AB : sorry

-- Define the theorem
theorem angle_ADF_measure :
  angle_measure A D F = 67.5 := sorry

end NUMINAMATH_CALUDE_angle_ADF_measure_l62_6258


namespace NUMINAMATH_CALUDE_train_crossing_time_l62_6287

/-- Theorem: Time taken for two trains to cross each other
    Given two trains moving in opposite directions with specified speeds and lengths,
    prove that the time taken for the slower train to cross the faster train is 24 seconds. -/
theorem train_crossing_time (speed1 speed2 length1 length2 : ℝ) 
    (h1 : speed1 = 315)
    (h2 : speed2 = 135)
    (h3 : length1 = 1.65)
    (h4 : length2 = 1.35) :
    (length1 + length2) / (speed1 + speed2) * 3600 = 24 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l62_6287


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l62_6220

theorem no_real_roots_quadratic : 
  ∀ x : ℝ, 2 * x^2 - 5 * x + 6 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l62_6220


namespace NUMINAMATH_CALUDE_final_student_count_l62_6283

/-- Calculates the number of students on a bus after three stops -/
def studentsOnBus (initial : ℝ) (stop1On stop1Off stop2On stop2Off stop3On stop3Off : ℝ) : ℝ :=
  initial + (stop1On - stop1Off) + (stop2On - stop2Off) + (stop3On - stop3Off)

/-- Theorem stating the final number of students on the bus -/
theorem final_student_count :
  studentsOnBus 21 7.5 2 1.2 5.3 11 4.8 = 28.6 := by
  sorry

end NUMINAMATH_CALUDE_final_student_count_l62_6283


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l62_6256

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 16 * x - 9 * y^2 + 18 * y - 23 = 0

-- State the theorem
theorem hyperbola_vertices_distance :
  ∃ (a b c d : ℝ),
    (∀ x y, hyperbola_equation x y ↔ ((x - a)^2 / b^2 - (y - c)^2 / d^2 = 1)) ∧
    2 * Real.sqrt b^2 = Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l62_6256


namespace NUMINAMATH_CALUDE_jessica_fraction_proof_l62_6214

/-- Represents Jessica's collection of quarters -/
structure QuarterCollection where
  total : ℕ
  from_1790s : ℕ

/-- The fraction of quarters from states admitted in 1790-1799 -/
def fraction_from_1790s (c : QuarterCollection) : ℚ :=
  c.from_1790s / c.total

/-- Jessica's actual collection -/
def jessica_collection : QuarterCollection :=
  { total := 30, from_1790s := 16 }

theorem jessica_fraction_proof :
  fraction_from_1790s jessica_collection = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_jessica_fraction_proof_l62_6214


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l62_6205

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = 2*x ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l62_6205


namespace NUMINAMATH_CALUDE_roses_to_sister_l62_6213

-- Define the initial number of roses
def initial_roses : ℕ := 20

-- Define the number of roses given to mother
def roses_to_mother : ℕ := 6

-- Define the number of roses given to grandmother
def roses_to_grandmother : ℕ := 9

-- Define the number of roses Ian kept for himself
def roses_kept : ℕ := 1

-- Theorem to prove
theorem roses_to_sister : 
  initial_roses - (roses_to_mother + roses_to_grandmother + roses_kept) = 4 := by
  sorry

end NUMINAMATH_CALUDE_roses_to_sister_l62_6213


namespace NUMINAMATH_CALUDE_composition_ratio_l62_6236

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 2 * x - 1

theorem composition_ratio : f (g (f 3)) / g (f (g 3)) = 79 / 37 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l62_6236


namespace NUMINAMATH_CALUDE_wall_width_calculation_l62_6252

/-- Calculates the width of a wall given brick dimensions and wall specifications -/
theorem wall_width_calculation (brick_length brick_width brick_height : ℝ)
  (wall_length wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_length = 29 →
  wall_height = 2 →
  num_bricks = 29000 →
  (brick_length * brick_width * brick_height * num_bricks) / (wall_length * wall_height) = 7.5 := by
  sorry

#check wall_width_calculation

end NUMINAMATH_CALUDE_wall_width_calculation_l62_6252


namespace NUMINAMATH_CALUDE_min_value_arithmetic_progression_l62_6294

/-- Given real numbers x, y, z in [0, 4] where x^2, y^2, z^2 form an arithmetic progression 
    with common difference 2, the minimum value of |x-y|+|y-z| is 4 - 2√3 -/
theorem min_value_arithmetic_progression (x y z : ℝ) 
  (h1 : 0 ≤ x ∧ x ≤ 4) 
  (h2 : 0 ≤ y ∧ y ≤ 4) 
  (h3 : 0 ≤ z ∧ z ≤ 4) 
  (h4 : y^2 - x^2 = z^2 - y^2) 
  (h5 : y^2 - x^2 = 2) : 
  ∃ (m : ℝ), m = 4 - 2 * Real.sqrt 3 ∧ 
  ∀ (x' y' z' : ℝ), 0 ≤ x' ∧ x' ≤ 4 → 0 ≤ y' ∧ y' ≤ 4 → 0 ≤ z' ∧ z' ≤ 4 → 
  y'^2 - x'^2 = z'^2 - y'^2 → y'^2 - x'^2 = 2 → 
  m ≤ |x' - y'| + |y' - z'| :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_arithmetic_progression_l62_6294


namespace NUMINAMATH_CALUDE_inequality_proof_l62_6251

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b * c * (a + b + c) = 3) :
  (a + b) * (b + c) * (c + a) ≥ 8 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l62_6251


namespace NUMINAMATH_CALUDE_average_age_of_ten_students_l62_6227

theorem average_age_of_ten_students
  (total_students : Nat)
  (total_average_age : ℝ)
  (nine_students_average : ℝ)
  (twentieth_student_age : ℝ)
  (h1 : total_students = 20)
  (h2 : total_average_age = 20)
  (h3 : nine_students_average = 11)
  (h4 : twentieth_student_age = 61) :
  let remaining_students := total_students - 10
  let total_age := total_students * total_average_age
  let nine_students_total_age := 9 * nine_students_average
  let ten_students_total_age := total_age - nine_students_total_age - twentieth_student_age
  ten_students_total_age / remaining_students = 24 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_ten_students_l62_6227


namespace NUMINAMATH_CALUDE_zach_lawn_mowing_pay_l62_6237

/-- Represents the financial situation for Zach's bike savings --/
structure BikeSavings where
  bikeCost : ℕ
  weeklyAllowance : ℕ
  currentSavings : ℕ
  babysittingPayRate : ℕ
  babysittingHours : ℕ
  additionalNeeded : ℕ

/-- Calculates the amount Zach's parent should pay him to mow the lawn --/
def lawnMowingPay (s : BikeSavings) : ℕ :=
  s.bikeCost - s.currentSavings - s.weeklyAllowance - s.babysittingPayRate * s.babysittingHours - s.additionalNeeded

/-- Theorem stating that the amount Zach's parent will pay him to mow the lawn is 10 --/
theorem zach_lawn_mowing_pay :
  let s : BikeSavings := {
    bikeCost := 100
    weeklyAllowance := 5
    currentSavings := 65
    babysittingPayRate := 7
    babysittingHours := 2
    additionalNeeded := 6
  }
  lawnMowingPay s = 10 := by sorry

end NUMINAMATH_CALUDE_zach_lawn_mowing_pay_l62_6237


namespace NUMINAMATH_CALUDE_pond_draining_time_l62_6253

theorem pond_draining_time (total_volume : ℝ) (pump_rate : ℝ) (h1 : pump_rate > 0) : 
  total_volume = 15 * 24 * pump_rate →
  ∃ (remaining_time : ℝ),
    remaining_time ≥ 144 ∧
    3 * 24 * pump_rate + remaining_time * (2 * pump_rate) = total_volume :=
by sorry

end NUMINAMATH_CALUDE_pond_draining_time_l62_6253


namespace NUMINAMATH_CALUDE_solution_set_f_shifted_empty_solution_set_l62_6268

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for part 1
theorem solution_set_f_shifted (x : ℝ) :
  f (x + 2) ≥ 2 ↔ x ≤ -3/2 ∨ x ≥ 1/2 :=
sorry

-- Theorem for part 2
theorem empty_solution_set (a : ℝ) :
  (∀ x, f x ≥ a) ↔ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_shifted_empty_solution_set_l62_6268


namespace NUMINAMATH_CALUDE_smallest_solution_equation_smallest_solution_l62_6282

theorem smallest_solution_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 5 / (2 * (x - 4))) ↔ x = 4 - Real.sqrt 5 ∨ x = 4 + Real.sqrt 5 :=
sorry

theorem smallest_solution (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 5 / (2 * (x - 4))) ∧ 
  (∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 5 / (2 * (y - 4))) → y ≥ x) →
  x = 4 - Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_equation_smallest_solution_l62_6282


namespace NUMINAMATH_CALUDE_square_side_length_l62_6224

theorem square_side_length
  (a : ℝ) -- side length of the square
  (x : ℝ) -- one leg of the right triangle
  (b : ℝ) -- hypotenuse of the right triangle
  (h1 : 4 * a + 2 * x = 58) -- perimeter of rectangle
  (h2 : 2 * a + 2 * b + 2 * x = 60) -- perimeter of trapezoid
  (h3 : a^2 + x^2 = b^2) -- Pythagorean theorem
  : a = 12 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l62_6224


namespace NUMINAMATH_CALUDE_reflection_line_sum_l62_6281

/-- Given a line y = mx + b, if the reflection of point (1, -2) across this line is (7, 4), then m + b = 4 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The midpoint of the segment is on the line
    y = m * x + b ∧ 
    -- The midpoint coordinates
    x = (1 + 7) / 2 ∧ 
    y = (-2 + 4) / 2 ∧ 
    -- The line is perpendicular to the segment
    m * ((7 - 1) / (4 - (-2))) = -1) → 
  m + b = 4 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l62_6281


namespace NUMINAMATH_CALUDE_valid_paths_count_l62_6218

/-- The number of paths on a grid from (0,0) to (m,n) -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The number of paths from A to C on the 6x3 grid -/
def pathsAtoC : ℕ := gridPaths 4 1

/-- The number of paths from D to B on the 6x3 grid -/
def pathsDtoB : ℕ := gridPaths 2 2

/-- The number of paths from A to E on the 6x3 grid -/
def pathsAtoE : ℕ := gridPaths 2 2

/-- The number of paths from F to B on the 6x3 grid -/
def pathsFtoB : ℕ := gridPaths 4 0

/-- The total number of paths on the 6x3 grid -/
def totalPaths : ℕ := gridPaths 6 3

/-- The number of invalid paths through the first forbidden segment -/
def invalidPaths1 : ℕ := pathsAtoC * pathsDtoB

/-- The number of invalid paths through the second forbidden segment -/
def invalidPaths2 : ℕ := pathsAtoE * pathsFtoB

theorem valid_paths_count :
  totalPaths - (invalidPaths1 + invalidPaths2) = 48 := by sorry

end NUMINAMATH_CALUDE_valid_paths_count_l62_6218


namespace NUMINAMATH_CALUDE_absolute_value_integral_l62_6284

theorem absolute_value_integral : ∫ x in (0)..(4), |x - 2| = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_integral_l62_6284


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l62_6276

theorem count_integers_satisfying_inequality : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, (169 * n : ℝ)^25 > (n : ℝ)^75 ∧ (n : ℝ)^75 > 3^150) ∧ 
    (∀ n : ℕ, (169 * n : ℝ)^25 > (n : ℝ)^75 ∧ (n : ℝ)^75 > 3^150 → n ∈ S) ∧
    S.card = 3 :=
by sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l62_6276


namespace NUMINAMATH_CALUDE_perimeter_of_specific_shape_l62_6266

/-- A shape with three sides of equal length -/
structure ThreeSidedShape where
  side_length : ℝ
  num_sides : ℕ
  h_num_sides : num_sides = 3

/-- The perimeter of a three-sided shape -/
def perimeter (shape : ThreeSidedShape) : ℝ :=
  shape.side_length * shape.num_sides

/-- Theorem: The perimeter of a shape with 3 sides, each of length 7 cm, is 21 cm -/
theorem perimeter_of_specific_shape :
  ∃ (shape : ThreeSidedShape), shape.side_length = 7 ∧ perimeter shape = 21 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_specific_shape_l62_6266


namespace NUMINAMATH_CALUDE_complex_angle_for_one_plus_i_sqrt_seven_l62_6297

theorem complex_angle_for_one_plus_i_sqrt_seven :
  let z : ℂ := 1 + Complex.I * Real.sqrt 7
  let r : ℝ := Complex.abs z
  let θ : ℝ := Complex.arg z
  θ = π / 8 := by sorry

end NUMINAMATH_CALUDE_complex_angle_for_one_plus_i_sqrt_seven_l62_6297


namespace NUMINAMATH_CALUDE_checkers_game_possibilities_l62_6215

/-- Represents the number of games played by each friend in a checkers game. -/
structure CheckersGames where
  friend1 : ℕ
  friend2 : ℕ
  friend3 : ℕ

/-- Checks if the given number of games for three friends is valid. -/
def isValidGameCount (games : CheckersGames) : Prop :=
  ∃ (a b c : ℕ), 
    a + b + c = (games.friend1 + games.friend2 + games.friend3) / 2 ∧
    a + c = games.friend1 ∧
    b + c = games.friend2 ∧
    a + b = games.friend3

/-- Theorem stating the validity of different game counts for the third friend. -/
theorem checkers_game_possibilities : 
  let games1 := CheckersGames.mk 25 17 34
  let games2 := CheckersGames.mk 25 17 35
  let games3 := CheckersGames.mk 25 17 56
  isValidGameCount games1 ∧ 
  ¬isValidGameCount games2 ∧ 
  ¬isValidGameCount games3 := by
  sorry

end NUMINAMATH_CALUDE_checkers_game_possibilities_l62_6215


namespace NUMINAMATH_CALUDE_kitten_weight_l62_6254

/-- Given the weights of a kitten and two dogs satisfying certain conditions,
    prove that the kitten weighs 6 pounds. -/
theorem kitten_weight (x y z : ℝ) 
  (h1 : x + y + z = 36)
  (h2 : x + z = 2*y)
  (h3 : x + y = z) :
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_kitten_weight_l62_6254


namespace NUMINAMATH_CALUDE_billy_video_watching_l62_6223

def total_time : ℕ := 90
def video_watch_time : ℕ := 4
def search_time : ℕ := 3
def break_time : ℕ := 5
def trial_count : ℕ := 5
def suggestions_per_trial : ℕ := 15
def additional_categories : ℕ := 2
def suggestions_per_category : ℕ := 10

def max_videos_watched : ℕ := 13

theorem billy_video_watching :
  let total_search_time := search_time * trial_count
  let total_break_time := break_time * (trial_count - 1)
  let available_watch_time := total_time - (total_search_time + total_break_time)
  max_videos_watched = available_watch_time / video_watch_time ∧
  max_videos_watched ≤ suggestions_per_trial * trial_count +
                       suggestions_per_category * additional_categories :=
by sorry

end NUMINAMATH_CALUDE_billy_video_watching_l62_6223


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l62_6239

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I) :
  z.im = (Real.sqrt 2 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l62_6239


namespace NUMINAMATH_CALUDE_arithmetic_progression_roots_l62_6295

/-- A polynomial of the form x^4 + px^2 + q has 4 real roots in arithmetic progression
    if and only if p ≤ 0 and q = 0.09p^2 -/
theorem arithmetic_progression_roots (p q : ℝ) :
  (∃ (a d : ℝ), ∀ (x : ℝ), x^4 + p*x^2 + q = 0 ↔ 
    x = a - 3*d ∨ x = a - d ∨ x = a + d ∨ x = a + 3*d) ↔ 
  (p ≤ 0 ∧ q = 0.09 * p^2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_roots_l62_6295


namespace NUMINAMATH_CALUDE_negation_of_existence_exp_l62_6274

theorem negation_of_existence_exp (p : Prop) : 
  (p ↔ ∃ x : ℝ, Real.exp x < 0) → 
  (¬p ↔ ∀ x : ℝ, Real.exp x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_exp_l62_6274


namespace NUMINAMATH_CALUDE_circumradius_range_l62_6279

/-- A square with side length 1 -/
structure UnitSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

/-- Points P and Q on side AB, and R on side CD of a unit square -/
structure TrianglePoints (square : UnitSquare) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  P_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t, 0)
  Q_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (t, 0)
  R_on_CD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (t, 1)

/-- The circumradius of a triangle -/
def circumradius (P Q R : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating the range of possible circumradius values -/
theorem circumradius_range (square : UnitSquare) (points : TrianglePoints square) :
  1/2 < circumradius points.P points.Q points.R ∧ 
  circumradius points.P points.Q points.R ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_range_l62_6279


namespace NUMINAMATH_CALUDE_power_mod_seven_l62_6241

theorem power_mod_seven : 2^19 % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seven_l62_6241


namespace NUMINAMATH_CALUDE_expression_value_l62_6238

theorem expression_value (m n : ℤ) (h : m - n = 2) : 2*m^2 - 4*m*n + 2*n^2 - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l62_6238


namespace NUMINAMATH_CALUDE_simplify_fraction_l62_6273

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l62_6273


namespace NUMINAMATH_CALUDE_rotateSemicircleDiameter_is_eight_l62_6232

/-- The diameter of a solid figure obtained by rotating a semicircle around its diameter -/
def rotateSemicircleDiameter (radius : ℝ) : ℝ :=
  2 * radius

/-- Theorem: The diameter of a solid figure obtained by rotating a semicircle 
    with a radius of 4 centimeters once around its diameter is 8 centimeters -/
theorem rotateSemicircleDiameter_is_eight :
  rotateSemicircleDiameter 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_rotateSemicircleDiameter_is_eight_l62_6232


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l62_6204

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(29 ∣ (87654321 - y))) ∧ 
  (29 ∣ (87654321 - x)) :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l62_6204


namespace NUMINAMATH_CALUDE_game_cost_l62_6278

theorem game_cost (initial_money : ℕ) (toys_count : ℕ) (toy_price : ℕ) (remaining_money : ℕ) :
  initial_money = 68 →
  toys_count = 3 →
  toy_price = 7 →
  remaining_money = toys_count * toy_price →
  initial_money - remaining_money = 47 :=
by
  sorry

end NUMINAMATH_CALUDE_game_cost_l62_6278


namespace NUMINAMATH_CALUDE_expected_fib_value_l62_6271

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the probability of getting tails on both coins
def p_both_tails : ℚ := 1 / 4

-- Define the probability of not getting tails on both coins
def p_not_both_tails : ℚ := 3 / 4

-- Define the expected value of Fₖ
def expected_fib : ℚ := 19 / 11

-- Theorem statement
theorem expected_fib_value :
  ∃ (S : ℕ → ℚ), 
    (∀ n, S n = p_both_tails * fib n + p_not_both_tails * S (n + 1)) ∧
    (S 0 = expected_fib) := by
  sorry

end NUMINAMATH_CALUDE_expected_fib_value_l62_6271


namespace NUMINAMATH_CALUDE_max_sum_reciprocals_l62_6234

theorem max_sum_reciprocals (p q r x y z : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hpqr : p + q + r = 2) (hxyz : x + y + z = 1) :
  1/(p+q) + 1/(p+r) + 1/(q+r) + 1/(x+y) + 1/(x+z) + 1/(y+z) ≤ 27/4 := by
sorry

end NUMINAMATH_CALUDE_max_sum_reciprocals_l62_6234


namespace NUMINAMATH_CALUDE_cylinder_symmetry_properties_l62_6261

/-- Represents the type of a rotational cylinder -/
inductive CylinderType
  | DoubleSidedBounded
  | SingleSidedBounded
  | DoubleSidedUnbounded

/-- Represents the symmetry properties of a cylinder -/
structure CylinderSymmetry where
  hasAxisSymmetry : Bool
  hasPerpendicularPlaneSymmetry : Bool
  hasBundlePlanesSymmetry : Bool
  hasCenterSymmetry : Bool
  hasInfiniteCentersSymmetry : Bool
  hasTwoSystemsPlanesSymmetry : Bool

/-- Returns the symmetry properties for a given cylinder type -/
def getSymmetryProperties (cType : CylinderType) : CylinderSymmetry :=
  match cType with
  | CylinderType.DoubleSidedBounded => {
      hasAxisSymmetry := true,
      hasPerpendicularPlaneSymmetry := true,
      hasBundlePlanesSymmetry := true,
      hasCenterSymmetry := true,
      hasInfiniteCentersSymmetry := false,
      hasTwoSystemsPlanesSymmetry := false
    }
  | CylinderType.SingleSidedBounded => {
      hasAxisSymmetry := true,
      hasPerpendicularPlaneSymmetry := false,
      hasBundlePlanesSymmetry := true,
      hasCenterSymmetry := false,
      hasInfiniteCentersSymmetry := false,
      hasTwoSystemsPlanesSymmetry := false
    }
  | CylinderType.DoubleSidedUnbounded => {
      hasAxisSymmetry := true,
      hasPerpendicularPlaneSymmetry := false,
      hasBundlePlanesSymmetry := false,
      hasCenterSymmetry := false,
      hasInfiniteCentersSymmetry := true,
      hasTwoSystemsPlanesSymmetry := true
    }

theorem cylinder_symmetry_properties (cType : CylinderType) :
  (getSymmetryProperties cType).hasAxisSymmetry = true ∧
  ((cType = CylinderType.DoubleSidedBounded) → (getSymmetryProperties cType).hasPerpendicularPlaneSymmetry = true) ∧
  ((cType = CylinderType.DoubleSidedBounded ∨ cType = CylinderType.SingleSidedBounded) → (getSymmetryProperties cType).hasBundlePlanesSymmetry = true) ∧
  ((cType = CylinderType.DoubleSidedBounded) → (getSymmetryProperties cType).hasCenterSymmetry = true) ∧
  ((cType = CylinderType.DoubleSidedUnbounded) → (getSymmetryProperties cType).hasInfiniteCentersSymmetry = true) ∧
  ((cType = CylinderType.DoubleSidedUnbounded) → (getSymmetryProperties cType).hasTwoSystemsPlanesSymmetry = true) :=
by
  sorry


end NUMINAMATH_CALUDE_cylinder_symmetry_properties_l62_6261


namespace NUMINAMATH_CALUDE_number_puzzle_l62_6244

theorem number_puzzle : ∃ x : ℤ, (x + 2) - 3 = 7 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l62_6244


namespace NUMINAMATH_CALUDE_simple_interest_problem_l62_6211

/-- Calculates the total amount after a given period using simple interest -/
def totalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given the conditions, prove that the total amount after 7 years is $595 -/
theorem simple_interest_problem :
  ∃ (rate : ℝ),
    (totalAmount 350 rate 2 = 420) →
    (totalAmount 350 rate 7 = 595) := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l62_6211


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l62_6202

theorem complex_magnitude_equality (t : ℝ) : 
  t > 0 → (Complex.abs (-5 + t * Complex.I) = 3 * Real.sqrt 6 ↔ t = Real.sqrt 29) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l62_6202


namespace NUMINAMATH_CALUDE_carp_classification_l62_6243

-- Define the characteristics of an individual
structure IndividualCharacteristics where
  birth : Bool
  death : Bool
  gender : Bool
  age : ℕ

-- Define the characteristics of a population
structure PopulationCharacteristics where
  birthRate : ℝ
  deathRate : ℝ
  genderRatio : ℝ
  ageComposition : List ℝ

-- Define the types
inductive EntityType
  | Carp
  | CarpPopulation

-- Define the main theorem
theorem carp_classification 
  (a : IndividualCharacteristics) 
  (b : PopulationCharacteristics) : 
  (EntityType.Carp, EntityType.CarpPopulation) = 
  (
    match a with
    | { birth := _, death := _, gender := _, age := _ } => EntityType.Carp,
    match b with
    | { birthRate := _, deathRate := _, genderRatio := _, ageComposition := _ } => EntityType.CarpPopulation
  ) := by
  sorry

end NUMINAMATH_CALUDE_carp_classification_l62_6243


namespace NUMINAMATH_CALUDE_hiking_distance_proof_l62_6226

theorem hiking_distance_proof (total_distance car_to_stream stream_to_meadow : ℝ) 
  (h1 : total_distance = 0.7)
  (h2 : car_to_stream = 0.2)
  (h3 : stream_to_meadow = 0.4) :
  total_distance - (car_to_stream + stream_to_meadow) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_hiking_distance_proof_l62_6226


namespace NUMINAMATH_CALUDE_largest_number_hcf_lcm_l62_6255

theorem largest_number_hcf_lcm (a b : ℕ+) : 
  (Nat.gcd a b = 52) → 
  (Nat.lcm a b = 52 * 11 * 12) → 
  (max a b = 624) := by
sorry

end NUMINAMATH_CALUDE_largest_number_hcf_lcm_l62_6255


namespace NUMINAMATH_CALUDE_unique_solution_condition_l62_6292

/-- The equation (3x+7)(x-5) = -27 + kx has exactly one real solution if and only if k = -8 + 4√6 or k = -8 - 4√6 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x+7)*(x-5) = -27 + k*x) ↔ 
  (k = -8 + 4*Real.sqrt 6 ∨ k = -8 - 4*Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l62_6292


namespace NUMINAMATH_CALUDE_shared_vertex_angle_is_84_l62_6216

/-- The angle between an edge of an equilateral triangle and an edge of a regular pentagon,
    when both shapes are inscribed in a circle and share a common vertex. -/
def shared_vertex_angle : ℝ := 84

/-- An equilateral triangle inscribed in a circle -/
structure EquilateralTriangleInCircle :=
  (vertices : Fin 3 → ℝ × ℝ)
  (is_equilateral : ∀ i j : Fin 3, i ≠ j → dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))
  (on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i : Fin 3, dist (vertices i) center = radius)

/-- A regular pentagon inscribed in a circle -/
structure RegularPentagonInCircle :=
  (vertices : Fin 5 → ℝ × ℝ)
  (is_regular : ∀ i j : Fin 5, dist (vertices i) (vertices ((i + 1) % 5)) = dist (vertices j) (vertices ((j + 1) % 5)))
  (on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i : Fin 5, dist (vertices i) center = radius)

theorem shared_vertex_angle_is_84 
  (triangle : EquilateralTriangleInCircle) 
  (pentagon : RegularPentagonInCircle) 
  (shared_vertex : ∃ i j, triangle.vertices i = pentagon.vertices j) :
  shared_vertex_angle = 84 := by
  sorry

end NUMINAMATH_CALUDE_shared_vertex_angle_is_84_l62_6216


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l62_6280

theorem quadratic_distinct_roots (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*(m-2)*x + m^2 = 0 → (∃ y : ℝ, x ≠ y ∧ y^2 - 2*(m-2)*y + m^2 = 0)) →
  m < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l62_6280


namespace NUMINAMATH_CALUDE_projection_v_onto_w_l62_6212

def v : Fin 2 → ℝ := ![3, -1]
def w : Fin 2 → ℝ := ![4, 2]

theorem projection_v_onto_w :
  (((v • w) / (w • w)) • w) = ![2, 1] := by sorry

end NUMINAMATH_CALUDE_projection_v_onto_w_l62_6212


namespace NUMINAMATH_CALUDE_bob_sandwich_options_l62_6286

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 6

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 4

/-- Represents whether Bob orders sandwiches with turkey and Swiss cheese. -/
def orders_turkey_swiss : Bool := false

/-- Represents whether Bob orders sandwiches with multigrain bread and turkey. -/
def orders_multigrain_turkey : Bool := false

/-- Calculates the number of sandwiches Bob can order. -/
def num_bob_sandwiches : ℕ := 
  num_breads * num_meats * num_cheeses - 
  (if orders_turkey_swiss then 0 else num_breads) - 
  (if orders_multigrain_turkey then 0 else num_cheeses)

/-- Theorem stating the number of different sandwiches Bob could order. -/
theorem bob_sandwich_options : num_bob_sandwiches = 111 := by
  sorry

end NUMINAMATH_CALUDE_bob_sandwich_options_l62_6286


namespace NUMINAMATH_CALUDE_range_of_m_l62_6208

def p (x : ℝ) : Prop := abs (2 * x + 1) ≤ 3

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m : 
  ∀ m : ℝ, (m > 0 ∧ 
    (∀ x : ℝ, p x → q x m) ∧ 
    (∃ x : ℝ, ¬(p x) ∧ q x m)) ↔ 
  m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l62_6208


namespace NUMINAMATH_CALUDE_domain_of_f_l62_6217

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem domain_of_f (f : ℝ → ℝ) :
  (∀ x, f (2 * x - 3) ≠ 0 → -2 ≤ x ∧ x ≤ 2) →
  (∀ y, f y ≠ 0 → -7 ≤ y ∧ y ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l62_6217


namespace NUMINAMATH_CALUDE_smallest_third_term_l62_6293

/-- An arithmetic sequence of five positive integers with sum 80 -/
structure ArithmeticSequence where
  a : ℕ+  -- first term
  d : ℕ+  -- common difference
  sum_eq_80 : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 80

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℕ := seq.a + 2*seq.d

/-- Theorem stating that the smallest possible third term is 16 -/
theorem smallest_third_term :
  ∀ seq : ArithmeticSequence, third_term seq ≥ 16 := by
  sorry

#check smallest_third_term

end NUMINAMATH_CALUDE_smallest_third_term_l62_6293


namespace NUMINAMATH_CALUDE_distance_between_points_l62_6240

-- Define the initial meeting time in hours
def initial_meeting_time : ℝ := 4

-- Define the new meeting time in hours after speed increase
def new_meeting_time : ℝ := 3.5

-- Define the speed increase in km/h
def speed_increase : ℝ := 3

-- Define the function to calculate the distance
def calculate_distance (v_A v_B : ℝ) : ℝ := initial_meeting_time * (v_A + v_B)

-- Theorem statement
theorem distance_between_points : 
  ∃ (v_A v_B : ℝ), 
    v_A > 0 ∧ v_B > 0 ∧
    calculate_distance v_A v_B = 
    new_meeting_time * ((v_A + speed_increase) + (v_B + speed_increase)) ∧
    calculate_distance v_A v_B = 168 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l62_6240


namespace NUMINAMATH_CALUDE_f_is_quadratic_l62_6201

/-- Definition of a quadratic equation with one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 4x - x² -/
def f (x : ℝ) : ℝ := 4 * x - x^2

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l62_6201


namespace NUMINAMATH_CALUDE_jeremy_stroll_distance_l62_6231

theorem jeremy_stroll_distance (speed : ℝ) (time : ℝ) (h1 : speed = 2) (h2 : time = 10) :
  speed * time = 20 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_stroll_distance_l62_6231


namespace NUMINAMATH_CALUDE_second_month_sale_l62_6275

def average_sale : ℕ := 6600
def num_months : ℕ := 6
def sale_month1 : ℕ := 6435
def sale_month3 : ℕ := 7230
def sale_month4 : ℕ := 6562
def sale_month5 : ℕ := 6855
def sale_month6 : ℕ := 5591

theorem second_month_sale :
  ∃ (sale_month2 : ℕ),
    sale_month2 = average_sale * num_months - (sale_month1 + sale_month3 + sale_month4 + sale_month5 + sale_month6) ∧
    sale_month2 = 6927 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l62_6275


namespace NUMINAMATH_CALUDE_mappings_count_l62_6247

/-- Set A with elements from 1 to 15 -/
def A : Finset ℕ := Finset.range 15

/-- Set B with elements 0 and 1 -/
def B : Finset ℕ := {0, 1}

/-- The number of mappings from A to B where 1 is the image of at least two elements of A -/
def num_mappings : ℕ := 2^15 - (1 + 15)

/-- Theorem stating that the number of mappings from A to B where 1 is the image of at least two elements of A is 32752 -/
theorem mappings_count : num_mappings = 32752 := by
  sorry

#eval num_mappings

end NUMINAMATH_CALUDE_mappings_count_l62_6247

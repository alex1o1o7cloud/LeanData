import Mathlib

namespace NUMINAMATH_CALUDE_solve_daily_wage_l1392_139245

def daily_wage_problem (a b c : ℕ) : Prop :=
  -- Define the ratio of daily wages
  a * 4 = b * 3 ∧ b * 5 = c * 4 ∧
  -- Define the total earnings
  6 * a + 9 * b + 4 * c = 1406 ∧
  -- The daily wage of c is 95
  c = 95

theorem solve_daily_wage : ∃ a b c : ℕ, daily_wage_problem a b c :=
sorry

end NUMINAMATH_CALUDE_solve_daily_wage_l1392_139245


namespace NUMINAMATH_CALUDE_pink_highlighters_count_l1392_139246

def total_highlighters : ℕ := 33
def yellow_highlighters : ℕ := 15
def blue_highlighters : ℕ := 8

theorem pink_highlighters_count :
  total_highlighters - yellow_highlighters - blue_highlighters = 10 := by
  sorry

end NUMINAMATH_CALUDE_pink_highlighters_count_l1392_139246


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1392_139276

theorem smallest_n_congruence (n : ℕ) : 
  (23 * n ≡ 310 [ZMOD 9]) ∧ (∀ m : ℕ, m < n → ¬(23 * m ≡ 310 [ZMOD 9])) ↔ n = 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1392_139276


namespace NUMINAMATH_CALUDE_parabola_ellipse_tangent_property_l1392_139264

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the parabola E
def parabola (x y : ℝ) : Prop := y^2 = 6*x + 15

-- Define the focus F
def F : ℝ × ℝ := (-1, 0)

-- Define a point on the parabola
def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

-- Define tangent points on the ellipse
def tangent_points (M N : ℝ × ℝ) : Prop := ellipse M.1 M.2 ∧ ellipse N.1 N.2

-- Theorem statement
theorem parabola_ellipse_tangent_property
  (A M N : ℝ × ℝ)
  (h_A : on_parabola A)
  (h_MN : tangent_points M N) :
  (∃ (t : ℝ), F.1 + t * (A.1 - F.1) = (M.1 + N.1) / 2 ∧
              F.2 + t * (A.2 - F.2) = (M.2 + N.2) / 2) ∧
  (∃ (θ : ℝ), θ = 2 * Real.pi / 3 ∧
    Real.cos θ = ((M.1 + 1) * (N.1 + 1) + M.2 * N.2) /
                 (Real.sqrt ((M.1 + 1)^2 + M.2^2) * Real.sqrt ((N.1 + 1)^2 + N.2^2))) :=
sorry

end NUMINAMATH_CALUDE_parabola_ellipse_tangent_property_l1392_139264


namespace NUMINAMATH_CALUDE_equidistant_point_x_value_l1392_139204

/-- A point in a 2D rectangular coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to the x-axis -/
def distToXAxis (p : Point) : ℝ := |p.y|

/-- Distance from a point to the y-axis -/
def distToYAxis (p : Point) : ℝ := |p.x|

/-- A point is equidistant from x-axis and y-axis -/
def isEquidistant (p : Point) : Prop :=
  distToXAxis p = distToYAxis p

/-- The main theorem -/
theorem equidistant_point_x_value (x : ℝ) :
  let p := Point.mk (-2*x) (x-6)
  isEquidistant p → x = 2 ∨ x = -6 := by
  sorry


end NUMINAMATH_CALUDE_equidistant_point_x_value_l1392_139204


namespace NUMINAMATH_CALUDE_bacteria_growth_l1392_139236

theorem bacteria_growth (n : ℕ) : 
  (∀ t : ℕ, t ≤ 10 → n * (4 ^ t) = n * 4 ^ t) →
  n * 4 ^ 10 = 1048576 ↔ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_l1392_139236


namespace NUMINAMATH_CALUDE_factorial_power_of_two_l1392_139291

theorem factorial_power_of_two (k : ℕ) :
  ∀ n m : ℕ, (2^k).factorial = 2^n * m ↔
  ∃ t : ℕ, n = 2^k - 1 - t ∧ m = (2^k).factorial / 2^(2^k - 1 - t) := by
  sorry

end NUMINAMATH_CALUDE_factorial_power_of_two_l1392_139291


namespace NUMINAMATH_CALUDE_prop_2_prop_4_l1392_139215

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Theorem for proposition ②
theorem prop_2 (l : Line) (α β : Plane) :
  perpendicular l α → parallel l β → plane_perpendicular α β := by sorry

-- Theorem for proposition ④
theorem prop_4 (α β γ : Plane) :
  plane_perpendicular α β → plane_parallel α γ → plane_perpendicular γ β := by sorry

end NUMINAMATH_CALUDE_prop_2_prop_4_l1392_139215


namespace NUMINAMATH_CALUDE_simplify_sqrt_18_l1392_139255

theorem simplify_sqrt_18 : Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_18_l1392_139255


namespace NUMINAMATH_CALUDE_min_calls_proof_l1392_139280

/-- Represents the minimum number of calls per month -/
def min_calls : ℕ := 66

/-- Represents the monthly rental fee in yuan -/
def rental_fee : ℚ := 12

/-- Represents the cost per call in yuan -/
def cost_per_call : ℚ := (1/5 : ℚ)

/-- Represents the minimum monthly phone bill in yuan -/
def min_monthly_bill : ℚ := 25

theorem min_calls_proof :
  (min_calls : ℚ) * cost_per_call + rental_fee > min_monthly_bill ∧
  ∀ n : ℕ, n < min_calls → (n : ℚ) * cost_per_call + rental_fee ≤ min_monthly_bill :=
by sorry

end NUMINAMATH_CALUDE_min_calls_proof_l1392_139280


namespace NUMINAMATH_CALUDE_skew_lines_definition_l1392_139294

-- Define a type for lines in 3D space
def Line3D : Type := ℝ × ℝ × ℝ → Prop

-- Define the property of two lines being parallel
def parallel (l1 l2 : Line3D) : Prop := sorry

-- Define the property of two lines intersecting
def intersect (l1 l2 : Line3D) : Prop := sorry

-- Define skew lines
def skew (l1 l2 : Line3D) : Prop :=
  ¬(parallel l1 l2) ∧ ¬(intersect l1 l2)

-- Theorem stating the definition of skew lines
theorem skew_lines_definition (l1 l2 : Line3D) :
  skew l1 l2 ↔ (¬(parallel l1 l2) ∧ ¬(intersect l1 l2)) := by sorry

end NUMINAMATH_CALUDE_skew_lines_definition_l1392_139294


namespace NUMINAMATH_CALUDE_total_cost_is_246_l1392_139201

/-- Represents a person's balloon collection --/
structure BalloonCollection where
  yellowCount : Nat
  yellowPrice : Nat
  redCount : Nat
  redPrice : Nat

/-- Calculates the total cost of a balloon collection --/
def totalCost (bc : BalloonCollection) : Nat :=
  bc.yellowCount * bc.yellowPrice + bc.redCount * bc.redPrice

/-- The balloon collections for each person --/
def fred : BalloonCollection := ⟨5, 3, 3, 4⟩
def sam : BalloonCollection := ⟨6, 4, 4, 5⟩
def mary : BalloonCollection := ⟨7, 5, 5, 6⟩
def susan : BalloonCollection := ⟨4, 6, 6, 7⟩
def tom : BalloonCollection := ⟨10, 2, 8, 3⟩

/-- Theorem: The total cost of all balloon collections is $246 --/
theorem total_cost_is_246 :
  totalCost fred + totalCost sam + totalCost mary + totalCost susan + totalCost tom = 246 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_246_l1392_139201


namespace NUMINAMATH_CALUDE_deborah_mail_count_l1392_139266

theorem deborah_mail_count :
  let standard_postage : ℚ := 108 / 100
  let international_surcharge : ℚ := 14 / 100
  let international_letters : ℕ := 2
  let total_cost : ℚ := 460 / 100
  ∃ domestic_letters : ℕ,
    (domestic_letters : ℚ) * standard_postage + 
    (international_letters : ℚ) * (standard_postage + international_surcharge) = total_cost ∧
    domestic_letters + international_letters = 4 :=
by sorry

end NUMINAMATH_CALUDE_deborah_mail_count_l1392_139266


namespace NUMINAMATH_CALUDE_line_perpendicular_condition_l1392_139234

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)
variable (linePerpPlane : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_condition
  (a b : Line) (α β : Plane)
  (h1 : lineInPlane a α)
  (h2 : linePerpPlane b β)
  (h3 : parallel α β) :
  perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_condition_l1392_139234


namespace NUMINAMATH_CALUDE_probability_blue_after_red_l1392_139238

/-- Probability of picking a blue marble after removing a red one --/
theorem probability_blue_after_red (total : ℕ) (yellow : ℕ) (green : ℕ) (red : ℕ) (blue : ℕ) :
  total = 120 →
  yellow = 30 →
  green = yellow / 3 →
  red = 2 * green →
  blue = total - yellow - green - red →
  (blue : ℚ) / (total - 1 : ℚ) = 60 / 119 := by
  sorry

end NUMINAMATH_CALUDE_probability_blue_after_red_l1392_139238


namespace NUMINAMATH_CALUDE_train_passing_time_l1392_139212

/-- The time it takes for a train to pass a man running in the opposite direction -/
theorem train_passing_time (train_length : Real) (train_speed : Real) (man_speed : Real) :
  train_length = 250 ∧ 
  train_speed = 120 * 1000 / 3600 ∧ 
  man_speed = 10 * 1000 / 3600 →
  ∃ t : Real, t < 30 ∧ t * (train_speed + man_speed) = train_length :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l1392_139212


namespace NUMINAMATH_CALUDE_sqrt_fifth_power_cubed_l1392_139214

theorem sqrt_fifth_power_cubed : (Real.sqrt ((Real.sqrt 5) ^ 4)) ^ 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifth_power_cubed_l1392_139214


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_33_l1392_139247

theorem modular_inverse_of_5_mod_33 : ∃ x : ℕ, x ≥ 0 ∧ x ≤ 32 ∧ (5 * x) % 33 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_33_l1392_139247


namespace NUMINAMATH_CALUDE_geometric_sequence_n_l1392_139233

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

theorem geometric_sequence_n (a₁ q : ℝ) (n : ℕ) :
  a₁ = 1 → q = 2 → geometric_sequence a₁ q n = 64 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_n_l1392_139233


namespace NUMINAMATH_CALUDE_unknown_room_width_is_15_l1392_139290

-- Define the room dimensions
def room_length : ℝ := 25
def room_height : ℝ := 12
def door_area : ℝ := 6 * 3
def window_area : ℝ := 4 * 3
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 10
def total_cost : ℝ := 9060

-- Define the function to calculate the total area to be whitewashed
def area_to_whitewash (x : ℝ) : ℝ :=
  2 * (room_length + x) * room_height - (door_area + num_windows * window_area)

-- Define the theorem
theorem unknown_room_width_is_15 :
  ∃ x : ℝ, x > 0 ∧ cost_per_sqft * area_to_whitewash x = total_cost ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_unknown_room_width_is_15_l1392_139290


namespace NUMINAMATH_CALUDE_perfect_square_property_l1392_139297

theorem perfect_square_property (x y p : ℕ+) (hp : Nat.Prime p.val) 
  (h : 4 * x.val^2 + 8 * y.val^2 + (2 * x.val - 3 * y.val) * p.val - 12 * x.val * y.val = 0) :
  ∃ (n : ℕ), 4 * y.val + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_property_l1392_139297


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1392_139289

theorem contrapositive_equivalence (a : ℝ) : 
  (¬(a > 1) → ¬(a > 0)) ↔ (a ≤ 1 → a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1392_139289


namespace NUMINAMATH_CALUDE_toy_cost_price_l1392_139248

theorem toy_cost_price (profit_equality : 30 * (12 - C) = 20 * (15 - C)) : C = 6 :=
by sorry

end NUMINAMATH_CALUDE_toy_cost_price_l1392_139248


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1392_139213

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) ↔ (a < 1 ∨ a > 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1392_139213


namespace NUMINAMATH_CALUDE_dog_weight_multiple_l1392_139263

/-- Given the weights of three dogs (chihuahua, pitbull, and great dane), 
    prove that the great dane's weight is 3 times the pitbull's weight plus 10 pounds. -/
theorem dog_weight_multiple (c p g : ℝ) 
  (h1 : c + p + g = 439)  -- Total weight
  (h2 : p = 3 * c)        -- Pitbull's weight relation to chihuahua
  (h3 : g = 307)          -- Great dane's weight
  : ∃ m : ℝ, g = m * p + 10 ∧ m = 3 := by
  sorry

#check dog_weight_multiple

end NUMINAMATH_CALUDE_dog_weight_multiple_l1392_139263


namespace NUMINAMATH_CALUDE_median_bisects_perimeter_implies_isosceles_l1392_139203

/-- A triangle is represented by its three side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- A median of a triangle -/
structure Median (t : Triangle) where
  base : ℝ
  is_median : base = t.a ∨ base = t.b ∨ base = t.c

/-- A triangle is isosceles if at least two of its sides are equal -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- The theorem statement -/
theorem median_bisects_perimeter_implies_isosceles (t : Triangle) (m : Median t) :
  (m.base / 2 + (t.perimeter - m.base) / 2 = t.perimeter / 2) → t.isIsosceles :=
by
  sorry

end NUMINAMATH_CALUDE_median_bisects_perimeter_implies_isosceles_l1392_139203


namespace NUMINAMATH_CALUDE_sum_mean_median_mode_l1392_139200

def numbers : List ℝ := [-3, -1, 0, 2, 2, 3, 3, 3, 4, 5]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

theorem sum_mean_median_mode :
  mean numbers + median numbers + mode numbers = 7.3 := by sorry

end NUMINAMATH_CALUDE_sum_mean_median_mode_l1392_139200


namespace NUMINAMATH_CALUDE_prob_more_than_4_draws_is_31_35_l1392_139235

-- Define the number of new and old coins
def new_coins : ℕ := 3
def old_coins : ℕ := 4
def total_coins : ℕ := new_coins + old_coins

-- Define the probability function
noncomputable def prob_more_than_4_draws : ℚ :=
  1 - (
    -- Probability of drawing all new coins in first 3 draws
    (new_coins / total_coins) * ((new_coins - 1) / (total_coins - 1)) * ((new_coins - 2) / (total_coins - 2)) +
    -- Probability of drawing all new coins in first 4 draws (but not in first 3)
    3 * ((old_coins / total_coins) * (new_coins / (total_coins - 1)) * ((new_coins - 1) / (total_coins - 2)) * ((new_coins - 2) / (total_coins - 3)))
  )

-- Theorem statement
theorem prob_more_than_4_draws_is_31_35 : prob_more_than_4_draws = 31 / 35 :=
  sorry

end NUMINAMATH_CALUDE_prob_more_than_4_draws_is_31_35_l1392_139235


namespace NUMINAMATH_CALUDE_solution_set_and_range_l1392_139274

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

theorem solution_set_and_range :
  (∀ x : ℝ, f x ≥ 1 ↔ x ≤ -5/2 ∨ x ≥ 3/2) ∧
  (∀ a : ℝ, (∀ x : ℝ, f x ≥ a^2 - a - 2) ↔ -1 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l1392_139274


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l1392_139257

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (sum_squares : a^2 + b^2 + c^2 = 5)
  (sum_cubes : a^3 + b^3 + c^3 = 10) :
  a^4 + b^4 + c^4 = 68/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l1392_139257


namespace NUMINAMATH_CALUDE_equation_root_one_l1392_139249

theorem equation_root_one (k : ℝ) : 
  let a : ℝ := 13 / 2
  let b : ℝ := -4
  ∃ x : ℝ, x = 1 ∧ (2 * k * x + a) / 3 = 2 + (x - b * k) / 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_root_one_l1392_139249


namespace NUMINAMATH_CALUDE_expression_value_l1392_139223

theorem expression_value : 
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 1
  x^2 + y^2 - 2*z^2 + 3*x*y = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1392_139223


namespace NUMINAMATH_CALUDE_similar_triangles_leg_l1392_139261

/-- Two similar right triangles with legs 10 and 8 in the first triangle,
    and x and 5 in the second triangle. -/
structure SimilarRightTriangles where
  x : ℝ
  similarity : (10 : ℝ) / x = 8 / 5

theorem similar_triangles_leg (t : SimilarRightTriangles) : t.x = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_l1392_139261


namespace NUMINAMATH_CALUDE_prob_one_black_one_red_is_three_fifths_l1392_139281

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | Black

/-- Represents a ball with a color and number -/
structure Ball where
  color : BallColor
  number : Nat

/-- The bag of balls -/
def bag : Finset Ball := sorry

/-- The number of red balls in the bag -/
def num_red_balls : Nat := sorry

/-- The number of black balls in the bag -/
def num_black_balls : Nat := sorry

/-- The total number of balls in the bag -/
def total_balls : Nat := sorry

/-- The probability of drawing one black ball and one red ball in the first two draws -/
def prob_one_black_one_red : ℚ := sorry

/-- Theorem stating the probability of drawing one black ball and one red ball in the first two draws -/
theorem prob_one_black_one_red_is_three_fifths :
  prob_one_black_one_red = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_one_black_one_red_is_three_fifths_l1392_139281


namespace NUMINAMATH_CALUDE_triangle_formation_l1392_139227

theorem triangle_formation (a b c : ℝ) : 
  a = 4 ∧ b = 6 ∧ c = 9 →
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l1392_139227


namespace NUMINAMATH_CALUDE_largest_difference_is_209_l1392_139228

/-- A type representing a 20 × 20 square table filled with distinct natural numbers from 1 to 400. -/
def Table := Fin 20 → Fin 20 → Fin 400

/-- The property that all numbers in the table are distinct. -/
def all_distinct (t : Table) : Prop :=
  ∀ i j k l, t i j = t k l → (i = k ∧ j = l)

/-- The property that there exist two numbers in the same row or column with a difference of at least N. -/
def has_difference_at_least (t : Table) (N : ℕ) : Prop :=
  ∃ i j k, (j = k ∧ |t i j - t i k| ≥ N) ∨ (i = k ∧ |t i j - t k j| ≥ N)

/-- The main theorem stating that 209 is the largest value satisfying the condition. -/
theorem largest_difference_is_209 :
  (∀ t : Table, all_distinct t → has_difference_at_least t 209) ∧
  ¬(∀ t : Table, all_distinct t → has_difference_at_least t 210) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_is_209_l1392_139228


namespace NUMINAMATH_CALUDE_sum_of_f_powers_equals_510_l1392_139210

def f (x : ℚ) : ℚ := (1 + 10 * x) / (10 - 100 * x)

def f_power (n : ℕ) : ℚ → ℚ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ (f_power n)

theorem sum_of_f_powers_equals_510 :
  (Finset.range 6000).sum (λ n => f_power (n + 1) (1/2)) = 510 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_powers_equals_510_l1392_139210


namespace NUMINAMATH_CALUDE_day_150_of_year_N_minus_1_is_friday_l1392_139283

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  day : Nat

/-- Function to determine the day of the week for a given YearDay -/
def dayOfWeek (yd : YearDay) : DayOfWeek := sorry

/-- Theorem stating the problem conditions and the result to be proved -/
theorem day_150_of_year_N_minus_1_is_friday 
  (N : Int) 
  (h1 : dayOfWeek ⟨N, 250⟩ = DayOfWeek.Monday) 
  (h2 : dayOfWeek ⟨N + 2, 300⟩ = DayOfWeek.Monday) :
  dayOfWeek ⟨N - 1, 150⟩ = DayOfWeek.Friday := by sorry

end NUMINAMATH_CALUDE_day_150_of_year_N_minus_1_is_friday_l1392_139283


namespace NUMINAMATH_CALUDE_tunnel_length_l1392_139285

/-- Calculates the length of a tunnel given train and journey parameters -/
theorem tunnel_length (train_length : ℝ) (exit_time : ℝ) (train_speed : ℝ) :
  train_length = 2 →
  exit_time = 5 / 60 →
  train_speed = 40 →
  (train_speed * exit_time) - train_length = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_tunnel_length_l1392_139285


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1392_139258

theorem sum_of_reciprocals_of_roots (p₁ p₂ : ℝ) : 
  p₁^2 - 17*p₁ + 8 = 0 → 
  p₂^2 - 17*p₂ + 8 = 0 → 
  p₁ ≠ p₂ →
  1/p₁ + 1/p₂ = 17/8 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1392_139258


namespace NUMINAMATH_CALUDE_count_with_zero_3017_l1392_139278

/-- A function that counts the number of integers from 1 to n that contain at least one digit '0' in their base-ten representation. -/
def count_with_zero (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the count of positive integers less than or equal to 3017 that contain at least one digit '0' in their base-ten representation is 740. -/
theorem count_with_zero_3017 : count_with_zero 3017 = 740 :=
  sorry

end NUMINAMATH_CALUDE_count_with_zero_3017_l1392_139278


namespace NUMINAMATH_CALUDE_chess_tournament_principled_trios_l1392_139239

/-- Represents the number of chess players in the tournament -/
def n : ℕ := 2017

/-- Defines a principled trio of chess players -/
def PrincipledTrio (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ A ≤ n ∧ B ≤ n ∧ C ≤ n

/-- Calculates the maximum number of principled trios for an odd number of players -/
def max_principled_trios_odd (k : ℕ) : ℕ := (k^3 - k) / 24

/-- The maximum number of principled trios in the tournament -/
def max_principled_trios : ℕ := max_principled_trios_odd n

theorem chess_tournament_principled_trios :
  max_principled_trios = 341606288 :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_principled_trios_l1392_139239


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1392_139254

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem union_of_A_and_B : A ∪ B = {x | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1392_139254


namespace NUMINAMATH_CALUDE_cos_15_degrees_l1392_139225

theorem cos_15_degrees : Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_15_degrees_l1392_139225


namespace NUMINAMATH_CALUDE_total_toys_is_160_l1392_139286

/-- The number of toys Kamari has -/
def kamari_toys : ℕ := 65

/-- The number of additional toys Anais has compared to Kamari -/
def anais_extra_toys : ℕ := 30

/-- The total number of toys Anais and Kamari have together -/
def total_toys : ℕ := kamari_toys + (kamari_toys + anais_extra_toys)

/-- Theorem stating that the total number of toys is 160 -/
theorem total_toys_is_160 : total_toys = 160 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_is_160_l1392_139286


namespace NUMINAMATH_CALUDE_initial_deck_size_l1392_139269

theorem initial_deck_size (red_cards : ℕ) (black_cards : ℕ) : 
  (red_cards : ℚ) / (red_cards + black_cards) = 1/3 →
  (red_cards : ℚ) / (red_cards + black_cards + 4) = 1/4 →
  red_cards + black_cards = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_deck_size_l1392_139269


namespace NUMINAMATH_CALUDE_increasing_quadratic_iff_l1392_139231

/-- A function f is increasing on an interval [x0, +∞) if for all x, y in the interval with x < y, f(x) < f(y) -/
def IncreasingOn (f : ℝ → ℝ) (x0 : ℝ) : Prop :=
  ∀ x y, x0 ≤ x → x < y → f x < f y

/-- The quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem increasing_quadratic_iff (a : ℝ) :
  IncreasingOn (f a) 4 ↔ a ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_increasing_quadratic_iff_l1392_139231


namespace NUMINAMATH_CALUDE_max_xy_value_l1392_139271

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 2 * y = 110) : x * y ≤ 216 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l1392_139271


namespace NUMINAMATH_CALUDE_closest_integer_to_double_sum_l1392_139229

/-- The number of distinct prime divisors of n that are at least k -/
def mho (n k : ℕ+) : ℕ := sorry

/-- The double sum in the problem -/
noncomputable def doubleSum : ℝ := sorry

theorem closest_integer_to_double_sum : 
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 1/2 ∧ doubleSum = 167 + ε := by sorry

end NUMINAMATH_CALUDE_closest_integer_to_double_sum_l1392_139229


namespace NUMINAMATH_CALUDE_range_of_a_l1392_139275

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x * Real.exp x else a * x^2 - 2 * x

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, y ≥ -(1 / Real.exp 1) → ∃ x : ℝ, f a x = y) →
  (∀ x : ℝ, f a x ≥ -(1 / Real.exp 1)) →
  a ≥ Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1392_139275


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1392_139240

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the distance from its right vertex to one of its asymptotes is b/2,
    then its eccentricity is 2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_distance : (a * b) / Real.sqrt (a^2 + b^2) = b / 2) : 
  (Real.sqrt (a^2 + b^2)) / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1392_139240


namespace NUMINAMATH_CALUDE_wrong_observation_value_l1392_139242

theorem wrong_observation_value (n : ℕ) (original_mean corrected_mean correct_value : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 41)
  (h3 : corrected_mean = 41.5)
  (h4 : correct_value = 48) :
  let wrong_value := n * original_mean - (n * corrected_mean - correct_value)
  wrong_value = 23 := by sorry

end NUMINAMATH_CALUDE_wrong_observation_value_l1392_139242


namespace NUMINAMATH_CALUDE_milk_fraction_after_pouring_l1392_139295

/-- Represents a cup containing a mixture of tea and milk -/
structure Cup where
  tea : ℚ
  milk : ℚ

/-- The pouring process described in the problem -/
def pour_process (initial_tea_cup : Cup) (initial_milk_cup : Cup) : Cup :=
  let first_pour := Cup.mk (initial_tea_cup.tea - 2) initial_milk_cup.milk
  let second_cup_total := initial_milk_cup.tea + initial_milk_cup.milk + 2
  let milk_ratio := initial_milk_cup.milk / second_cup_total
  let tea_ratio := (initial_milk_cup.tea + 2) / second_cup_total
  Cup.mk (first_pour.tea + 2 * tea_ratio) (first_pour.milk + 2 * milk_ratio)

theorem milk_fraction_after_pouring 
  (initial_tea_cup : Cup) 
  (initial_milk_cup : Cup) 
  (h1 : initial_tea_cup.tea = 6) 
  (h2 : initial_tea_cup.milk = 0) 
  (h3 : initial_milk_cup.tea = 0) 
  (h4 : initial_milk_cup.milk = 6) :
  let final_cup := pour_process initial_tea_cup initial_milk_cup
  (final_cup.milk / (final_cup.tea + final_cup.milk)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_milk_fraction_after_pouring_l1392_139295


namespace NUMINAMATH_CALUDE_polynomial_intersection_l1392_139224

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem polynomial_intersection (a b c d : ℝ) : 
  -- f and g are distinct polynomials
  (∃ x, f a b x ≠ g c d x) →
  -- The x-coordinate of the vertex of f is a root of g
  g c d (-a/2) = 0 →
  -- The x-coordinate of the vertex of g is a root of f
  f a b (-c/2) = 0 →
  -- Both f and g have the same minimum value
  (f a b (-a/2) = g c d (-c/2)) →
  -- The graphs of f and g intersect at the point (150, -150)
  (f a b 150 = -150 ∧ g c d 150 = -150) →
  -- Conclusion: a + c = -600
  a + c = -600 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_intersection_l1392_139224


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l1392_139207

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_cond : a + b + c = 2)
  (prod_sum_cond : a * b + a * c + b * c = -3)
  (prod_cond : a * b * c = -3) :
  a^3 + b^3 + c^3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l1392_139207


namespace NUMINAMATH_CALUDE_smartphone_price_exists_l1392_139265

/-- Reverses a four-digit number -/
def reverse_four_digit (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  d * 1000 + c * 100 + b * 10 + a

/-- The main theorem stating the existence of a four-digit number satisfying the problem conditions -/
theorem smartphone_price_exists : ∃ (price : ℕ), 
  1000 ≤ price ∧ price < 10000 ∧ 
  (120 * price : ℚ) / 100 = reverse_four_digit price := by
  sorry

end NUMINAMATH_CALUDE_smartphone_price_exists_l1392_139265


namespace NUMINAMATH_CALUDE_school_play_ticket_value_l1392_139273

/-- Calculates the total value of tickets sold for a school play --/
def total_ticket_value (student_price : ℕ) (adult_price : ℕ) (child_price : ℕ) (senior_price : ℕ)
                       (student_count : ℕ) (adult_count : ℕ) (child_count : ℕ) (senior_count : ℕ) : ℕ :=
  student_price * student_count + adult_price * adult_count + child_price * child_count + senior_price * senior_count

theorem school_play_ticket_value :
  total_ticket_value 6 8 4 7 20 12 15 10 = 346 := by
  sorry

end NUMINAMATH_CALUDE_school_play_ticket_value_l1392_139273


namespace NUMINAMATH_CALUDE_quadratic_sum_l1392_139293

/-- Given a quadratic polynomial 12x^2 + 144x + 1728, when written in the form a(x+b)^2+c
    where a, b, and c are constants, prove that a + b + c = 1314 -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a b c : ℝ), (12 * x^2 + 144 * x + 1728 = a * (x + b)^2 + c) ∧ (a + b + c = 1314) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1392_139293


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l1392_139270

theorem sum_of_special_numbers (a b c : ℤ) : 
  (∀ n : ℕ, n ≥ a) → 
  (∀ m : ℤ, m < 0 → m ≤ b) → 
  (c = -c) → 
  a + b + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l1392_139270


namespace NUMINAMATH_CALUDE_modified_rubiks_cube_cubie_count_l1392_139251

/-- Represents a modified Rubik's cube with 8 corner cubies removed --/
structure ModifiedRubiksCube where
  /-- The number of small cubies with 4 painted faces --/
  four_face_cubies : Nat
  /-- The number of small cubies with 1 painted face --/
  one_face_cubies : Nat
  /-- The number of small cubies with 0 painted faces --/
  zero_face_cubies : Nat

/-- Theorem stating the correct number of cubies for each type in a modified Rubik's cube --/
theorem modified_rubiks_cube_cubie_count :
  ∃ (cube : ModifiedRubiksCube),
    cube.four_face_cubies = 12 ∧
    cube.one_face_cubies = 6 ∧
    cube.zero_face_cubies = 1 :=
by sorry

end NUMINAMATH_CALUDE_modified_rubiks_cube_cubie_count_l1392_139251


namespace NUMINAMATH_CALUDE_range_of_squared_plus_linear_l1392_139250

theorem range_of_squared_plus_linear (a b : ℝ) (h1 : a < -2) (h2 : b > 4) :
  a^2 + b > 8 := by sorry

end NUMINAMATH_CALUDE_range_of_squared_plus_linear_l1392_139250


namespace NUMINAMATH_CALUDE_floor_sqrt_18_squared_l1392_139232

theorem floor_sqrt_18_squared : ⌊Real.sqrt 18⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_18_squared_l1392_139232


namespace NUMINAMATH_CALUDE_fraction_unchanged_l1392_139262

theorem fraction_unchanged (x y : ℝ) : 
  x / (3 * x + y) = (3 * x) / (3 * (3 * x + y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l1392_139262


namespace NUMINAMATH_CALUDE_bubble_pass_probability_correct_l1392_139259

/-- Given a sequence of 50 distinct real numbers, this function calculates
    the probability that the number initially in the 25th position
    will end up in the 35th position after one bubble pass. -/
def bubble_pass_probability (seq : Fin 50 → ℝ) (h : Function.Injective seq) : ℚ :=
  1 / 1190

/-- The theorem stating that the probability is correct -/
theorem bubble_pass_probability_correct (seq : Fin 50 → ℝ) (h : Function.Injective seq) :
    bubble_pass_probability seq h = 1 / 1190 := by
  sorry

end NUMINAMATH_CALUDE_bubble_pass_probability_correct_l1392_139259


namespace NUMINAMATH_CALUDE_total_price_increase_l1392_139206

-- Define the sequence of price increases
def price_increases : List Real := [0.375, 0.31, 0.427, 0.523, 0.272]

-- Function to calculate the total price increase factor
def total_increase_factor (increases : List Real) : Real :=
  List.foldl (fun acc x => acc * (1 + x)) 1 increases

-- Theorem stating the total equivalent percent increase
theorem total_price_increase : 
  ∀ ε > 0, 
  |total_increase_factor price_increases - 1 - 3.9799| < ε := by
  sorry

end NUMINAMATH_CALUDE_total_price_increase_l1392_139206


namespace NUMINAMATH_CALUDE_expression_value_l1392_139252

theorem expression_value : (3^2 - 5*3 + 6) / (3 - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1392_139252


namespace NUMINAMATH_CALUDE_computer_price_increase_l1392_139216

def price_increase (d : ℝ) : Prop :=
  2 * d = 540 →
  ((351 - d) / d) * 100 = 30

theorem computer_price_increase : price_increase 270 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l1392_139216


namespace NUMINAMATH_CALUDE_vector_operations_and_parallelism_l1392_139284

/-- Given two 2D vectors a and b, prove properties about their linear combinations and parallelism. -/
theorem vector_operations_and_parallelism 
  (a b : ℝ × ℝ) 
  (ha : a = (2, 0)) 
  (hb : b = (1, 4)) : 
  (2 • a + 3 • b = (7, 12)) ∧ 
  (a - 2 • b = (0, -8)) ∧ 
  (∃ k : ℝ, k • a + b = (2*k + 1, 4) ∧ a + 2 • b = (4, 8) ∧ k = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_vector_operations_and_parallelism_l1392_139284


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l1392_139282

def polynomial (x : ℝ) : ℝ :=
  (x - 3) * (3 * x^3 + 2 * x^2 - 4 * x + 1) + 4 * (x^4 + x^3 - 2 * x^2 + x) - 5 * (x^3 - 3 * x + 1)

theorem nonzero_terms_count :
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    ∀ x, polynomial x = a * x^4 + b * x^3 + c * x^2 + d * x + e :=
by sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l1392_139282


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1392_139202

theorem triangle_perimeter (a b c : ℝ) (ha : a = 10) (hb : b = 15) (hc : c = 19) :
  a + b + c = 44 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1392_139202


namespace NUMINAMATH_CALUDE_ellipse_focus_coordinates_l1392_139288

-- Define the ellipse
structure Ellipse where
  major_axis_end1 : ℝ × ℝ
  major_axis_end2 : ℝ × ℝ
  minor_axis_end1 : ℝ × ℝ
  minor_axis_end2 : ℝ × ℝ

-- Define the focus
def focus (e : Ellipse) : ℝ × ℝ := sorry

-- Theorem statement
theorem ellipse_focus_coordinates (e : Ellipse) 
  (h1 : e.major_axis_end1 = (1, -2))
  (h2 : e.major_axis_end2 = (7, -2))
  (h3 : e.minor_axis_end1 = (4, 1))
  (h4 : e.minor_axis_end2 = (4, -5)) :
  focus e = (4, -2) := by sorry

end NUMINAMATH_CALUDE_ellipse_focus_coordinates_l1392_139288


namespace NUMINAMATH_CALUDE_similar_triangles_side_ratio_l1392_139208

theorem similar_triangles_side_ratio 
  (a b ka kb : ℝ) 
  (C : Real) 
  (k : ℝ) 
  (h1 : ka = k * a) 
  (h2 : kb = k * b) 
  (h3 : C > 0 ∧ C < 180) : 
  ∃ (c kc : ℝ), c > 0 ∧ kc > 0 ∧ kc = k * c :=
sorry

end NUMINAMATH_CALUDE_similar_triangles_side_ratio_l1392_139208


namespace NUMINAMATH_CALUDE_radical_axis_is_line_l1392_139218

/-- The radical axis of two non-concentric circles -/
theorem radical_axis_is_line (a R₁ R₂ : ℝ) (h : a ≠ 0) :
  {p : ℝ × ℝ | (p.1 + a)^2 + p.2^2 - R₁^2 = (p.1 - a)^2 + p.2^2 - R₂^2} =
  {p : ℝ × ℝ | p.1 = (R₂^2 - R₁^2) / (4 * a)} :=
sorry

end NUMINAMATH_CALUDE_radical_axis_is_line_l1392_139218


namespace NUMINAMATH_CALUDE_probability_prime_sum_digits_l1392_139287

def ball_numbers : List Nat := [10, 11, 13, 14, 17, 19, 21, 23]

def sum_of_digits (n : Nat) : Nat :=
  n.repr.foldl (fun acc d => acc + d.toNat - 48) 0

def is_prime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

theorem probability_prime_sum_digits :
  let favorable_outcomes := (ball_numbers.map sum_of_digits).filter is_prime |>.length
  let total_outcomes := ball_numbers.length
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_prime_sum_digits_l1392_139287


namespace NUMINAMATH_CALUDE_tuesday_temperature_l1392_139219

-- Define temperatures for each day
def tuesday_temp : ℝ := sorry
def wednesday_temp : ℝ := sorry
def thursday_temp : ℝ := sorry
def friday_temp : ℝ := 53

-- Define the conditions
axiom avg_tue_wed_thu : (tuesday_temp + wednesday_temp + thursday_temp) / 3 = 52
axiom avg_wed_thu_fri : (wednesday_temp + thursday_temp + friday_temp) / 3 = 54

-- Theorem to prove
theorem tuesday_temperature : tuesday_temp = 47 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_temperature_l1392_139219


namespace NUMINAMATH_CALUDE_base4_division_theorem_l1392_139243

/-- Represents a number in base 4 --/
def Base4 : Type := Nat

/-- Converts a Base4 number to its decimal representation --/
def to_decimal (n : Base4) : Nat :=
  sorry

/-- Converts a decimal number to its Base4 representation --/
def to_base4 (n : Nat) : Base4 :=
  sorry

/-- Performs division in Base4 --/
def base4_div (a b : Base4) : Base4 :=
  sorry

theorem base4_division_theorem :
  base4_div (to_base4 1023) (to_base4 11) = to_base4 33 := by
  sorry

end NUMINAMATH_CALUDE_base4_division_theorem_l1392_139243


namespace NUMINAMATH_CALUDE_rooster_to_hen_ratio_l1392_139205

/-- Given a chicken farm with roosters and hens, prove the ratio of roosters to hens. -/
theorem rooster_to_hen_ratio 
  (total_chickens : ℕ) 
  (roosters : ℕ) 
  (h_total : total_chickens = 9000)
  (h_roosters : roosters = 6000) : 
  (roosters : ℚ) / (total_chickens - roosters) = 2 := by
  sorry

end NUMINAMATH_CALUDE_rooster_to_hen_ratio_l1392_139205


namespace NUMINAMATH_CALUDE_unique_f_one_l1392_139211

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (x * y) = x^2 + y^2 - x * y

/-- The theorem stating that f(1) = 2 is the only solution -/
theorem unique_f_one (f : ℝ → ℝ) (h : SatisfiesEquation f) : f 1 = 2 := by
  sorry

#check unique_f_one

end NUMINAMATH_CALUDE_unique_f_one_l1392_139211


namespace NUMINAMATH_CALUDE_no_adjacent_standing_probability_l1392_139292

/-- Recursive function to calculate the number of valid arrangements -/
def validArrangements : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => validArrangements (n + 1) + validArrangements n

/-- The number of people around the table -/
def numPeople : ℕ := 10

/-- The total number of possible outcomes when flipping n fair coins -/
def totalOutcomes (n : ℕ) : ℕ := 2^n

/-- The probability of no two adjacent people standing for n people -/
def noAdjacentStandingProb (n : ℕ) : ℚ :=
  (validArrangements n : ℚ) / (totalOutcomes n : ℚ)

theorem no_adjacent_standing_probability :
  noAdjacentStandingProb numPeople = 123 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_no_adjacent_standing_probability_l1392_139292


namespace NUMINAMATH_CALUDE_grid_whitening_theorem_l1392_139226

/-- Represents the color of a square -/
inductive Color
| Black
| White

/-- Represents a grid of squares -/
def Grid := Matrix (Fin 98) (Fin 98) Color

/-- Represents a sub-rectangle in the grid -/
structure SubRectangle where
  top_left : Fin 98 × Fin 98
  width : Nat
  height : Nat
  width_valid : width > 1
  height_valid : height > 1
  in_bounds : top_left.1 + width ≤ 98 ∧ top_left.2 + height ≤ 98

/-- Represents a color-flipping operation on a sub-rectangle -/
def flip_operation (grid : Grid) (rect : SubRectangle) : Grid :=
  sorry

/-- Represents a sequence of color-flipping operations -/
def operation_sequence := List SubRectangle

/-- Applies a sequence of operations to a grid -/
def apply_operations (grid : Grid) (ops : operation_sequence) : Grid :=
  sorry

/-- Checks if all squares in the grid are white -/
def all_white (grid : Grid) : Prop :=
  sorry

/-- Main theorem: There exists a finite sequence of operations that turns any grid all white -/
theorem grid_whitening_theorem (initial_grid : Grid) :
  ∃ (ops : operation_sequence), all_white (apply_operations initial_grid ops) :=
sorry

end NUMINAMATH_CALUDE_grid_whitening_theorem_l1392_139226


namespace NUMINAMATH_CALUDE_two_digit_number_value_l1392_139277

/-- Represents a two-digit number with 5 as its tens digit and A as its units digit -/
def two_digit_number (A : ℕ) : ℕ := 50 + A

/-- The value of a two-digit number with 5 as its tens digit and A as its units digit -/
theorem two_digit_number_value (A : ℕ) (h : A < 10) : 
  two_digit_number A = 50 + A := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_value_l1392_139277


namespace NUMINAMATH_CALUDE_subtraction_from_percentage_l1392_139244

theorem subtraction_from_percentage (n : ℝ) : n = 85 → 0.4 * n - 11 = 23 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_from_percentage_l1392_139244


namespace NUMINAMATH_CALUDE_x_one_minus_f_equals_one_l1392_139237

/-- Proves that for x = (3 + √5)^20, n = ⌊x⌋, and f = x - n, x(1 - f) = 1 -/
theorem x_one_minus_f_equals_one :
  let x : ℝ := (3 + Real.sqrt 5) ^ 20
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_x_one_minus_f_equals_one_l1392_139237


namespace NUMINAMATH_CALUDE_inequality_proof_l1392_139221

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≤ 1) : x^6 - y^6 + 2*y^3 < π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1392_139221


namespace NUMINAMATH_CALUDE_solve_for_C_l1392_139253

theorem solve_for_C : ∃ C : ℝ, 4 * C + 3 = 31 ∧ C = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_C_l1392_139253


namespace NUMINAMATH_CALUDE_selection_method1_selection_method2_selection_method3_selection_method4_l1392_139296

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of athletes -/
def total_athletes : ℕ := 10

/-- The number of male athletes -/
def male_athletes : ℕ := 6

/-- The number of female athletes -/
def female_athletes : ℕ := 4

/-- The number of athletes to be selected -/
def selected_athletes : ℕ := 5

/-- The number of ways to select 3 males and 2 females -/
theorem selection_method1 : choose male_athletes 3 * choose female_athletes 2 = 120 := sorry

/-- The number of ways to select with at least one captain participating -/
theorem selection_method2 : 2 * choose 8 4 + choose 8 3 = 196 := sorry

/-- The number of ways to select with at least one female athlete -/
theorem selection_method3 : choose total_athletes selected_athletes - choose male_athletes selected_athletes = 246 := sorry

/-- The number of ways to select with both a captain and at least one female athlete -/
theorem selection_method4 : choose 9 4 + choose 8 4 - choose 5 4 = 191 := sorry

end NUMINAMATH_CALUDE_selection_method1_selection_method2_selection_method3_selection_method4_l1392_139296


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1392_139279

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => x^2 + 2*x - 3
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -3 ∨ x ≥ 1} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1392_139279


namespace NUMINAMATH_CALUDE_largest_perimeter_l1392_139272

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℤ

/-- Checks if the given lengths can form a valid triangle --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℝ :=
  t.side1 + t.side2 + t.side3

/-- Theorem stating the largest possible perimeter for the given triangle --/
theorem largest_perimeter :
  ∃ (t : Triangle), t.side1 = 8 ∧ t.side2 = 12 ∧ is_valid_triangle t ∧
  ∀ (t' : Triangle), t'.side1 = 8 ∧ t'.side2 = 12 ∧ is_valid_triangle t' →
  perimeter t ≥ perimeter t' ∧ perimeter t = 39 :=
sorry

end NUMINAMATH_CALUDE_largest_perimeter_l1392_139272


namespace NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_a_range_l1392_139222

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 2|

-- Theorem for part 1
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f a x ≥ 3) ∧ (∃ x, f a x = 3) → a = 1 ∨ a = -5 :=
sorry

-- Theorem for part 2
theorem inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_a_range_l1392_139222


namespace NUMINAMATH_CALUDE_halloween_cleanup_time_l1392_139268

theorem halloween_cleanup_time
  (egg_cleanup_time : ℕ)
  (tp_cleanup_time : ℕ)
  (egg_count : ℕ)
  (tp_count : ℕ)
  (h1 : egg_cleanup_time = 15)  -- 15 seconds per egg
  (h2 : tp_cleanup_time = 30)   -- 30 minutes per roll of toilet paper
  (h3 : egg_count = 60)         -- 60 eggs
  (h4 : tp_count = 7)           -- 7 rolls of toilet paper
  : (egg_count * egg_cleanup_time) / 60 + tp_count * tp_cleanup_time = 225 := by
  sorry

#check halloween_cleanup_time

end NUMINAMATH_CALUDE_halloween_cleanup_time_l1392_139268


namespace NUMINAMATH_CALUDE_f_negative_l1392_139217

-- Define an even function f
def f (x : ℝ) : ℝ := sorry

-- State the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_positive : ∀ x > 0, f x = x^2 + x

-- Theorem to prove
theorem f_negative : ∀ x < 0, f x = x^2 - x := by sorry

end NUMINAMATH_CALUDE_f_negative_l1392_139217


namespace NUMINAMATH_CALUDE_kendall_driving_distance_l1392_139267

/-- The distance Kendall drove with her mother in miles -/
def mother_distance : ℝ := 0.17

/-- The distance Kendall drove with her father in miles -/
def father_distance : ℝ := 0.5

/-- The distance Kendall drove with her friend in miles -/
def friend_distance : ℝ := 0.68

/-- The conversion factor from miles to kilometers -/
def mile_to_km : ℝ := 1.60934

/-- The total distance Kendall drove in kilometers -/
def total_distance_km : ℝ := (mother_distance + father_distance + friend_distance) * mile_to_km

theorem kendall_driving_distance :
  ∃ ε > 0, |total_distance_km - 2.17| < ε :=
sorry

end NUMINAMATH_CALUDE_kendall_driving_distance_l1392_139267


namespace NUMINAMATH_CALUDE_original_bananas_total_l1392_139209

theorem original_bananas_total (willie_bananas charles_bananas : ℝ) 
  (h1 : willie_bananas = 48.0) 
  (h2 : charles_bananas = 35.0) : 
  willie_bananas + charles_bananas = 83.0 := by
  sorry

end NUMINAMATH_CALUDE_original_bananas_total_l1392_139209


namespace NUMINAMATH_CALUDE_angle_equality_l1392_139260

theorem angle_equality (θ : Real) (h1 : Real.sqrt 2 * Real.sin (10 * π / 180) = Real.cos θ - Real.sin θ) 
                       (h2 : 0 < θ ∧ θ < π / 2) : θ = 35 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l1392_139260


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l1392_139299

theorem expression_equals_negative_one 
  (x y z : ℝ) 
  (hx : x ≠ 1) 
  (hy : y ≠ 2) 
  (hz : z ≠ 3) : 
  (x - 1) / (3 - z) * (y - 2) / (1 - x) * (z - 3) / (2 - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l1392_139299


namespace NUMINAMATH_CALUDE_x_squared_mod_20_l1392_139220

theorem x_squared_mod_20 (x : ℕ) (h1 : 5 * x ≡ 10 [ZMOD 20]) (h2 : 2 * x ≡ 14 [ZMOD 20]) :
  x^2 ≡ 16 [ZMOD 20] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_20_l1392_139220


namespace NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l1392_139241

theorem sqrt_inequality_solution_set (x : ℝ) :
  x + 3 ≥ 0 →
  (Real.sqrt (x + 3) > 3 - x ↔ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l1392_139241


namespace NUMINAMATH_CALUDE_east_north_not_opposite_forward_backward_opposite_main_theorem_l1392_139298

/-- Represents a direction of movement --/
inductive Direction
  | Forward
  | Backward
  | East
  | North

/-- Represents a quantity with a value and a direction --/
structure Quantity where
  value : ℝ
  direction : Direction

/-- Defines when two quantities are opposite --/
def are_opposite (q1 q2 : Quantity) : Prop :=
  (q1.value = q2.value) ∧
  ((q1.direction = Direction.Forward ∧ q2.direction = Direction.Backward) ∨
   (q1.direction = Direction.Backward ∧ q2.direction = Direction.Forward))

/-- Theorem stating that east and north movements are not opposite --/
theorem east_north_not_opposite :
  ¬(are_opposite
      { value := 10, direction := Direction.East }
      { value := 10, direction := Direction.North }) :=
by
  sorry

/-- Theorem stating that forward and backward movements are opposite --/
theorem forward_backward_opposite :
  are_opposite
    { value := 5, direction := Direction.Forward }
    { value := 5, direction := Direction.Backward } :=
by
  sorry

/-- Main theorem proving that east and north movements are not opposite,
    while forward and backward movements are opposite --/
theorem main_theorem :
  (¬(are_opposite
      { value := 10, direction := Direction.East }
      { value := 10, direction := Direction.North })) ∧
  (are_opposite
    { value := 5, direction := Direction.Forward }
    { value := 5, direction := Direction.Backward }) :=
by
  sorry

end NUMINAMATH_CALUDE_east_north_not_opposite_forward_backward_opposite_main_theorem_l1392_139298


namespace NUMINAMATH_CALUDE_kite_parabolas_theorem_l1392_139256

/-- Represents a parabola in the form y = ax² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Represents the parameters of our problem -/
structure KiteParameters where
  parabola1 : Parabola
  parabola2 : Parabola
  kite_area : ℝ

/-- The main theorem statement -/
theorem kite_parabolas_theorem (params : KiteParameters) : 
  params.parabola1.a + params.parabola2.a = 1.04 :=
by
  sorry

/-- The specific instance of our problem -/
def our_problem : KiteParameters :=
  { parabola1 := { a := 2, b := -3 }
  , parabola2 := { a := -1, b := 5 }
  , kite_area := 20
  }

#check kite_parabolas_theorem our_problem

end NUMINAMATH_CALUDE_kite_parabolas_theorem_l1392_139256


namespace NUMINAMATH_CALUDE_logarithm_sum_property_l1392_139230

theorem logarithm_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by sorry

end NUMINAMATH_CALUDE_logarithm_sum_property_l1392_139230

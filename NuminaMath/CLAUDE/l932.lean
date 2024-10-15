import Mathlib

namespace NUMINAMATH_CALUDE_circle_area_polar_l932_93299

/-- The area of the circle described by the polar equation r = 4 cos θ - 3 sin θ is 25π/4 -/
theorem circle_area_polar (θ : ℝ) (r : ℝ → ℝ) : 
  (r θ = 4 * Real.cos θ - 3 * Real.sin θ) → 
  (∃ c : ℝ × ℝ, ∃ radius : ℝ, 
    (∀ x y : ℝ, (x - c.1)^2 + (y - c.2)^2 = radius^2 ↔ 
      ∃ θ : ℝ, x = r θ * Real.cos θ ∧ y = r θ * Real.sin θ) ∧
    π * radius^2 = 25 * π / 4) :=
sorry

end NUMINAMATH_CALUDE_circle_area_polar_l932_93299


namespace NUMINAMATH_CALUDE_total_candy_is_98_l932_93270

/-- The number of boxes of chocolate candy Adam bought -/
def chocolate_boxes : ℕ := 3

/-- The number of pieces in each box of chocolate candy -/
def chocolate_pieces_per_box : ℕ := 6

/-- The number of boxes of caramel candy Adam bought -/
def caramel_boxes : ℕ := 5

/-- The number of pieces in each box of caramel candy -/
def caramel_pieces_per_box : ℕ := 8

/-- The number of boxes of gummy candy Adam bought -/
def gummy_boxes : ℕ := 4

/-- The number of pieces in each box of gummy candy -/
def gummy_pieces_per_box : ℕ := 10

/-- The total number of candy pieces Adam bought -/
def total_candy : ℕ := chocolate_boxes * chocolate_pieces_per_box + 
                        caramel_boxes * caramel_pieces_per_box + 
                        gummy_boxes * gummy_pieces_per_box

theorem total_candy_is_98 : total_candy = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_candy_is_98_l932_93270


namespace NUMINAMATH_CALUDE_francis_muffins_l932_93244

/-- The cost of breakfast for Francis and Kiera -/
def breakfast_cost : ℕ → ℕ
| m => 2 * m + 2 * 3 + 2 * 2 + 3

/-- The theorem stating that Francis had 2 muffins -/
theorem francis_muffins : 
  ∃ m : ℕ, breakfast_cost m = 17 ∧ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_francis_muffins_l932_93244


namespace NUMINAMATH_CALUDE_smallest_positive_value_36k_minus_5m_l932_93297

theorem smallest_positive_value_36k_minus_5m (k m : ℕ+) :
  (∀ n : ℕ+, 36^(k : ℕ) - 5^(m : ℕ) ≠ n) →
  (36^(k : ℕ) - 5^(m : ℕ) = 11 ∨ 36^(k : ℕ) - 5^(m : ℕ) > 11) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_value_36k_minus_5m_l932_93297


namespace NUMINAMATH_CALUDE_no_perfect_squares_l932_93255

/-- Represents a 100-digit number with a repeating pattern -/
def RepeatingNumber (pattern : ℕ) : ℕ :=
  -- Implementation details omitted for simplicity
  sorry

/-- N₁ is a 100-digit number consisting of all 3's -/
def N1 : ℕ := RepeatingNumber 3

/-- N₂ is a 100-digit number consisting of all 6's -/
def N2 : ℕ := RepeatingNumber 6

/-- N₃ is a 100-digit number with repeating pattern 15 -/
def N3 : ℕ := RepeatingNumber 15

/-- N₄ is a 100-digit number with repeating pattern 21 -/
def N4 : ℕ := RepeatingNumber 21

/-- N₅ is a 100-digit number with repeating pattern 27 -/
def N5 : ℕ := RepeatingNumber 27

/-- A number is a perfect square if there exists an integer whose square equals the number -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_perfect_squares : ¬(is_perfect_square N1 ∨ is_perfect_square N2 ∨ 
                               is_perfect_square N3 ∨ is_perfect_square N4 ∨ 
                               is_perfect_square N5) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l932_93255


namespace NUMINAMATH_CALUDE_sum_x_y_equals_thirteen_l932_93238

theorem sum_x_y_equals_thirteen 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 16)
  (h2 : (16:ℝ)^(x+y) / (4:ℝ)^(7*y) = 1024)
  : x + y = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_thirteen_l932_93238


namespace NUMINAMATH_CALUDE_hyperbola_equation_l932_93229

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the eccentricity of the hyperbola
def hyperbola_eccentricity : ℝ := 2

-- Define the standard form of a hyperbola
def is_standard_hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Theorem statement
theorem hyperbola_equation : 
  ∃ (a b : ℝ), a^2 = 4 ∧ b^2 = 12 ∧ 
  (∀ (x y : ℝ), is_standard_hyperbola a b x y) ∧
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ 
   c = hyperbola_eccentricity * a ∧
   c^2 = 25 - 9) := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l932_93229


namespace NUMINAMATH_CALUDE_laptop_sale_price_l932_93266

def original_price : ℝ := 500
def first_discount : ℝ := 0.10
def second_discount : ℝ := 0.20
def delivery_fee : ℝ := 30

theorem laptop_sale_price :
  let price_after_first_discount := original_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  let final_price := price_after_second_discount + delivery_fee
  final_price = 390 := by sorry

end NUMINAMATH_CALUDE_laptop_sale_price_l932_93266


namespace NUMINAMATH_CALUDE_smallest_valid_integers_difference_l932_93210

def is_valid (n : ℕ) : Prop :=
  n > 2 ∧ ∀ k : ℕ, 2 ≤ k ∧ k ≤ 12 → n % k = 2

theorem smallest_valid_integers_difference :
  ∃ n m : ℕ, is_valid n ∧ is_valid m ∧
  (∀ x : ℕ, is_valid x → n ≤ x) ∧
  (∀ x : ℕ, is_valid x ∧ x ≠ n → m ≤ x) ∧
  m - n = 13860 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_integers_difference_l932_93210


namespace NUMINAMATH_CALUDE_parallelogram_area_l932_93217

/-- The area of a parallelogram with a 150-degree angle and consecutive sides of 10 and 20 units --/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 20) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin θ = 100 * Real.sqrt 3 := by
  sorry

#check parallelogram_area

end NUMINAMATH_CALUDE_parallelogram_area_l932_93217


namespace NUMINAMATH_CALUDE_stone_99_is_11_l932_93243

/-- Represents the counting pattern for 12 stones -/
def stone_count (n : ℕ) : ℕ :=
  let cycle := 22  -- The pattern repeats every 22 counts
  let within_cycle := n % cycle
  if within_cycle ≤ 12
  then within_cycle
  else 13 - (within_cycle - 11)

/-- The theorem stating that the 99th count corresponds to the 11th stone -/
theorem stone_99_is_11 : stone_count 99 = 11 := by
  sorry

#eval stone_count 99  -- This should output 11

end NUMINAMATH_CALUDE_stone_99_is_11_l932_93243


namespace NUMINAMATH_CALUDE_equation_solution_l932_93287

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 8) * x = 14 ∧ x = 392 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l932_93287


namespace NUMINAMATH_CALUDE_lemonade_stand_revenue_l932_93254

theorem lemonade_stand_revenue 
  (total_cups : ℝ) 
  (small_cup_price : ℝ) 
  (h1 : small_cup_price > 0) : 
  let small_cups := (3 / 5) * total_cups
  let large_cups := (2 / 5) * total_cups
  let large_cup_price := (1 / 6) * small_cup_price
  let small_revenue := small_cups * small_cup_price
  let large_revenue := large_cups * large_cup_price
  let total_revenue := small_revenue + large_revenue
  (large_revenue / total_revenue) = (1 / 10) := by
sorry

end NUMINAMATH_CALUDE_lemonade_stand_revenue_l932_93254


namespace NUMINAMATH_CALUDE_ms_delmont_class_size_l932_93233

/-- Proves the number of students in Ms. Delmont's class given the cupcake distribution -/
theorem ms_delmont_class_size 
  (total_cupcakes : ℕ)
  (adults_received : ℕ)
  (mrs_donnelly_class : ℕ)
  (leftover_cupcakes : ℕ)
  (h1 : total_cupcakes = 40)
  (h2 : adults_received = 4)
  (h3 : mrs_donnelly_class = 16)
  (h4 : leftover_cupcakes = 2) :
  total_cupcakes - adults_received - mrs_donnelly_class - leftover_cupcakes = 18 := by
  sorry

#check ms_delmont_class_size

end NUMINAMATH_CALUDE_ms_delmont_class_size_l932_93233


namespace NUMINAMATH_CALUDE_intersection_chord_length_l932_93256

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := x^2 = 8*y
def line (x y : ℝ) : Prop := y = x + 2

-- Define the intersection points
def intersection_points (M N : ℝ × ℝ) : Prop :=
  parabola M.1 M.2 ∧ line M.1 M.2 ∧
  parabola N.1 N.2 ∧ line N.1 N.2 ∧
  M ≠ N

-- Theorem statement
theorem intersection_chord_length :
  ∀ M N : ℝ × ℝ, intersection_points M N →
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 16 :=
sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l932_93256


namespace NUMINAMATH_CALUDE_euro_equation_solution_l932_93265

def euro (x y : ℝ) : ℝ := 2 * x * y

theorem euro_equation_solution (x : ℝ) : 
  euro 9 (euro 4 x) = 720 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_euro_equation_solution_l932_93265


namespace NUMINAMATH_CALUDE_stella_toilet_paper_l932_93204

/-- The number of packs of toilet paper Stella needs to buy for 4 weeks -/
def toilet_paper_packs (bathrooms : ℕ) (rolls_per_bathroom_per_day : ℕ) 
  (days_per_week : ℕ) (rolls_per_pack : ℕ) (weeks : ℕ) : ℕ :=
  (bathrooms * rolls_per_bathroom_per_day * days_per_week * weeks) / rolls_per_pack

/-- Stella's toilet paper restocking problem -/
theorem stella_toilet_paper : 
  toilet_paper_packs 6 1 7 12 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_stella_toilet_paper_l932_93204


namespace NUMINAMATH_CALUDE_inequality_system_solution_l932_93273

-- Define the inequality system
def inequality_system (x : ℝ) : Prop := x + 1 > 0 ∧ x > -3

-- Define the solution set
def solution_set : Set ℝ := {x | x > -1}

-- Theorem statement
theorem inequality_system_solution :
  {x : ℝ | inequality_system x} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l932_93273


namespace NUMINAMATH_CALUDE_x_2000_value_l932_93249

def sequence_property (x : ℕ → ℝ) :=
  ∀ n, x (n + 1) + x (n + 2) + x (n + 3) = 20

theorem x_2000_value (x : ℕ → ℝ) 
  (h1 : sequence_property x) 
  (h2 : x 4 = 9) 
  (h3 : x 12 = 7) : 
  x 2000 = 4 := by
sorry

end NUMINAMATH_CALUDE_x_2000_value_l932_93249


namespace NUMINAMATH_CALUDE_min_workers_for_profit_l932_93248

/-- Represents the problem of finding the minimum number of workers needed for profit --/
theorem min_workers_for_profit (
  maintenance_cost : ℝ)
  (worker_hourly_wage : ℝ)
  (widgets_per_hour : ℝ)
  (widget_price : ℝ)
  (work_hours : ℝ)
  (h1 : maintenance_cost = 600)
  (h2 : worker_hourly_wage = 20)
  (h3 : widgets_per_hour = 6)
  (h4 : widget_price = 3.5)
  (h5 : work_hours = 8)
  : ∃ n : ℕ, n = 76 ∧ ∀ m : ℕ, m < n → maintenance_cost + m * worker_hourly_wage * work_hours ≥ m * widgets_per_hour * widget_price * work_hours :=
sorry

end NUMINAMATH_CALUDE_min_workers_for_profit_l932_93248


namespace NUMINAMATH_CALUDE_triangle_perimeter_l932_93207

/-- Proves that a triangle with inradius 2.5 cm and area 50 cm² has a perimeter of 40 cm -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 50 → A = r * (p / 2) → p = 40 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l932_93207


namespace NUMINAMATH_CALUDE_minimize_sum_distances_l932_93267

/-- The point that minimizes the sum of distances from two fixed points on a line --/
theorem minimize_sum_distances (A B C : ℝ × ℝ) : 
  A = (3, 6) → 
  B = (6, 2) → 
  C.2 = 0 → 
  (∀ k : ℝ, C = (k, 0) → 
    Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) + 
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) ≥ 
    Real.sqrt ((6.75 - A.1)^2 + (0 - A.2)^2) + 
    Real.sqrt ((6.75 - B.1)^2 + (0 - B.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_minimize_sum_distances_l932_93267


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l932_93283

theorem quadratic_equal_roots (m n : ℝ) : 
  (∃ x : ℝ, x^(m-1) + 4*x - n = 0 ∧ 
   ∀ y : ℝ, y^(m-1) + 4*y - n = 0 → y = x) →
  m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l932_93283


namespace NUMINAMATH_CALUDE_ninth_term_of_sequence_l932_93260

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r ^ (n - 1)

theorem ninth_term_of_sequence :
  let a₁ : ℚ := 5
  let r : ℚ := 3/2
  geometric_sequence a₁ r 9 = 32805/256 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_sequence_l932_93260


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l932_93208

/-- An ellipse passing through (2,3) with foci at (-2,0) and (2,0) has eccentricity 1/2 -/
theorem ellipse_eccentricity : ∀ (e : ℝ), 
  (∃ (a b : ℝ), 
    (2/a)^2 + (3/b)^2 = 1 ∧  -- ellipse passes through (2,3)
    a > b ∧ b > 0 ∧         -- standard form constraints
    4 = a^2 - b^2) →        -- distance between foci is 4
  e = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l932_93208


namespace NUMINAMATH_CALUDE_cosine_identity_l932_93239

theorem cosine_identity (n : Real) : 
  (Real.cos (30 * π / 180 - n * π / 180)) / (Real.cos (n * π / 180)) = 
  (1 / 2) * (Real.sqrt 3 + Real.tan (n * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l932_93239


namespace NUMINAMATH_CALUDE_one_ta_grading_time_l932_93201

/-- The number of initial teaching assistants -/
def N : ℕ := 5

/-- The time it takes N teaching assistants to grade all homework -/
def initial_time : ℕ := 5

/-- The time it takes N+1 teaching assistants to grade all homework -/
def new_time : ℕ := 4

/-- The total work required to grade all homework -/
def total_work : ℕ := N * initial_time

theorem one_ta_grading_time :
  (total_work : ℚ) = 20 :=
by sorry

end NUMINAMATH_CALUDE_one_ta_grading_time_l932_93201


namespace NUMINAMATH_CALUDE_distance_to_concert_l932_93246

/-- The distance to a concert given the distance driven before and after a gas stop -/
theorem distance_to_concert (distance_before_gas : ℕ) (distance_after_gas : ℕ) :
  distance_before_gas = 32 →
  distance_after_gas = 46 →
  distance_before_gas + distance_after_gas = 78 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_concert_l932_93246


namespace NUMINAMATH_CALUDE_ellipse_intersection_properties_l932_93271

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the upper vertex A
def A : ℝ × ℝ := (0, 1)

-- Define a line not passing through A
def line (k m : ℝ) (x : ℝ) : ℝ := k * x + m

-- Define the condition that the line intersects the ellipse at P and Q
def intersects (k m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    y₁ = line k m x₁ ∧ y₂ = line k m x₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - A.1) * (x₂ - A.1) + (y₁ - A.2) * (y₂ - A.2) = 0

-- Main theorem
theorem ellipse_intersection_properties :
  ∀ (k m : ℝ),
    intersects k m →
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
      y₁ = line k m x₁ ∧ y₂ = line k m x₂ →
      perpendicular x₁ y₁ x₂ y₂) →
    (m = -1/2) ∧
    (∃ (S : Set ℝ), S = {s | s ≥ 9/4} ∧
      ∀ (x₁ y₁ x₂ y₂ : ℝ),
        ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
        y₁ = line k m x₁ ∧ y₂ = line k m x₂ →
        ∃ (area : ℝ), area ∈ S ∧
          (∃ (Bx By : ℝ), area = 1/2 * |Bx - x₁| * |By - y₁|)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_properties_l932_93271


namespace NUMINAMATH_CALUDE_digit_property_l932_93289

theorem digit_property (z : Nat) :
  (z < 10) →
  (∀ k : Nat, k ≥ 1 → ∃ n : Nat, n ≥ 1 ∧ n^9 % (10^k) = z^k % (10^k)) ↔
  z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9 :=
by sorry

end NUMINAMATH_CALUDE_digit_property_l932_93289


namespace NUMINAMATH_CALUDE_population_increase_birth_rate_l932_93257

/-- Calculates the percentage increase in population due to birth over a given time period. -/
def population_increase_percentage (initial_population : ℕ) (final_population : ℕ) 
  (years : ℕ) (emigration_rate : ℕ) (immigration_rate : ℕ) : ℚ :=
  let net_migration := (immigration_rate - emigration_rate) * years
  let total_increase := final_population - initial_population - net_migration
  (total_increase : ℚ) / (initial_population : ℚ) * 100

/-- The percentage increase in population due to birth over 10 years is 55%. -/
theorem population_increase_birth_rate : 
  population_increase_percentage 100000 165000 10 2000 2500 = 55 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_birth_rate_l932_93257


namespace NUMINAMATH_CALUDE_book_weight_l932_93218

theorem book_weight (total_weight : ℝ) (num_books : ℕ) (h1 : total_weight = 42) (h2 : num_books = 14) :
  total_weight / num_books = 3 := by
sorry

end NUMINAMATH_CALUDE_book_weight_l932_93218


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l932_93276

def p (x : ℝ) : ℝ := 8 * x^4 + 26 * x^3 - 65 * x^2 + 24 * x

theorem roots_of_polynomial :
  (p 0 = 0) ∧ (p (1/2) = 0) ∧ (p (3/2) = 0) ∧ (p (-4) = 0) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l932_93276


namespace NUMINAMATH_CALUDE_max_runs_in_match_l932_93212

/-- Represents the maximum number of runs that can be scored in a single delivery -/
def max_runs_per_delivery : ℕ := 6

/-- Represents the number of deliveries in an over -/
def deliveries_per_over : ℕ := 6

/-- Represents the total number of overs in the match -/
def total_overs : ℕ := 35

/-- Represents the maximum number of consecutive boundaries allowed in an over -/
def max_consecutive_boundaries : ℕ := 3

/-- Calculates the maximum runs that can be scored in a single over -/
def max_runs_per_over : ℕ :=
  max_consecutive_boundaries * max_runs_per_delivery + 
  (deliveries_per_over - max_consecutive_boundaries)

/-- Theorem: The maximum number of runs a batsman can score in the given match is 735 -/
theorem max_runs_in_match : 
  total_overs * max_runs_per_over = 735 := by
  sorry

end NUMINAMATH_CALUDE_max_runs_in_match_l932_93212


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l932_93242

theorem arithmetic_calculations : 
  (1 - (-5) * ((-1)^2) - 4 / ((-1/2)^2) = -11) ∧ 
  ((-2)^3 * (1/8) + (2/3 - 1/2 - 1/4) / (1/12) = -2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l932_93242


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_10001_l932_93215

theorem largest_prime_factor_of_10001 : ∃ p : ℕ, 
  p.Prime ∧ p ∣ 10001 ∧ ∀ q : ℕ, q.Prime → q ∣ 10001 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_10001_l932_93215


namespace NUMINAMATH_CALUDE_a_in_interval_l932_93284

/-- The function f(x) = x^2 + ax + b -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The set A = {x ∈ ℝ | f(x) ≤ 0} -/
def set_A (a b : ℝ) : Set ℝ := {x | f a b x ≤ 0}

/-- The set B = {x ∈ ℝ | f(f(x) + 1) ≤ 0} -/
def set_B (a b : ℝ) : Set ℝ := {x | f a b (f a b x + 1) ≤ 0}

/-- Theorem: If A = B ≠ ∅, then a ∈ [-2, 2] -/
theorem a_in_interval (a b : ℝ) :
  set_A a b = set_B a b ∧ (set_A a b).Nonempty → a ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_a_in_interval_l932_93284


namespace NUMINAMATH_CALUDE_factor_x4_minus_81_l932_93288

theorem factor_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end NUMINAMATH_CALUDE_factor_x4_minus_81_l932_93288


namespace NUMINAMATH_CALUDE_race_distance_l932_93219

/-- The race problem -/
theorem race_distance (a_time b_time : ℝ) (lead_distance : ℝ) (race_distance : ℝ) : 
  a_time = 36 →
  b_time = 45 →
  lead_distance = 30 →
  (race_distance / a_time) * b_time = race_distance + lead_distance →
  race_distance = 120 := by
sorry

end NUMINAMATH_CALUDE_race_distance_l932_93219


namespace NUMINAMATH_CALUDE_tamara_height_is_62_l932_93258

/-- Calculates Tamara's height given Kim's height and the age difference effect -/
def tamaraHeight (kimHeight : ℝ) (ageDifference : ℕ) : ℝ :=
  2 * kimHeight - 4

/-- Calculates Gavin's height given Kim's height -/
def gavinHeight (kimHeight : ℝ) : ℝ :=
  2 * kimHeight + 6

/-- The combined height of all three people -/
def combinedHeight : ℝ := 200

/-- The age difference between Tamara and Kim -/
def ageDifference : ℕ := 5

/-- The change in height ratio per year of age difference -/
def ratioChangePerYear : ℝ := 0.2

theorem tamara_height_is_62 :
  ∃ (kimHeight : ℝ),
    tamaraHeight kimHeight ageDifference +
    kimHeight +
    gavinHeight kimHeight = combinedHeight ∧
    tamaraHeight kimHeight ageDifference = 62 := by
  sorry

end NUMINAMATH_CALUDE_tamara_height_is_62_l932_93258


namespace NUMINAMATH_CALUDE_f_properties_l932_93253

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 1 / (2^x - 1)

theorem f_properties (a : ℝ) :
  (∀ x : ℝ, f a x ≠ 0 ↔ x ≠ 0) ∧
  (∀ x : ℝ, f a (-x) = -(f a x) ↔ a = 1/2) ∧
  (a = 1/2 → ∀ x : ℝ, x ≠ 0 → x^3 * f a x > 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l932_93253


namespace NUMINAMATH_CALUDE_polynomial_remainder_l932_93280

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^3 - 20 * x^2 + 28 * x - 26) % (4 * x - 8) = 14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l932_93280


namespace NUMINAMATH_CALUDE_g_is_odd_l932_93272

noncomputable def g (x : ℝ) : ℝ := Real.log (x^3 + Real.sqrt (1 + x^6))

theorem g_is_odd : ∀ x, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_g_is_odd_l932_93272


namespace NUMINAMATH_CALUDE_solve_equation_l932_93213

theorem solve_equation (x : ℝ) :
  (1 / 7 : ℝ) + 7 / x = 15 / x + (1 / 15 : ℝ) → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l932_93213


namespace NUMINAMATH_CALUDE_no_intersection_in_S_l932_93262

-- Define the set S of polynomials
inductive S : (ℝ → ℝ) → Prop
  | base : S (λ x => x)
  | mul {f} : S f → S (λ x => x * f x)
  | add {f} : S f → S (λ x => x + (1 - x) * f x)

-- Theorem statement
theorem no_intersection_in_S (f g : ℝ → ℝ) (hf : S f) (hg : S g) (h_neq : f ≠ g) :
  ∀ x, 0 < x → x < 1 → f x ≠ g x :=
sorry

end NUMINAMATH_CALUDE_no_intersection_in_S_l932_93262


namespace NUMINAMATH_CALUDE_abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two_l932_93278

theorem abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two :
  (∀ x : ℝ, x < -2 → |x| > 2) ∧
  ¬(∀ x : ℝ, |x| > 2 → x < -2) :=
by sorry

end NUMINAMATH_CALUDE_abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two_l932_93278


namespace NUMINAMATH_CALUDE_sum_of_inscribed_angles_is_180_l932_93227

/-- A regular pentagon inscribed in a circle -/
structure RegularPentagonInCircle where
  /-- The circle in which the pentagon is inscribed -/
  circle : Real
  /-- The regular pentagon inscribed in the circle -/
  pentagon : Real
  /-- The sides of the pentagon divide the circle into five equal arcs -/
  equal_arcs : pentagon = 5

/-- The sum of angles inscribed in the five arcs cut off by the sides of a regular pentagon inscribed in a circle -/
def sum_of_inscribed_angles (p : RegularPentagonInCircle) : Real :=
  sorry

/-- Theorem: The sum of angles inscribed in the five arcs cut off by the sides of a regular pentagon inscribed in a circle is 180° -/
theorem sum_of_inscribed_angles_is_180 (p : RegularPentagonInCircle) :
  sum_of_inscribed_angles p = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_inscribed_angles_is_180_l932_93227


namespace NUMINAMATH_CALUDE_ratio_y_to_x_l932_93236

theorem ratio_y_to_x (x y z : ℝ) : 
  (0.6 * (x - y) = 0.4 * (x + y) + 0.3 * (x - 3 * z)) →
  (∃ k : ℤ, z = k * y) →
  (z = 7 * y) →
  (y = 5 * x / 7) →
  y / x = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_y_to_x_l932_93236


namespace NUMINAMATH_CALUDE_prom_attendance_l932_93228

/-- The number of students who attended the prom on their own -/
def solo_students : ℕ := 3

/-- The number of couples who came to the prom -/
def couples : ℕ := 60

/-- The total number of students who attended the prom -/
def total_students : ℕ := solo_students + 2 * couples

/-- Theorem: The total number of students who attended the prom is 123 -/
theorem prom_attendance : total_students = 123 := by
  sorry

end NUMINAMATH_CALUDE_prom_attendance_l932_93228


namespace NUMINAMATH_CALUDE_parallelogram_base_l932_93245

/-- The base of a parallelogram given its area and height -/
theorem parallelogram_base (area height base : ℝ) (h1 : area = 648) (h2 : height = 18) 
    (h3 : area = base * height) : base = 36 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l932_93245


namespace NUMINAMATH_CALUDE_power_function_through_fixed_point_l932_93252

-- Define the fixed point
def P : ℝ × ℝ := (4, 2)

-- Define the power function
def f (x : ℝ) : ℝ := x^(1/2)

-- State the theorem
theorem power_function_through_fixed_point :
  f P.1 = P.2 ∧ ∀ x > 0, f x = Real.sqrt x := by sorry

end NUMINAMATH_CALUDE_power_function_through_fixed_point_l932_93252


namespace NUMINAMATH_CALUDE_absolute_value_square_sum_zero_l932_93209

theorem absolute_value_square_sum_zero (x y : ℝ) :
  |x + 2| + (y - 1)^2 = 0 → x = -2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_sum_zero_l932_93209


namespace NUMINAMATH_CALUDE_angle_between_perpendicular_lines_in_dihedral_l932_93250

-- Define the dihedral angle
def dihedral_angle (α l β : Line3) : ℝ := sorry

-- Define perpendicularity between a line and a plane
def perpendicular (m : Line3) (α : Plane3) : Prop := sorry

-- Define the angle between two lines
def angle_between_lines (m n : Line3) : ℝ := sorry

-- Main theorem
theorem angle_between_perpendicular_lines_in_dihedral 
  (α l β : Line3) (m n : Line3) :
  dihedral_angle α l β = 60 →
  m ≠ n →
  perpendicular m α →
  perpendicular n β →
  angle_between_lines m n = 60 :=
sorry

end NUMINAMATH_CALUDE_angle_between_perpendicular_lines_in_dihedral_l932_93250


namespace NUMINAMATH_CALUDE_square_area_proof_l932_93294

theorem square_area_proof (x : ℝ) :
  (6 * x - 27 = 30 - 2 * x) →
  (6 * x - 27) ^ 2 = 248.0625 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l932_93294


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_three_l932_93291

theorem sum_of_squares_divisible_by_three (a b c : ℤ) 
  (ha : ¬ 3 ∣ a) (hb : ¬ 3 ∣ b) (hc : ¬ 3 ∣ c) : 
  3 ∣ (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_three_l932_93291


namespace NUMINAMATH_CALUDE_zilla_savings_calculation_l932_93235

def monthly_savings (total_earnings : ℝ) (rent_amount : ℝ) : ℝ :=
  let after_tax := total_earnings * 0.9
  let rent_percent := rent_amount / after_tax
  let groceries := after_tax * 0.3
  let entertainment := after_tax * 0.2
  let transportation := after_tax * 0.12
  let total_expenses := rent_amount + groceries + entertainment + transportation
  let remaining := after_tax - total_expenses
  remaining * 0.15

theorem zilla_savings_calculation (total_earnings : ℝ) (h1 : total_earnings > 0) 
  (h2 : monthly_savings total_earnings 133 = 77.52) : 
  ∃ (e : ℝ), e = total_earnings ∧ monthly_savings e 133 = 77.52 :=
by
  sorry

#eval monthly_savings 1900 133

end NUMINAMATH_CALUDE_zilla_savings_calculation_l932_93235


namespace NUMINAMATH_CALUDE_gilda_remaining_marbles_l932_93282

/-- The percentage of marbles Gilda has left after giving away to her friends and family -/
def gildasRemainingMarbles : ℝ :=
  let initialMarbles := 100
  let afterPedro := initialMarbles * (1 - 0.30)
  let afterEbony := afterPedro * (1 - 0.05)
  let afterJimmy := afterEbony * (1 - 0.30)
  let afterTina := afterJimmy * (1 - 0.10)
  afterTina

/-- Theorem stating that Gilda has 41.895% of her original marbles left -/
theorem gilda_remaining_marbles :
  ∀ ε > 0, |gildasRemainingMarbles - 41.895| < ε :=
sorry

end NUMINAMATH_CALUDE_gilda_remaining_marbles_l932_93282


namespace NUMINAMATH_CALUDE_pages_read_l932_93223

/-- Given that Tom read a certain number of chapters in a book with a fixed number of pages per chapter,
    prove that the total number of pages read is equal to the product of chapters and pages per chapter. -/
theorem pages_read (chapters : ℕ) (pages_per_chapter : ℕ) (h1 : chapters = 20) (h2 : pages_per_chapter = 15) :
  chapters * pages_per_chapter = 300 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_l932_93223


namespace NUMINAMATH_CALUDE_abs_neg_reciprocal_2023_l932_93214

theorem abs_neg_reciprocal_2023 : |-1 / 2023| = 1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_reciprocal_2023_l932_93214


namespace NUMINAMATH_CALUDE_special_triangle_longest_altitudes_sum_l932_93279

/-- A triangle with sides 8, 15, and 17 -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 8
  hb : b = 15
  hc : c = 17

/-- The sum of the lengths of the two longest altitudes in the special triangle -/
def longestAltitudesSum (t : SpecialTriangle) : ℝ := 23

/-- Theorem stating that the sum of the lengths of the two longest altitudes
    in the special triangle is 23 -/
theorem special_triangle_longest_altitudes_sum (t : SpecialTriangle) :
  longestAltitudesSum t = 23 := by sorry

end NUMINAMATH_CALUDE_special_triangle_longest_altitudes_sum_l932_93279


namespace NUMINAMATH_CALUDE_correct_road_determination_l932_93247

/-- Represents the two tribes on the island -/
inductive Tribe
| TruthTeller
| Liar

/-- Represents the possible roads -/
inductive Road
| ToVillage
| AwayFromVillage

/-- Represents the possible answers to a question -/
inductive Answer
| Yes
| No

/-- The actual state of the road -/
def actual_road : Road := sorry

/-- The tribe of the islander being asked -/
def islander_tribe : Tribe := sorry

/-- Function that determines how a member of a given tribe would answer a direct question about the road -/
def direct_answer (t : Tribe) (r : Road) : Answer := sorry

/-- Function that determines how an islander would answer the traveler's question -/
def islander_answer (t : Tribe) (r : Road) : Answer := sorry

/-- The traveler's interpretation of the islander's answer -/
def traveler_interpretation (a : Answer) : Road := sorry

theorem correct_road_determination :
  traveler_interpretation (islander_answer islander_tribe actual_road) = actual_road := by sorry

end NUMINAMATH_CALUDE_correct_road_determination_l932_93247


namespace NUMINAMATH_CALUDE_irrational_among_given_numbers_l932_93259

theorem irrational_among_given_numbers : 
  (∃ (q : ℚ), (1 : ℝ) / 2 = ↑q) ∧ 
  (∃ (q : ℚ), (1 : ℝ) / 3 = ↑q) ∧ 
  (∃ (q : ℚ), Real.sqrt 4 = ↑q) ∧ 
  (∀ (q : ℚ), Real.sqrt 5 ≠ ↑q) := by
  sorry

end NUMINAMATH_CALUDE_irrational_among_given_numbers_l932_93259


namespace NUMINAMATH_CALUDE_tower_heights_count_l932_93237

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : Nat
  width : Nat
  height : Nat

/-- Represents the possible orientations of a brick -/
inductive BrickOrientation
  | Length
  | Width
  | Height

/-- Calculates the number of different tower heights achievable -/
def calculateTowerHeights (brickDimensions : BrickDimensions) (totalBricks : Nat) : Nat :=
  sorry

/-- Theorem stating the number of different tower heights achievable -/
theorem tower_heights_count (brickDimensions : BrickDimensions) 
  (h1 : brickDimensions.length = 3)
  (h2 : brickDimensions.width = 12)
  (h3 : brickDimensions.height = 20)
  (h4 : totalBricks = 100) :
  calculateTowerHeights brickDimensions totalBricks = 187 := by
    sorry

end NUMINAMATH_CALUDE_tower_heights_count_l932_93237


namespace NUMINAMATH_CALUDE_b_contribution_is_90000_l932_93296

/-- Represents the business partnership between A and B --/
structure Partnership where
  a_investment : ℕ  -- A's initial investment
  b_join_time : ℕ  -- Time when B joins (in months)
  total_time : ℕ   -- Total investment period (in months)
  profit_ratio_a : ℕ  -- A's part in profit ratio
  profit_ratio_b : ℕ  -- B's part in profit ratio

/-- Calculates B's contribution given the partnership details --/
def calculate_b_contribution (p : Partnership) : ℕ :=
  -- Placeholder for the actual calculation
  0

/-- Theorem stating that B's contribution is 90000 given the specific partnership details --/
theorem b_contribution_is_90000 :
  let p : Partnership := {
    a_investment := 35000,
    b_join_time := 5,
    total_time := 12,
    profit_ratio_a := 2,
    profit_ratio_b := 3
  }
  calculate_b_contribution p = 90000 := by
  sorry


end NUMINAMATH_CALUDE_b_contribution_is_90000_l932_93296


namespace NUMINAMATH_CALUDE_investment_solution_l932_93226

def investment_problem (amount_A : ℝ) : Prop :=
  let yield_A : ℝ := 0.30
  let yield_B : ℝ := 0.50
  let amount_B : ℝ := 200
  (amount_A * (1 + yield_A)) = (amount_B * (1 + yield_B) + 90)

theorem investment_solution : 
  ∃ (amount_A : ℝ), investment_problem amount_A ∧ amount_A = 300 := by
  sorry

end NUMINAMATH_CALUDE_investment_solution_l932_93226


namespace NUMINAMATH_CALUDE_arrangement_theorem_l932_93264

/-- The number of ways to arrange 3 male and 2 female students in a row with females not at ends -/
def arrangement_count : ℕ := sorry

/-- There are 3 male students -/
def male_count : ℕ := 3

/-- There are 2 female students -/
def female_count : ℕ := 2

/-- Total number of students -/
def total_students : ℕ := male_count + female_count

/-- Number of positions where female students can stand (not at ends) -/
def female_positions : ℕ := total_students - 2

theorem arrangement_theorem : arrangement_count = 36 := by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l932_93264


namespace NUMINAMATH_CALUDE_smallest_n_is_eight_l932_93216

/-- A geometric sequence (a_n) with given conditions -/
def geometric_sequence (x : ℝ) (a : ℕ → ℝ) : Prop :=
  x > 0 ∧
  a 1 = Real.exp x ∧
  a 2 = x ∧
  a 3 = Real.log x ∧
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = r * a n

/-- The smallest n for which a_n = 2x is 8 -/
theorem smallest_n_is_eight (x : ℝ) (a : ℕ → ℝ) 
  (h : geometric_sequence x a) : 
  (∃ n : ℕ, n ≥ 1 ∧ a n = 2 * x) ∧ 
  (∀ m : ℕ, m ≥ 1 ∧ m < 8 → a m ≠ 2 * x) ∧ 
  a 8 = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_is_eight_l932_93216


namespace NUMINAMATH_CALUDE_last_four_average_l932_93292

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 62 →
  (list.take 3).sum / 3 = 58 →
  (list.drop 3).sum / 4 = 65 := by
sorry

end NUMINAMATH_CALUDE_last_four_average_l932_93292


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l932_93275

/-- Given a positive integer, returns the number obtained by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ := sorry

theorem divisibility_by_eleven (A : ℕ) (h : A > 0) :
  let B := reverse_digits A
  (11 ∣ (A + B)) ∨ (11 ∣ (A - B)) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l932_93275


namespace NUMINAMATH_CALUDE_market_purchase_cost_l932_93298

/-- The total cost of buying tomatoes and cabbage -/
def total_cost (a b : ℝ) : ℝ :=
  30 * a + 50 * b

/-- Theorem: The total cost of buying 30 kg of tomatoes at 'a' yuan per kg
    and 50 kg of cabbage at 'b' yuan per kg is 30a + 50b yuan -/
theorem market_purchase_cost (a b : ℝ) :
  total_cost a b = 30 * a + 50 * b := by
  sorry

end NUMINAMATH_CALUDE_market_purchase_cost_l932_93298


namespace NUMINAMATH_CALUDE_andrew_kept_490_stickers_l932_93269

/-- The number of stickers Andrew bought -/
def total_stickers : ℕ := 1500

/-- The number of stickers Daniel received -/
def daniel_stickers : ℕ := 250

/-- The number of stickers Fred received -/
def fred_stickers : ℕ := daniel_stickers + 120

/-- The number of stickers Emily received -/
def emily_stickers : ℕ := (daniel_stickers + fred_stickers) / 2

/-- The number of stickers Gina received -/
def gina_stickers : ℕ := 80

/-- The number of stickers Andrew kept -/
def andrew_stickers : ℕ := total_stickers - (daniel_stickers + fred_stickers + emily_stickers + gina_stickers)

theorem andrew_kept_490_stickers : andrew_stickers = 490 := by
  sorry

end NUMINAMATH_CALUDE_andrew_kept_490_stickers_l932_93269


namespace NUMINAMATH_CALUDE_compound_interest_principal_l932_93240

/-- Given a future value, time, annual interest rate, and compounding frequency,
    calculate the principal amount using the compound interest formula. -/
theorem compound_interest_principal
  (A : ℝ) -- Future value
  (t : ℝ) -- Time in years
  (r : ℝ) -- Annual interest rate (as a decimal)
  (n : ℝ) -- Number of times interest is compounded per year
  (h1 : A = 1000000)
  (h2 : t = 5)
  (h3 : r = 0.08)
  (h4 : n = 4)
  : ∃ P : ℝ, A = P * (1 + r/n)^(n*t) :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_principal_l932_93240


namespace NUMINAMATH_CALUDE_sin_inequality_l932_93241

theorem sin_inequality (θ : Real) (h : 0 < θ ∧ θ < Real.pi) :
  Real.sin θ + (1/2) * Real.sin (2*θ) + (1/3) * Real.sin (3*θ) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_inequality_l932_93241


namespace NUMINAMATH_CALUDE_only_statement3_correct_l932_93221

-- Define the propositions
variable (p q : Prop)

-- Define the four statements
def statement1 : Prop := ∀ (p q : Prop), (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)
def statement2 : Prop := ∀ (p q : Prop), (¬(p ∧ q) → p ∨ q) ∧ ¬(p ∨ q → ¬(p ∧ q))
def statement3 : Prop := ∀ (p q : Prop), (p ∨ q → ¬(¬p)) ∧ ¬(¬(¬p) → p ∨ q)
def statement4 : Prop := ∀ (p q : Prop), (¬p → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → ¬p)

-- Theorem stating that only the third statement is correct
theorem only_statement3_correct :
  ¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4 :=
sorry

end NUMINAMATH_CALUDE_only_statement3_correct_l932_93221


namespace NUMINAMATH_CALUDE_validSquaresCount_l932_93222

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  topLeft : Nat × Nat

/-- Checks if a square contains at least 7 black squares -/
def hasAtLeast7BlackSquares (s : Square) : Bool :=
  sorry

/-- Counts the number of valid squares on the checkerboard -/
def countValidSquares : Nat :=
  sorry

/-- Theorem stating the correct number of valid squares -/
theorem validSquaresCount :
  countValidSquares = 140 := by sorry

end NUMINAMATH_CALUDE_validSquaresCount_l932_93222


namespace NUMINAMATH_CALUDE_min_sum_p_q_l932_93232

theorem min_sum_p_q (p q : ℝ) : 
  0 < p → 0 < q → 
  (∃ x : ℝ, x^2 + p*x + 2*q = 0) → 
  (∃ x : ℝ, x^2 + 2*q*x + p = 0) → 
  6 ≤ p + q ∧ ∃ p₀ q₀ : ℝ, 0 < p₀ ∧ 0 < q₀ ∧ p₀ + q₀ = 6 ∧ 
    (∃ x : ℝ, x^2 + p₀*x + 2*q₀ = 0) ∧ 
    (∃ x : ℝ, x^2 + 2*q₀*x + p₀ = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_p_q_l932_93232


namespace NUMINAMATH_CALUDE_charlie_snowball_count_l932_93211

theorem charlie_snowball_count (lucy_snowballs : ℕ) (charlie_extra : ℕ) 
  (h1 : lucy_snowballs = 19)
  (h2 : charlie_extra = 31) : 
  lucy_snowballs + charlie_extra = 50 := by
  sorry

end NUMINAMATH_CALUDE_charlie_snowball_count_l932_93211


namespace NUMINAMATH_CALUDE_man_downstream_speed_l932_93293

/-- Calculates the downstream speed of a man given his upstream and still water speeds -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem: Given a man's upstream speed of 25 kmph and still water speed of 45 kmph, 
    his downstream speed is 65 kmph -/
theorem man_downstream_speed :
  downstream_speed 25 45 = 65 := by
  sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l932_93293


namespace NUMINAMATH_CALUDE_combined_selling_price_is_3620_l932_93285

def article1_cost : ℝ := 1200
def article2_cost : ℝ := 800
def article3_cost : ℝ := 600

def article1_profit_rate : ℝ := 0.4
def article2_profit_rate : ℝ := 0.3
def article3_profit_rate : ℝ := 0.5

def selling_price (cost : ℝ) (profit_rate : ℝ) : ℝ :=
  cost * (1 + profit_rate)

def combined_selling_price : ℝ :=
  selling_price article1_cost article1_profit_rate +
  selling_price article2_cost article2_profit_rate +
  selling_price article3_cost article3_profit_rate

theorem combined_selling_price_is_3620 :
  combined_selling_price = 3620 := by
  sorry

end NUMINAMATH_CALUDE_combined_selling_price_is_3620_l932_93285


namespace NUMINAMATH_CALUDE_rabbit_carrots_l932_93286

theorem rabbit_carrots (rabbit_carrots_per_hole fox_carrots_per_hole : ℕ)
  (hole_difference : ℕ) :
  rabbit_carrots_per_hole = 5 →
  fox_carrots_per_hole = 7 →
  hole_difference = 6 →
  ∃ (rabbit_holes fox_holes : ℕ),
    rabbit_holes = fox_holes + hole_difference ∧
    rabbit_carrots_per_hole * rabbit_holes = fox_carrots_per_hole * fox_holes ∧
    rabbit_carrots_per_hole * rabbit_holes = 105 :=
by sorry

end NUMINAMATH_CALUDE_rabbit_carrots_l932_93286


namespace NUMINAMATH_CALUDE_sin_300_degrees_l932_93261

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l932_93261


namespace NUMINAMATH_CALUDE_correct_delivery_probability_l932_93203

def num_packages : ℕ := 5
def num_correct : ℕ := 3

theorem correct_delivery_probability :
  (num_packages.choose num_correct * (num_correct.factorial * (num_packages - num_correct).factorial)) /
  num_packages.factorial = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_correct_delivery_probability_l932_93203


namespace NUMINAMATH_CALUDE_circle_tangent_chord_relation_l932_93206

/-- Given a circle O with radius r, prove the relationship between x and y -/
theorem circle_tangent_chord_relation (r : ℝ) (x y : ℝ) : y^2 = x^3 / (2*r - x) :=
  sorry

end NUMINAMATH_CALUDE_circle_tangent_chord_relation_l932_93206


namespace NUMINAMATH_CALUDE_stream_speed_l932_93225

/-- Proves that the speed of the stream is 8 km/hr, given the conditions of the boat's travel --/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 10 →
  downstream_distance = 54 →
  downstream_time = 3 →
  (boat_speed + (downstream_distance / downstream_time - boat_speed)) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l932_93225


namespace NUMINAMATH_CALUDE_a_squared_b_plus_ab_squared_l932_93224

theorem a_squared_b_plus_ab_squared (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a * b = 6) : 
  a^2 * b + a * b^2 = 30 := by
sorry

end NUMINAMATH_CALUDE_a_squared_b_plus_ab_squared_l932_93224


namespace NUMINAMATH_CALUDE_vertices_form_parabola_l932_93290

/-- The set of vertices of a family of parabolas forms a parabola -/
theorem vertices_form_parabola (a c d : ℝ) (ha : a > 0) (hc : c > 0) (hd : d > 0) :
  ∃ (f : ℝ → ℝ × ℝ), ∀ (b : ℝ),
    let (x, y) := f b
    (∀ t, y = a * t^2 + b * t + c * t + d → (x - t) * (2 * a * t + b + c) = 0) ∧
    y = -a * x^2 + d :=
  sorry

end NUMINAMATH_CALUDE_vertices_form_parabola_l932_93290


namespace NUMINAMATH_CALUDE_function_symmetry_l932_93231

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x - b * Real.cos x

theorem function_symmetry 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (h_sym : ∀ x : ℝ, f a b (π/4 + x) = f a b (π/4 - x)) :
  let y := fun x => f a b (3*π/4 - x)
  (∀ x : ℝ, y (-x) = -y x) ∧ 
  (∀ x : ℝ, y (2*π - x) = y x) := by
sorry

end NUMINAMATH_CALUDE_function_symmetry_l932_93231


namespace NUMINAMATH_CALUDE_smallest_modulus_z_l932_93234

theorem smallest_modulus_z (z : ℂ) (h : 3 * Complex.abs (z - 8) + 2 * Complex.abs (z - Complex.I * 7) = 26) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ 3 * Complex.abs (w - 8) + 2 * Complex.abs (w - Complex.I * 7) = 26 ∧ Complex.abs w = 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_modulus_z_l932_93234


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l932_93268

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  n % 5 = 3 ∧ 
  n % 7 = 4 ∧ 
  n % 4 = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l932_93268


namespace NUMINAMATH_CALUDE_difference_2010th_2008th_triangular_l932_93274

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_2010th_2008th_triangular : 
  triangular_number 2010 - triangular_number 2008 = 4019 := by
  sorry

end NUMINAMATH_CALUDE_difference_2010th_2008th_triangular_l932_93274


namespace NUMINAMATH_CALUDE_unique_solution_condition_l932_93295

theorem unique_solution_condition (p q : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + p = q * x - 2) ↔ q ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l932_93295


namespace NUMINAMATH_CALUDE_octagon_diagonals_l932_93230

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: An octagon has 20 internal diagonals -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l932_93230


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l932_93220

theorem distance_to_x_axis (P : ℝ × ℝ) (h : P = (-4, 1)) : 
  |P.2| = 1 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l932_93220


namespace NUMINAMATH_CALUDE_chloe_profit_l932_93263

/-- Calculates the profit from selling chocolate-dipped strawberries -/
def strawberry_profit (cost_per_dozen : ℚ) (price_per_half_dozen : ℚ) (dozens_sold : ℕ) : ℚ :=
  let profit_per_half_dozen := price_per_half_dozen - (cost_per_dozen / 2)
  let total_half_dozens := dozens_sold * 2
  profit_per_half_dozen * total_half_dozens

/-- Theorem: Chloe's profit from selling chocolate-dipped strawberries is $500 -/
theorem chloe_profit :
  strawberry_profit 50 30 50 = 500 := by
  sorry

end NUMINAMATH_CALUDE_chloe_profit_l932_93263


namespace NUMINAMATH_CALUDE_sector_area_90_degrees_l932_93205

/-- The area of a sector with radius 2 and central angle 90° is π. -/
theorem sector_area_90_degrees : 
  let r : ℝ := 2
  let angle : ℝ := 90
  let sector_area := (angle / 360) * π * r^2
  sector_area = π := by sorry

end NUMINAMATH_CALUDE_sector_area_90_degrees_l932_93205


namespace NUMINAMATH_CALUDE_gardener_mowing_time_l932_93251

theorem gardener_mowing_time (B : ℝ) (together : ℝ) (h1 : B = 5) (h2 : together = 1.875) :
  ∃ A : ℝ, A = 3 ∧ 1 / A + 1 / B = 1 / together :=
by sorry

end NUMINAMATH_CALUDE_gardener_mowing_time_l932_93251


namespace NUMINAMATH_CALUDE_tangency_points_coordinates_l932_93277

/-- The coordinates of points of tangency to the discriminant parabola -/
theorem tangency_points_coordinates (p q : ℝ) :
  let parabola := {(x, y) : ℝ × ℝ | x^2 - 4*y = 0}
  let tangent_point := (p, q)
  ∃ (p₀ q₀ : ℝ), (p₀, q₀) ∈ parabola ∧
    (p₀ = p + Real.sqrt (p^2 - 4*q) ∨ p₀ = p - Real.sqrt (p^2 - 4*q)) ∧
    q₀ = (p^2 - 2*q + p * Real.sqrt (p^2 - 4*q)) / 2 ∨
    q₀ = (p^2 - 2*q - p * Real.sqrt (p^2 - 4*q)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_tangency_points_coordinates_l932_93277


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l932_93200

theorem square_sum_given_difference_and_product (a b : ℝ) 
  (h1 : a - b = 8) (h2 : a * b = 20) : a^2 + b^2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l932_93200


namespace NUMINAMATH_CALUDE_direct_inverse_variation_l932_93202

theorem direct_inverse_variation (c R₁ R₂ S₁ S₂ T₁ T₂ : ℚ) 
  (h1 : R₁ = c * (S₁ / T₁))
  (h2 : R₂ = c * (S₂ / T₂))
  (h3 : R₁ = 2)
  (h4 : T₁ = 1/2)
  (h5 : S₁ = 8)
  (h6 : R₂ = 16)
  (h7 : T₂ = 1/4) :
  S₂ = 32 := by
sorry

end NUMINAMATH_CALUDE_direct_inverse_variation_l932_93202


namespace NUMINAMATH_CALUDE_interest_rate_problem_l932_93281

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Problem statement -/
theorem interest_rate_problem (principal interest time : ℚ) 
  (h1 : principal = 2000)
  (h2 : interest = 500)
  (h3 : time = 2)
  (h4 : simple_interest principal (12.5 : ℚ) time = interest) :
  12.5 = (interest * 100) / (principal * time) := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l932_93281

import Mathlib

namespace NUMINAMATH_CALUDE_tangent_slope_at_2_l3894_389429

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 3*x

-- State the theorem
theorem tangent_slope_at_2 :
  (deriv f) 2 = 7 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_2_l3894_389429


namespace NUMINAMATH_CALUDE_jake_arrival_time_l3894_389489

-- Define the problem parameters
def floors : ℕ := 9
def steps_per_floor : ℕ := 30
def jake_steps_per_second : ℕ := 3
def elevator_time : ℕ := 60  -- in seconds

-- Calculate the total number of steps
def total_steps : ℕ := floors * steps_per_floor

-- Calculate Jake's descent time
def jake_time : ℕ := total_steps / jake_steps_per_second

-- Define the theorem
theorem jake_arrival_time :
  jake_time - elevator_time = 30 := by sorry

end NUMINAMATH_CALUDE_jake_arrival_time_l3894_389489


namespace NUMINAMATH_CALUDE_first_player_wins_l3894_389472

-- Define the chessboard as a type
def Chessboard : Type := Unit

-- Define a position on the chessboard
def Position : Type := Nat × Nat

-- Define a move as a function from one position to another
def Move : Type := Position → Position

-- Define the property of a move being valid
def ValidMove (m : Move) (visited : Set Position) : Prop :=
  ∀ p, p ∉ visited → 
    (m p).1 = p.1 ∧ ((m p).2 = p.2 + 1 ∨ (m p).2 = p.2 - 1) ∨
    (m p).2 = p.2 ∧ ((m p).1 = p.1 + 1 ∨ (m p).1 = p.1 - 1)

-- Define the game state
structure GameState :=
  (position : Position)
  (visited : Set Position)

-- Define the property of a player having a winning strategy
def HasWinningStrategy (player : Nat) : Prop :=
  ∀ (state : GameState),
    ∃ (m : Move), ValidMove m state.visited →
      ¬∃ (m' : Move), ValidMove m' (insert (m state.position) state.visited)

-- Theorem statement
theorem first_player_wins :
  HasWinningStrategy 0 :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l3894_389472


namespace NUMINAMATH_CALUDE_bicycle_trip_time_l3894_389462

theorem bicycle_trip_time (adam_speed simon_speed separation_distance : ℝ) 
  (adam_speed_pos : adam_speed > 0)
  (simon_speed_pos : simon_speed > 0)
  (separation_distance_pos : separation_distance > 0)
  (h_adam : adam_speed = 12)
  (h_simon : simon_speed = 9)
  (h_separation : separation_distance = 90) : 
  ∃ t : ℝ, t > 0 ∧ t * (adam_speed ^ 2 + simon_speed ^ 2) ^ (1/2 : ℝ) = separation_distance ∧ t = 6 :=
sorry

end NUMINAMATH_CALUDE_bicycle_trip_time_l3894_389462


namespace NUMINAMATH_CALUDE_cubic_factorization_l3894_389445

theorem cubic_factorization (m : ℝ) : m^3 - 4*m = m*(m - 2)*(m + 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3894_389445


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l3894_389436

theorem triangle_side_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  let A : ℝ := 2 * Real.pi / 3
  a^2 = 2*b*c + 3*c^2 →
  c / b = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l3894_389436


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3894_389469

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (x + 3) / (k * x - 2) = x) ↔ k = -3/4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3894_389469


namespace NUMINAMATH_CALUDE_gcd_lcm_1729_867_l3894_389476

theorem gcd_lcm_1729_867 :
  (Nat.gcd 1729 867 = 1) ∧ (Nat.lcm 1729 867 = 1499003) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_1729_867_l3894_389476


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l3894_389406

theorem quadratic_root_condition (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ x₂ > 1 ∧ 
    3 * x₁^2 + a * (a - 6) * x₁ - 3 = 0 ∧ 
    3 * x₂^2 + a * (a - 6) * x₂ - 3 = 0) ↔ 
  (0 < a ∧ a < 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l3894_389406


namespace NUMINAMATH_CALUDE_retail_price_calculation_l3894_389428

def wholesale_price : ℝ := 90

def discount_rate : ℝ := 0.10

def profit_rate : ℝ := 0.20

def retail_price : ℝ := 120

theorem retail_price_calculation :
  let profit := profit_rate * wholesale_price
  let selling_price := wholesale_price + profit
  selling_price = retail_price * (1 - discount_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l3894_389428


namespace NUMINAMATH_CALUDE_one_and_one_third_of_number_is_48_l3894_389446

theorem one_and_one_third_of_number_is_48 :
  ∃ x : ℚ, (4 / 3) * x = 48 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_one_and_one_third_of_number_is_48_l3894_389446


namespace NUMINAMATH_CALUDE_cone_base_diameter_l3894_389403

theorem cone_base_diameter (r : ℝ) (h1 : r > 0) : 
  (π * r^2 + π * r * (2 * r) = 3 * π) → 
  (2 * r = 2) := by
sorry

end NUMINAMATH_CALUDE_cone_base_diameter_l3894_389403


namespace NUMINAMATH_CALUDE_jerrys_age_l3894_389480

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 30 →
  mickey_age = 4 * jerry_age + 10 →
  jerry_age = 5 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l3894_389480


namespace NUMINAMATH_CALUDE_mayoral_election_votes_l3894_389407

theorem mayoral_election_votes (Z : ℕ) (hZ : Z = 25000) :
  let Y := (3 / 5 : ℚ) * Z
  let X := (8 / 5 : ℚ) * Y
  X = 24000 := by
  sorry

end NUMINAMATH_CALUDE_mayoral_election_votes_l3894_389407


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3894_389486

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane 
  (α β : Plane) (m : Line)
  (h1 : subset m α)
  (h2 : parallel α β) :
  line_parallel m β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3894_389486


namespace NUMINAMATH_CALUDE_milkshake_hours_l3894_389496

/-- Given that Augustus makes 3 milkshakes per hour and Luna makes 7 milkshakes per hour,
    prove that they have been making milkshakes for 8 hours when they have made 80 milkshakes in total. -/
theorem milkshake_hours (augustus_rate : ℕ) (luna_rate : ℕ) (total_milkshakes : ℕ) (hours : ℕ) :
  augustus_rate = 3 →
  luna_rate = 7 →
  total_milkshakes = 80 →
  augustus_rate * hours + luna_rate * hours = total_milkshakes →
  hours = 8 := by
sorry

end NUMINAMATH_CALUDE_milkshake_hours_l3894_389496


namespace NUMINAMATH_CALUDE_factorial_sum_units_digit_l3894_389408

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_units_digit :
  ∀ n ≥ 99, units_digit (factorial_sum n) = 7 :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_units_digit_l3894_389408


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3894_389405

theorem binomial_expansion_coefficient (n : ℕ) : 
  (3^2 * (n.choose 2) = 54) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3894_389405


namespace NUMINAMATH_CALUDE_fraction_to_decimal_decimal_representation_three_twentieths_decimal_l3894_389430

theorem fraction_to_decimal :
  (3 : ℚ) / 20 = (15 : ℚ) / 100 := by sorry

theorem decimal_representation :
  (15 : ℚ) / 100 = 0.15 := by sorry

theorem three_twentieths_decimal :
  (3 : ℚ) / 20 = 0.15 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_decimal_representation_three_twentieths_decimal_l3894_389430


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l3894_389424

/-- The area of a triangle with side lengths 15, 36, and 39 is 270 -/
theorem triangle_area : ℝ → Prop :=
  fun a => ∀ s₁ s₂ s₃ : ℝ,
    s₁ = 15 ∧ s₂ = 36 ∧ s₃ = 39 →
    (∃ A : ℝ, A = a ∧ A = 270)

/-- Proof of the theorem -/
theorem triangle_area_proof : triangle_area 270 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l3894_389424


namespace NUMINAMATH_CALUDE_class_mean_calculation_l3894_389487

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group2_students : ℕ) 
  (group1_mean : ℚ) (group2_mean : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 40 →
  group2_students = 10 →
  group1_mean = 85/100 →
  group2_mean = 80/100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 84/100 := by
sorry

#eval (40 * (85/100) + 10 * (80/100)) / 50

end NUMINAMATH_CALUDE_class_mean_calculation_l3894_389487


namespace NUMINAMATH_CALUDE_cylindrical_to_rectangular_conversion_l3894_389423

theorem cylindrical_to_rectangular_conversion :
  let r : ℝ := 5
  let θ : ℝ := π / 3
  let z : ℝ := 2
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y, z) = (2.5, 5 * Real.sqrt 3 / 2, 2) :=
by sorry

end NUMINAMATH_CALUDE_cylindrical_to_rectangular_conversion_l3894_389423


namespace NUMINAMATH_CALUDE_articles_count_l3894_389497

/-- 
Given:
- The selling price is double the cost price
- The cost price of X articles equals the selling price of 25 articles
Prove that X = 50
-/
theorem articles_count (cost_price selling_price : ℝ) (X : ℕ) 
  (h1 : selling_price = 2 * cost_price) 
  (h2 : X * cost_price = 25 * selling_price) : 
  X = 50 := by
  sorry

end NUMINAMATH_CALUDE_articles_count_l3894_389497


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l3894_389401

/-- The number of ways to arrange 7 people with specific adjacency conditions -/
def arrangement_count : ℕ := 960

/-- Proves that the number of arrangements is correct -/
theorem correct_arrangement_count : arrangement_count = 960 := by
  sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l3894_389401


namespace NUMINAMATH_CALUDE_negation_and_converse_of_divisibility_proposition_l3894_389435

def last_digit (n : ℤ) : ℤ := n % 10

theorem negation_and_converse_of_divisibility_proposition :
  (¬ (∀ n : ℤ, (last_digit n = 0 ∨ last_digit n = 5) → n % 5 = 0) ↔ 
   (∃ n : ℤ, (last_digit n = 0 ∨ last_digit n = 5) ∧ n % 5 ≠ 0)) ∧
  ((∀ n : ℤ, n % 5 = 0 → (last_digit n = 0 ∨ last_digit n = 5)) ↔
   (∀ n : ℤ, (last_digit n ≠ 0 ∧ last_digit n ≠ 5) → n % 5 ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_and_converse_of_divisibility_proposition_l3894_389435


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3894_389437

def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_inequality (b c : ℝ) (h : f b c (-1) = f b c 3) :
  f b c 1 < c ∧ c < f b c (-1) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3894_389437


namespace NUMINAMATH_CALUDE_taxi_count_2008_l3894_389442

/-- Represents the number of taxis (in thousands) at the end of a given year -/
def taxiCount : ℕ → ℝ
| 0 => 100  -- End of 2005
| n + 1 => taxiCount n * 1.1 - 20  -- Subsequent years

/-- The year we're interested in (2008 is 3 years after 2005) -/
def targetYear : ℕ := 3

theorem taxi_count_2008 :
  12 ≤ taxiCount targetYear ∧ taxiCount targetYear < 13 := by
  sorry

end NUMINAMATH_CALUDE_taxi_count_2008_l3894_389442


namespace NUMINAMATH_CALUDE_third_term_is_seven_l3894_389467

/-- An arithmetic sequence with general term aₙ = 2n + 1 -/
def a (n : ℕ) : ℝ := 2 * n + 1

/-- The third term of the sequence is 7 -/
theorem third_term_is_seven : a 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_seven_l3894_389467


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l3894_389448

noncomputable def f (x : ℝ) : ℝ := Real.sin 1 - Real.cos x

theorem f_derivative_at_one : 
  deriv f 1 = Real.sin 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l3894_389448


namespace NUMINAMATH_CALUDE_fraction_value_l3894_389465

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3894_389465


namespace NUMINAMATH_CALUDE_f_min_value_l3894_389415

def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

theorem f_min_value : ∀ x : ℝ, f x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l3894_389415


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_a_eq_neg_two_l3894_389463

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of real number a. -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 1)

/-- If z(a) is a pure imaginary number, then a = -2. -/
theorem pure_imaginary_implies_a_eq_neg_two :
  ∀ a : ℝ, is_pure_imaginary (z a) → a = -2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_a_eq_neg_two_l3894_389463


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3894_389440

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 20 → p.Prime → ¬(p ∣ n)

theorem smallest_composite_no_small_factors : 
  (is_composite 529) ∧ 
  (has_no_small_prime_factors 529) ∧ 
  (∀ m : ℕ, m < 529 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l3894_389440


namespace NUMINAMATH_CALUDE_cake_eating_contest_l3894_389483

theorem cake_eating_contest : (7 : ℚ) / 8 - (5 : ℚ) / 6 = (1 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_cake_eating_contest_l3894_389483


namespace NUMINAMATH_CALUDE_speed_increases_with_height_l3894_389427

/-- Represents a data point of height and time -/
structure DataPoint where
  height : ℝ
  time : ℝ

/-- The data set from the experiment -/
def dataSet : List DataPoint := [
  ⟨10, 4.23⟩, ⟨20, 3.00⟩, ⟨30, 2.45⟩, ⟨40, 2.13⟩, 
  ⟨50, 1.89⟩, ⟨60, 1.71⟩, ⟨70, 1.59⟩
]

/-- Theorem stating that average speed increases with height -/
theorem speed_increases_with_height :
  ∀ (d1 d2 : DataPoint), 
    d1 ∈ dataSet → d2 ∈ dataSet →
    d2.height > d1.height → 
    d2.height / d2.time > d1.height / d1.time :=
by sorry

end NUMINAMATH_CALUDE_speed_increases_with_height_l3894_389427


namespace NUMINAMATH_CALUDE_trigonometric_equation_l3894_389411

theorem trigonometric_equation (x : ℝ) :
  (1 / Real.cos (2022 * x) + Real.tan (2022 * x) = 1 / 2022) →
  (1 / Real.cos (2022 * x) - Real.tan (2022 * x) = 2022) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_l3894_389411


namespace NUMINAMATH_CALUDE_simplify_cube_roots_l3894_389404

theorem simplify_cube_roots : (512 : ℝ)^(1/3) * (125 : ℝ)^(1/3) = 40 := by sorry

end NUMINAMATH_CALUDE_simplify_cube_roots_l3894_389404


namespace NUMINAMATH_CALUDE_guest_car_wheels_l3894_389414

/-- Given information about a parking lot situation, prove that each guest car has 4 wheels. -/
theorem guest_car_wheels
  (total_guests : ℕ)
  (guest_cars : ℕ)
  (total_wheels : ℕ)
  (parent_cars : ℕ)
  (h_guests : total_guests = 40)
  (h_guest_cars : guest_cars = 10)
  (h_total_wheels : total_wheels = 48)
  (h_parent_cars : parent_cars = 2)
  : (total_wheels - parent_cars * 4) / guest_cars = 4 := by
  sorry

end NUMINAMATH_CALUDE_guest_car_wheels_l3894_389414


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l3894_389466

/-- Given points A and B on the parabola y = -x^2 forming an equilateral triangle with the origin,
    prove that their x-coordinates are ±√3 and the side length is 2√3. -/
theorem equilateral_triangle_on_parabola :
  ∀ (a : ℝ),
  let A : ℝ × ℝ := (a, -a^2)
  let B : ℝ × ℝ := (-a, -a^2)
  let O : ℝ × ℝ := (0, 0)
  -- Distance between two points (x₁, y₁) and (x₂, y₂)
  let dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  -- Condition for equilateral triangle
  (dist A O = dist B O ∧ dist A O = dist A B) →
  (a = Real.sqrt 3 ∧ dist A O = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l3894_389466


namespace NUMINAMATH_CALUDE_exp_addition_property_l3894_389456

open Real

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem exp_addition_property (x y : ℝ) : f (x + y) = f x * f y := by
  sorry

end NUMINAMATH_CALUDE_exp_addition_property_l3894_389456


namespace NUMINAMATH_CALUDE_unique_perpendicular_line_l3894_389432

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a plane type
structure Plane where
  points : Set Point

-- Define what it means for a point to be on a line
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define what it means for two lines to be perpendicular
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- State the theorem
theorem unique_perpendicular_line 
  (plane : Plane) (l : Line) (p : Point) 
  (h : ¬ p.onLine l) : 
  ∃! l_perp : Line, 
    l_perp.perpendicular l ∧ 
    p.onLine l_perp :=
  sorry

end NUMINAMATH_CALUDE_unique_perpendicular_line_l3894_389432


namespace NUMINAMATH_CALUDE_max_value_of_function_l3894_389410

theorem max_value_of_function (x : ℝ) (h : x < -1) :
  ∃ (M : ℝ), M = -3 ∧ ∀ y, y = x + 1 / (x + 1) → y ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3894_389410


namespace NUMINAMATH_CALUDE_parabola_equation_l3894_389479

theorem parabola_equation (a : ℝ) (x₀ : ℝ) : 
  (∃ (x : ℝ → ℝ) (y : ℝ → ℝ), 
    (∀ t, x t ^ 2 = a * y t) ∧ 
    (y x₀ = 2) ∧ 
    ((x x₀ - 0) ^ 2 + (y x₀ - a / 4) ^ 2 = 3 ^ 2)) → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3894_389479


namespace NUMINAMATH_CALUDE_b_minus_a_value_l3894_389412

theorem b_minus_a_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 2) (h3 : a + b > 0) :
  b - a = -1 ∨ b - a = -5 := by
  sorry

end NUMINAMATH_CALUDE_b_minus_a_value_l3894_389412


namespace NUMINAMATH_CALUDE_division_remainder_l3894_389438

theorem division_remainder : 
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := 2 * diff
  let dividend := 220040
  dividend % sum = 40 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_l3894_389438


namespace NUMINAMATH_CALUDE_chord_length_l3894_389450

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

-- Define the line l
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 11 = 0}

-- Define the intersection points A and B
def intersection_points : Set (ℝ × ℝ) :=
  circle_C ∩ line_l

-- Theorem statement
theorem chord_length :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_chord_length_l3894_389450


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_l3894_389453

/-- Given a piece of wood and a rope with unknown lengths, prove that the system of equations
    describing their relationship is correct based on the given measurements. -/
theorem sunzi_wood_measurement (x y : ℝ) : 
  (y - x = 4.5 ∧ y > x) →  -- Full rope measurement
  (x - y / 2 = 1 ∧ x > y / 2) →  -- Half rope measurement
  y - x = 4.5 ∧ x - y / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_l3894_389453


namespace NUMINAMATH_CALUDE_cube_root_nine_thirty_two_squared_l3894_389452

theorem cube_root_nine_thirty_two_squared :
  (((9 : ℝ) / 32) ^ (1/3 : ℝ)) ^ 2 = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_cube_root_nine_thirty_two_squared_l3894_389452


namespace NUMINAMATH_CALUDE_reading_time_proof_l3894_389400

/-- The number of days it took for Ryan and his brother to finish their books -/
def days_to_finish : ℕ := 7

/-- Ryan's total number of pages -/
def ryan_total_pages : ℕ := 2100

/-- Number of pages Ryan's brother reads per day -/
def brother_pages_per_day : ℕ := 200

/-- The difference in pages read per day between Ryan and his brother -/
def page_difference : ℕ := 100

theorem reading_time_proof :
  ryan_total_pages = (brother_pages_per_day + page_difference) * days_to_finish ∧
  ryan_total_pages % (brother_pages_per_day + page_difference) = 0 := by
  sorry

#check reading_time_proof

end NUMINAMATH_CALUDE_reading_time_proof_l3894_389400


namespace NUMINAMATH_CALUDE_farm_water_consumption_l3894_389441

theorem farm_water_consumption 
  (num_cows : ℕ)
  (cow_daily_water : ℕ)
  (sheep_cow_ratio : ℕ)
  (sheep_water_ratio : ℚ)
  (days_in_week : ℕ)
  (h1 : num_cows = 40)
  (h2 : cow_daily_water = 80)
  (h3 : sheep_cow_ratio = 10)
  (h4 : sheep_water_ratio = 1/4)
  (h5 : days_in_week = 7) :
  (num_cows * cow_daily_water * days_in_week) + 
  (num_cows * sheep_cow_ratio * (sheep_water_ratio * cow_daily_water) * days_in_week) = 78400 :=
by sorry

end NUMINAMATH_CALUDE_farm_water_consumption_l3894_389441


namespace NUMINAMATH_CALUDE_prime_divisors_of_50_factorial_l3894_389464

theorem prime_divisors_of_50_factorial (n : ℕ) :
  (n = 50) →
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card =
  (Finset.filter (λ p => p.Prime ∧ p ∣ n!) (Finset.range (n + 1))).card :=
sorry

end NUMINAMATH_CALUDE_prime_divisors_of_50_factorial_l3894_389464


namespace NUMINAMATH_CALUDE_power_function_value_l3894_389477

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x ^ b

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) :
  isPowerFunction f → f (1/2) = 8 → f (-2) = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l3894_389477


namespace NUMINAMATH_CALUDE_roberta_listening_time_l3894_389498

/-- The number of days it takes Roberta to listen to her entire record collection -/
def listen_time (initial_records : ℕ) (gift_records : ℕ) (bought_records : ℕ) (days_per_record : ℕ) : ℕ :=
  (initial_records + gift_records + bought_records) * days_per_record

theorem roberta_listening_time :
  listen_time 8 12 30 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_roberta_listening_time_l3894_389498


namespace NUMINAMATH_CALUDE_max_value_of_function_l3894_389433

theorem max_value_of_function (x : ℝ) (hx : x < 0) : 
  2 * x + 2 / x ≤ -4 ∧ 
  ∃ y : ℝ, y < 0 ∧ 2 * y + 2 / y = -4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3894_389433


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l3894_389488

/-- Represents the number of students in each grade --/
structure Students where
  eighth : ℕ
  seventh : ℕ
  sixth : ℕ

/-- The ratio of 8th-graders to 6th-graders is 5:3 --/
def ratio_8th_to_6th (s : Students) : Prop :=
  5 * s.sixth = 3 * s.eighth

/-- The ratio of 8th-graders to 7th-graders is 8:5 --/
def ratio_8th_to_7th (s : Students) : Prop :=
  8 * s.seventh = 5 * s.eighth

/-- The total number of students --/
def total_students (s : Students) : ℕ :=
  s.eighth + s.seventh + s.sixth

/-- The main theorem: The smallest possible number of students is 89 --/
theorem smallest_number_of_students :
  ∃ (s : Students), ratio_8th_to_6th s ∧ ratio_8th_to_7th s ∧
  (∀ (t : Students), ratio_8th_to_6th t ∧ ratio_8th_to_7th t →
    total_students s ≤ total_students t) ∧
  total_students s = 89 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l3894_389488


namespace NUMINAMATH_CALUDE_sin_polar_complete_circle_l3894_389425

open Real

theorem sin_polar_complete_circle (t : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = sin θ) →
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → sin θ = sin (θ + t)) →
  t = 2 * π :=
sorry

end NUMINAMATH_CALUDE_sin_polar_complete_circle_l3894_389425


namespace NUMINAMATH_CALUDE_factor_sum_l3894_389478

theorem factor_sum (a b : ℤ) : 
  (∀ x, 25 * x^2 - 130 * x - 120 = (5 * x + a) * (5 * x + b)) →
  a + 3 * b = -86 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l3894_389478


namespace NUMINAMATH_CALUDE_fraction_division_equality_l3894_389468

theorem fraction_division_equality : 
  (-1/42) / (1/6 - 3/14 + 2/3 - 2/7) = -1/14 := by sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l3894_389468


namespace NUMINAMATH_CALUDE_cube_side_length_l3894_389482

/-- Given a cube where the length of its space diagonal is 6.92820323027551 m,
    prove that the side length of the cube is 4 m. -/
theorem cube_side_length (d : ℝ) (h : d = 6.92820323027551) : 
  ∃ (a : ℝ), a * Real.sqrt 3 = d ∧ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_l3894_389482


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3894_389484

/-- 
Given a rectangular field with one side of length 20 feet and a perimeter 
(excluding that side) of 85 feet, the area of the field is 650 square feet.
-/
theorem rectangular_field_area : 
  ∀ (length width : ℝ), 
    length = 20 →
    2 * width + length = 85 →
    length * width = 650 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3894_389484


namespace NUMINAMATH_CALUDE_vowel_writing_count_l3894_389447

theorem vowel_writing_count (num_vowels : ℕ) (total_alphabets : ℕ) : 
  num_vowels = 5 → 
  total_alphabets = 10 → 
  ∃ (times_written : ℕ), times_written * num_vowels = total_alphabets ∧ times_written = 2 :=
by sorry

end NUMINAMATH_CALUDE_vowel_writing_count_l3894_389447


namespace NUMINAMATH_CALUDE_square_roots_equation_l3894_389451

theorem square_roots_equation (a b : ℝ) :
  let f (x : ℝ) := a * b * x^2 - (a + b) * x + 1
  let g (x : ℝ) := a^2 * b^2 * x^2 - (a^2 + b^2) * x + 1
  ∀ (r : ℝ), f r = 0 → g (r^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_equation_l3894_389451


namespace NUMINAMATH_CALUDE_solution_difference_l3894_389455

theorem solution_difference (r s : ℝ) : 
  r ≠ s ∧ 
  (r - 5) * (r + 5) = 25 * r - 125 ∧
  (s - 5) * (s + 5) = 25 * s - 125 ∧
  r > s →
  r - s = 15 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l3894_389455


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3894_389475

/-- A hyperbola with equation y²/2 - x²/4 = 1 -/
def hyperbola (x y : ℝ) : Prop := y^2 / 2 - x^2 / 4 = 1

/-- The reference hyperbola with equation x²/2 - y² = 1 -/
def reference_hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

theorem hyperbola_properties :
  (∃ (x y : ℝ), hyperbola x y ∧ x = 2 ∧ y = -2) ∧
  (∀ (x y : ℝ), ∃ (k : ℝ), hyperbola x y ↔ reference_hyperbola (x * k) (y * k)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3894_389475


namespace NUMINAMATH_CALUDE_larger_number_proof_l3894_389494

theorem larger_number_proof (a b : ℕ) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 3300) (h3 : a > b) : a = 300 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3894_389494


namespace NUMINAMATH_CALUDE_geometric_progression_terms_l3894_389492

-- Define the geometric progression
def geometric_progression (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

-- Define the sum of a geometric progression
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

theorem geometric_progression_terms (a : ℚ) :
  (geometric_progression a (1/3) 4 = 1/54) →
  (geometric_sum a (1/3) 5 = 121/162) →
  ∃ n : ℕ, geometric_sum a (1/3) n = 121/162 ∧ n = 5 :=
by
  sorry

#check geometric_progression_terms

end NUMINAMATH_CALUDE_geometric_progression_terms_l3894_389492


namespace NUMINAMATH_CALUDE_two_valid_M_values_l3894_389481

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem two_valid_M_values :
  ∃! (s : Finset ℕ), 
    (∀ M ∈ s, is_two_digit M ∧ 
      let diff := M - reverse_digits M
      diff > 0 ∧ 
      is_perfect_cube diff ∧ 
      27 < diff ∧ 
      diff < 100) ∧
    s.card = 2 := by sorry

end NUMINAMATH_CALUDE_two_valid_M_values_l3894_389481


namespace NUMINAMATH_CALUDE_xyz_product_l3894_389413

theorem xyz_product (x y z : ℚ) 
  (eq1 : x + y + z = 1)
  (eq2 : x + y - z = 2)
  (eq3 : x - y - z = 3) :
  x * y * z = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l3894_389413


namespace NUMINAMATH_CALUDE_walter_fish_fry_guests_l3894_389434

-- Define the constants from the problem
def hushpuppies_per_guest : ℕ := 5
def hushpuppies_per_batch : ℕ := 10
def minutes_per_batch : ℕ := 8
def total_cooking_time : ℕ := 80

-- Define the function to calculate the number of guests
def number_of_guests : ℕ :=
  (total_cooking_time / minutes_per_batch * hushpuppies_per_batch) / hushpuppies_per_guest

-- State the theorem
theorem walter_fish_fry_guests :
  number_of_guests = 20 :=
sorry

end NUMINAMATH_CALUDE_walter_fish_fry_guests_l3894_389434


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l3894_389409

theorem base_10_to_base_7 (n : ℕ) (h : n = 3589) :
  ∃ (a b c d e : ℕ),
    n = a * 7^4 + b * 7^3 + c * 7^2 + d * 7^1 + e * 7^0 ∧
    a = 1 ∧ b = 3 ∧ c = 3 ∧ d = 1 ∧ e = 5 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l3894_389409


namespace NUMINAMATH_CALUDE_expression_simplification_l3894_389485

theorem expression_simplification : (((2 + 3 + 6 + 7) / 3) + ((3 * 6 + 9) / 4)) = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3894_389485


namespace NUMINAMATH_CALUDE_cube_sum_value_l3894_389420

theorem cube_sum_value (a b R S : ℝ) : 
  a + b = R → 
  a^2 + b^2 = 12 → 
  a^3 + b^3 = S → 
  S = 32 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_value_l3894_389420


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3894_389431

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) → 
  n + (n + 1) = 43 :=
sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3894_389431


namespace NUMINAMATH_CALUDE_project_distribution_theorem_l3894_389443

def number_of_arrangements (total_projects : ℕ) 
                            (company_a_projects : ℕ) 
                            (company_b_projects : ℕ) 
                            (company_c_projects : ℕ) 
                            (company_d_projects : ℕ) : ℕ :=
  (Nat.choose total_projects company_a_projects) * 
  (Nat.choose (total_projects - company_a_projects) company_b_projects) * 
  (Nat.choose (total_projects - company_a_projects - company_b_projects) company_c_projects)

theorem project_distribution_theorem :
  number_of_arrangements 8 3 1 2 2 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_project_distribution_theorem_l3894_389443


namespace NUMINAMATH_CALUDE_lice_check_time_is_three_hours_l3894_389439

/-- The total time required to check all students for lice -/
def total_check_time (kindergarteners first_graders second_graders third_graders : ℕ) 
  (check_time_per_student : ℕ) : ℚ :=
  let total_students := kindergarteners + first_graders + second_graders + third_graders
  let total_minutes := total_students * check_time_per_student
  (total_minutes : ℚ) / 60

/-- Theorem stating that the total check time for the given number of students is 3 hours -/
theorem lice_check_time_is_three_hours :
  total_check_time 26 19 20 25 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lice_check_time_is_three_hours_l3894_389439


namespace NUMINAMATH_CALUDE_set_equality_implies_difference_l3894_389454

theorem set_equality_implies_difference (a b : ℝ) :
  ({0, b/a, b} : Set ℝ) = {1, a+b, a} → b - a = 2 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_difference_l3894_389454


namespace NUMINAMATH_CALUDE_special_function_inequality_l3894_389419

/-- A function satisfying the given properties in the problem -/
structure SpecialFunction where
  f : ℝ → ℝ
  odd : ∀ x, f (-x) = -f x
  special_property : ∀ x, f (1 + x) + f (1 - x) = f 1
  decreasing_on_unit_interval : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f y < f x

/-- The main theorem to be proved -/
theorem special_function_inequality (sf : SpecialFunction) :
  sf.f (-2 + Real.sqrt 2 / 2) < -sf.f (10 / 3) ∧ -sf.f (10 / 3) < sf.f (9 / 2) := by
  sorry


end NUMINAMATH_CALUDE_special_function_inequality_l3894_389419


namespace NUMINAMATH_CALUDE_cd_purchase_total_l3894_389444

/-- The total cost of purchasing 3 copies each of three different CDs -/
def total_cost (price1 price2 price3 : ℕ) : ℕ :=
  3 * (price1 + price2 + price3)

theorem cd_purchase_total : total_cost 100 50 85 = 705 := by
  sorry

end NUMINAMATH_CALUDE_cd_purchase_total_l3894_389444


namespace NUMINAMATH_CALUDE_constraint_implies_equality_and_minimum_value_l3894_389459

open Real

-- Define the constraint function
def constraint (a b c : ℝ) : Prop :=
  exp (a - c) + b * exp (c + 1) ≤ a + log b + 3

-- Define the objective function
def objective (a b c : ℝ) : ℝ :=
  a + b + 2 * c

-- Theorem statement
theorem constraint_implies_equality_and_minimum_value
  (a b c : ℝ) (h : constraint a b c) :
  a = c ∧ ∀ x y z, constraint x y z → objective a b c ≤ objective x y z ∧ objective a b c = -3 * log 3 :=
sorry

end NUMINAMATH_CALUDE_constraint_implies_equality_and_minimum_value_l3894_389459


namespace NUMINAMATH_CALUDE_line_passes_through_quadrants_l3894_389499

/-- A line in the plane defined by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The quadrants of the Cartesian plane -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Predicate to check if a line passes through a quadrant -/
def passes_through (l : Line) (q : Quadrant) : Prop := sorry

/-- Theorem stating that under given conditions, the line passes through specific quadrants -/
theorem line_passes_through_quadrants (l : Line) 
  (h1 : l.A * l.C < 0) (h2 : l.B * l.C < 0) : 
  passes_through l Quadrant.first ∧ 
  passes_through l Quadrant.second ∧ 
  passes_through l Quadrant.fourth :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_quadrants_l3894_389499


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l3894_389421

def f (x a : ℝ) : ℝ := (x + 1) * (x + a)

theorem even_function_implies_a_eq_neg_one (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l3894_389421


namespace NUMINAMATH_CALUDE_visible_part_of_third_mountain_l3894_389470

/-- Represents a mountain with a height and position on a great circle. -/
structure Mountain where
  height : ℝ
  position : ℝ

/-- Represents the Earth as a sphere. -/
structure Earth where
  radius : ℝ

/-- Calculates the visible height of a distant mountain. -/
def visibleHeight (earth : Earth) (m1 m2 m3 : Mountain) : ℝ :=
  sorry

theorem visible_part_of_third_mountain
  (earth : Earth)
  (m1 m2 m3 : Mountain)
  (h_earth_radius : earth.radius = 6366000) -- in meters
  (h_m1_height : m1.height = 2500)
  (h_m2_height : m2.height = 3000)
  (h_m3_height : m3.height = 8800)
  (h_m1_m2_distance : m2.position - m1.position = 1 * π / 180) -- 1 degree in radians
  (h_m2_m3_distance : m3.position - m2.position = 1.5 * π / 180) -- 1.5 degrees in radians
  : visibleHeight earth m1 m2 m3 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_visible_part_of_third_mountain_l3894_389470


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3894_389473

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_M_intersect_N :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3894_389473


namespace NUMINAMATH_CALUDE_paperclips_exceed_200_l3894_389457

def paperclips (k : ℕ) : ℕ := 3 * 2^k

theorem paperclips_exceed_200 : ∀ k : ℕ, paperclips k ≤ 200 ↔ k < 7 := by sorry

end NUMINAMATH_CALUDE_paperclips_exceed_200_l3894_389457


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_quadratic_equation_l3894_389495

theorem arithmetic_geometric_mean_quadratic_equation 
  (a b : ℝ) 
  (h_arithmetic_mean : (a + b) / 2 = 8) 
  (h_geometric_mean : Real.sqrt (a * b) = 15) : 
  ∀ x : ℝ, x^2 - 16*x + 225 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_quadratic_equation_l3894_389495


namespace NUMINAMATH_CALUDE_count_palindrome_pairs_l3894_389449

def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ 
  (n / 1000 = n % 10) ∧ 
  ((n / 100) % 10 = (n / 10) % 10)

def palindrome_pair (p1 p2 : ℕ) : Prop :=
  is_four_digit_palindrome p1 ∧ 
  is_four_digit_palindrome p2 ∧ 
  p1 - p2 = 3674

theorem count_palindrome_pairs : 
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ S ↔ palindrome_pair p.1 p.2) ∧ 
    Finset.card S = 35 := by
  sorry

end NUMINAMATH_CALUDE_count_palindrome_pairs_l3894_389449


namespace NUMINAMATH_CALUDE_smallest_sum_of_roots_l3894_389461

theorem smallest_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + 3*a*x + 4*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 4*b*x + 3*a = 0) :
  a + b ≥ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_roots_l3894_389461


namespace NUMINAMATH_CALUDE_a_range_for_g_three_zeros_l3894_389402

open Real

noncomputable def f (a b x : ℝ) : ℝ := exp x - 2 * (a - 1) * x - b

noncomputable def g (a b x : ℝ) : ℝ := exp x - (a - 1) * x^2 - b * x - 1

theorem a_range_for_g_three_zeros (a b : ℝ) :
  (g a b 1 = 0) →
  (∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < 1 ∧
    g a b x₁ = 0 ∧ g a b x₂ = 0 ∧ g a b x₃ = 0) →
  (e - 1 < a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_a_range_for_g_three_zeros_l3894_389402


namespace NUMINAMATH_CALUDE_distinct_parenthesizations_l3894_389491

-- Define a function to represent exponentiation
def exp (a : ℕ) (b : ℕ) : ℕ := a ^ b

-- Define the five possible parenthesizations
def p1 : ℕ := exp 3 (exp 3 (exp 3 3))
def p2 : ℕ := exp 3 ((exp 3 3) ^ 3)
def p3 : ℕ := ((exp 3 3) ^ 3) ^ 3
def p4 : ℕ := (exp 3 (exp 3 3)) ^ 3
def p5 : ℕ := (exp 3 3) ^ (exp 3 3)

-- Theorem stating that there are exactly 5 distinct values
theorem distinct_parenthesizations :
  ∃! (s : Finset ℕ), s = {p1, p2, p3, p4, p5} ∧ s.card = 5 :=
sorry

end NUMINAMATH_CALUDE_distinct_parenthesizations_l3894_389491


namespace NUMINAMATH_CALUDE_profit_margin_in_terms_of_selling_price_l3894_389417

/-- Given a selling price S, cost C, and profit margin M, prove that
    if S = 3C and M = (1/2n)C + (1/3n)S, then M = S/(2n) -/
theorem profit_margin_in_terms_of_selling_price
  (S C : ℝ) (n : ℝ) (hn : n ≠ 0) (M : ℝ) 
  (h_selling_price : S = 3 * C)
  (h_profit_margin : M = (1 / (2 * n)) * C + (1 / (3 * n)) * S) :
  M = S / (2 * n) := by
sorry

end NUMINAMATH_CALUDE_profit_margin_in_terms_of_selling_price_l3894_389417


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3894_389418

theorem min_value_quadratic (x : ℝ) : 
  ∃ (z_min : ℝ), ∀ (z : ℝ), z = x^2 + 16*x + 20 → z ≥ z_min ∧ ∃ (x_min : ℝ), x_min^2 + 16*x_min + 20 = z_min :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3894_389418


namespace NUMINAMATH_CALUDE_chess_team_girls_l3894_389460

theorem chess_team_girls (total : ℕ) (boys girls : ℕ) 
  (h1 : total = boys + girls)
  (h2 : total = 26)
  (h3 : 3 * boys / 4 + girls / 4 = 13) : 
  girls = 13 := by
sorry

end NUMINAMATH_CALUDE_chess_team_girls_l3894_389460


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3894_389471

theorem right_triangle_hypotenuse (area : ℝ) (leg : ℝ) (hypotenuse : ℝ) :
  area = 320 →
  leg = 16 →
  area = (1 / 2) * leg * (area / (1 / 2 * leg)) →
  hypotenuse^2 = leg^2 + (area / (1 / 2 * leg))^2 →
  hypotenuse = 4 * Real.sqrt 116 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3894_389471


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l3894_389416

theorem shoe_price_calculation (initial_price : ℝ) 
  (price_increase_percent : ℝ) (discount_percent : ℝ) (tax_percent : ℝ) : 
  initial_price = 50 ∧ 
  price_increase_percent = 20 ∧ 
  discount_percent = 15 ∧ 
  tax_percent = 5 → 
  initial_price * (1 + price_increase_percent / 100) * 
  (1 - discount_percent / 100) * (1 + tax_percent / 100) = 53.55 := by
sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l3894_389416


namespace NUMINAMATH_CALUDE_range_of_a_given_points_on_opposite_sides_l3894_389490

/-- Given points M(1, -a) and N(a, 1) are on opposite sides of the line 2x-3y+1=0,
    prove that the range of the real number a is -1 < a < 1. -/
theorem range_of_a_given_points_on_opposite_sides (a : ℝ) : 
  (∃ (M N : ℝ × ℝ), 
    M = (1, -a) ∧ 
    N = (a, 1) ∧ 
    (2 * M.1 - 3 * M.2 + 1) * (2 * N.1 - 3 * N.2 + 1) < 0) →
  -1 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_given_points_on_opposite_sides_l3894_389490


namespace NUMINAMATH_CALUDE_inequality_proof_l3894_389474

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3894_389474


namespace NUMINAMATH_CALUDE_f_max_value_l3894_389458

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + Real.sin x) + Real.sin (x - Real.sin x) + (Real.pi / 2 - 2) * Real.sin (Real.sin x)

theorem f_max_value : 
  ∃ (M : ℝ), M = (Real.pi - 2) / Real.sqrt 2 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_f_max_value_l3894_389458


namespace NUMINAMATH_CALUDE_dish_price_proof_l3894_389422

theorem dish_price_proof (discount_rate : Real) (tip_rate : Real) (price_difference : Real) :
  let original_price : Real := 36
  let john_payment := original_price * (1 - discount_rate) + original_price * tip_rate
  let jane_payment := original_price * (1 - discount_rate) * (1 + tip_rate)
  discount_rate = 0.1 ∧ tip_rate = 0.15 ∧ price_difference = 0.54 →
  john_payment - jane_payment = price_difference :=
by
  sorry

end NUMINAMATH_CALUDE_dish_price_proof_l3894_389422


namespace NUMINAMATH_CALUDE_abc_sum_eq_three_l3894_389426

theorem abc_sum_eq_three (a b c : ℕ+) 
  (h1 : c = b^2)
  (h2 : (a + b + c)^3 - a^3 - b^3 - c^3 = 210) :
  a + b + c = 3 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_eq_three_l3894_389426


namespace NUMINAMATH_CALUDE_rocky_total_miles_l3894_389493

/-- Rocky's training schedule for the first three days -/
def rocky_training : Fin 3 → ℕ
| 0 => 4  -- Day 1: 4 miles
| 1 => 4 * 2  -- Day 2: Double day 1
| 2 => 4 * 2 * 3  -- Day 3: Triple day 2

/-- The total miles Rocky ran in the first three days of training -/
theorem rocky_total_miles :
  (Finset.sum Finset.univ rocky_training) = 36 := by
  sorry

end NUMINAMATH_CALUDE_rocky_total_miles_l3894_389493

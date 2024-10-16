import Mathlib

namespace NUMINAMATH_CALUDE_integral_exp_plus_x_l2301_230134

theorem integral_exp_plus_x : ∫ x in (0 : ℝ)..(1 : ℝ), (Real.exp x + x) = Real.exp 1 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_exp_plus_x_l2301_230134


namespace NUMINAMATH_CALUDE_cylinder_cone_volume_relation_l2301_230129

theorem cylinder_cone_volume_relation :
  ∀ (d : ℝ) (h : ℝ),
    d > 0 →
    h = 2 * d →
    π * (d / 2)^2 * h = 81 * π →
    (1 / 3) * π * (d / 2)^2 * h = 27 * π * (6 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cone_volume_relation_l2301_230129


namespace NUMINAMATH_CALUDE_travel_time_ratio_is_one_to_one_l2301_230109

-- Define the time spent on each leg of the journey
def walk_to_bus : ℕ := 5
def bus_ride : ℕ := 20
def walk_to_job : ℕ := 5

-- Define the total travel time per year in hours
def total_travel_time_per_year : ℕ := 365

-- Define the number of days worked per year
def days_per_year : ℕ := 365

-- Define the total travel time for one way (morning or evening)
def one_way_travel_time : ℕ := walk_to_bus + bus_ride + walk_to_job

-- Theorem to prove
theorem travel_time_ratio_is_one_to_one :
  one_way_travel_time = (total_travel_time_per_year * 60) / (2 * days_per_year) :=
by
  sorry

#check travel_time_ratio_is_one_to_one

end NUMINAMATH_CALUDE_travel_time_ratio_is_one_to_one_l2301_230109


namespace NUMINAMATH_CALUDE_statement_is_false_l2301_230188

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem statement_is_false : ∃ n : ℕ, 
  (sum_of_digits n % 6 = 0) ∧ (n % 6 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_statement_is_false_l2301_230188


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2301_230135

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + a ≥ 0) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2301_230135


namespace NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_three_l2301_230124

theorem simplify_fraction_with_sqrt_three : 
  (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_three_l2301_230124


namespace NUMINAMATH_CALUDE_spinner_probability_l2301_230170

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_C = 1/6 → p_A + p_B + p_C + p_D = 1 → p_D = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l2301_230170


namespace NUMINAMATH_CALUDE_double_length_isosceles_triangle_base_length_l2301_230143

/-- A triangle is double-length if one side is twice the length of another side. -/
def is_double_length_triangle (a b c : ℝ) : Prop :=
  a = 2 * b ∨ a = 2 * c ∨ b = 2 * c

/-- An isosceles triangle has two sides of equal length. -/
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ a = c ∨ b = c

theorem double_length_isosceles_triangle_base_length
  (a b c : ℝ)
  (h_isosceles : is_isosceles_triangle a b c)
  (h_double_length : is_double_length_triangle a b c)
  (h_ab_length : a = 10) :
  c = 5 := by
  sorry

end NUMINAMATH_CALUDE_double_length_isosceles_triangle_base_length_l2301_230143


namespace NUMINAMATH_CALUDE_quadratic_equation_h_value_l2301_230185

theorem quadratic_equation_h_value (h : ℝ) : 
  (∃ r s : ℝ, r^2 + 2*h*r = 3 ∧ s^2 + 2*h*s = 3 ∧ r^2 + s^2 = 10) → 
  |h| = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_h_value_l2301_230185


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2301_230189

/-- The function f(x) = a^(x-1) + 2 has (1, 3) as a fixed point, where a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x - 1) + 2
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2301_230189


namespace NUMINAMATH_CALUDE_new_average_age_with_teacher_l2301_230194

theorem new_average_age_with_teacher 
  (num_students : ℕ) 
  (student_avg_age : ℝ) 
  (teacher_age : ℝ) : 
  num_students = 10 → 
  student_avg_age = 15 → 
  teacher_age = 26 → 
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_new_average_age_with_teacher_l2301_230194


namespace NUMINAMATH_CALUDE_kylie_coins_from_brother_l2301_230163

/-- The number of coins Kylie got from her piggy bank -/
def piggy_bank_coins : ℕ := 15

/-- The number of coins Kylie got from her father -/
def father_coins : ℕ := 8

/-- The number of coins Kylie gave to her friend Laura -/
def coins_given_away : ℕ := 21

/-- The number of coins Kylie had left -/
def coins_left : ℕ := 15

/-- The number of coins Kylie got from her brother -/
def brother_coins : ℕ := 13

theorem kylie_coins_from_brother :
  piggy_bank_coins + brother_coins + father_coins - coins_given_away = coins_left :=
by sorry

end NUMINAMATH_CALUDE_kylie_coins_from_brother_l2301_230163


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2301_230191

theorem complex_modulus_problem (z : ℂ) (h : (1 - I) * z = 1 + 2*I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2301_230191


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2301_230108

theorem inequality_solution_set (x : ℝ) :
  (∀ y : ℝ, y > 0 → (4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y) ↔
  0 ≤ x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2301_230108


namespace NUMINAMATH_CALUDE_abs_less_of_even_increasing_fn_l2301_230181

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- State the theorem
theorem abs_less_of_even_increasing_fn (a b : ℝ) 
  (h_even : is_even f) 
  (h_incr : is_increasing_on_nonneg f) 
  (h_less : f a < f b) : 
  |a| < |b| := by
  sorry

end NUMINAMATH_CALUDE_abs_less_of_even_increasing_fn_l2301_230181


namespace NUMINAMATH_CALUDE_min_garden_cost_l2301_230168

/-- Represents a rectangular region in the flower bed -/
structure Region where
  length : ℝ
  width : ℝ

/-- Represents a type of flower -/
structure Flower where
  name : String
  price : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Calculates the cost of filling a region with a specific flower -/
def cost (r : Region) (f : Flower) : ℝ := area r * f.price

/-- The flower bed arrangement -/
def flowerBed : List Region := [
  { length := 5, width := 2 },
  { length := 4, width := 2 },
  { length := 7, width := 4 },
  { length := 3, width := 5 }
]

/-- Available flower types -/
def flowers : List Flower := [
  { name := "Fuchsia", price := 3.5 },
  { name := "Gardenia", price := 4 },
  { name := "Canna", price := 2 },
  { name := "Begonia", price := 1.5 }
]

/-- Theorem stating the minimum cost of the garden -/
theorem min_garden_cost :
  ∃ (arrangement : List (Region × Flower)),
    arrangement.length = flowerBed.length ∧
    (∀ r ∈ flowerBed, ∃ f ∈ flowers, (r, f) ∈ arrangement) ∧
    (arrangement.map (λ (r, f) => cost r f)).sum = 140 ∧
    ∀ (other_arrangement : List (Region × Flower)),
      other_arrangement.length = flowerBed.length →
      (∀ r ∈ flowerBed, ∃ f ∈ flowers, (r, f) ∈ other_arrangement) →
      (other_arrangement.map (λ (r, f) => cost r f)).sum ≥ 140 := by
  sorry

end NUMINAMATH_CALUDE_min_garden_cost_l2301_230168


namespace NUMINAMATH_CALUDE_f_properties_l2301_230195

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sin x * Real.cos x - 1/2

theorem f_properties :
  ∃ (k : ℤ), 
    (∀ x : ℝ, f x = f (π/4 - x)) ∧ 
    (∀ x : ℝ, (∃ k : ℤ, x ∈ Set.Icc (k * π - 3*π/8) (k * π + π/8)) → (deriv f) x > 0) ∧
    (f (π/2) = -1/2 ∧ ∀ x ∈ Set.Icc 0 (π/2), f x ≥ -1/2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2301_230195


namespace NUMINAMATH_CALUDE_max_distance_and_squared_distance_coincide_l2301_230146

-- Define a triangle ABC in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the sum of distances from a point to the vertices
def sumDistances (t : Triangle) (p : ℝ × ℝ) : ℝ :=
  distance p t.A + distance p t.B + distance p t.C

-- Define a function to calculate the sum of squared distances from a point to the vertices
def sumSquaredDistances (t : Triangle) (p : ℝ × ℝ) : ℝ :=
  (distance p t.A)^2 + (distance p t.B)^2 + (distance p t.C)^2

-- Define a function to find the shortest side of a triangle
def shortestSide (t : Triangle) : ℝ := sorry

-- Define a function to find the vertex opposite the shortest side
def vertexOppositeShortestSide (t : Triangle) : ℝ × ℝ := sorry

theorem max_distance_and_squared_distance_coincide (t : Triangle) :
  ∃ p : ℝ × ℝ,
    (∀ q : ℝ × ℝ, sumDistances t q ≤ sumDistances t p) ∧
    (∀ q : ℝ × ℝ, sumSquaredDistances t q ≤ sumSquaredDistances t p) ∧
    p = vertexOppositeShortestSide t :=
  sorry


end NUMINAMATH_CALUDE_max_distance_and_squared_distance_coincide_l2301_230146


namespace NUMINAMATH_CALUDE_car_speed_problem_l2301_230147

/-- Proves that given a 6-hour trip where the average speed for the first 4 hours is 35 mph
    and the average speed for the entire trip is 38 mph, the average speed for the remaining 2 hours is 44 mph. -/
theorem car_speed_problem (total_time : ℝ) (initial_time : ℝ) (initial_speed : ℝ) (total_avg_speed : ℝ) :
  total_time = 6 →
  initial_time = 4 →
  initial_speed = 35 →
  total_avg_speed = 38 →
  let remaining_time := total_time - initial_time
  let total_distance := total_avg_speed * total_time
  let initial_distance := initial_speed * initial_time
  let remaining_distance := total_distance - initial_distance
  remaining_distance / remaining_time = 44 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2301_230147


namespace NUMINAMATH_CALUDE_andrews_friends_pizza_slices_l2301_230121

/-- The total number of pizza slices brought by Andrew's friends -/
def total_pizza_slices (num_friends : ℕ) (slices_per_friend : ℕ) : ℕ :=
  num_friends * slices_per_friend

/-- Theorem stating that the total number of pizza slices is 16 -/
theorem andrews_friends_pizza_slices :
  total_pizza_slices 4 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_andrews_friends_pizza_slices_l2301_230121


namespace NUMINAMATH_CALUDE_triangle_angle_R_l2301_230120

theorem triangle_angle_R (P Q R : Real) (h1 : 2 * Real.sin P + 5 * Real.cos Q = 4) 
  (h2 : 5 * Real.sin Q + 2 * Real.cos P = 3) 
  (h3 : P + Q + R = Real.pi) : R = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_R_l2301_230120


namespace NUMINAMATH_CALUDE_union_of_sets_l2301_230122

theorem union_of_sets : 
  let A : Set ℕ := {1, 3, 5}
  let B : Set ℕ := {3, 5, 7}
  A ∪ B = {1, 3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l2301_230122


namespace NUMINAMATH_CALUDE_movies_watched_count_l2301_230167

/-- The number of movies watched in the 'crazy silly school' series -/
def movies_watched : ℕ := 21

/-- The number of books read in the 'crazy silly school' series -/
def books_read : ℕ := 7

/-- Theorem stating that the number of movies watched is 21 -/
theorem movies_watched_count : 
  movies_watched = books_read + 14 := by sorry

end NUMINAMATH_CALUDE_movies_watched_count_l2301_230167


namespace NUMINAMATH_CALUDE_stock_investment_calculation_l2301_230159

/-- Given a stock with price 64 and dividend yield 1623%, prove that an investment
    earning 1900 in dividends is approximately 117.00 -/
theorem stock_investment_calculation (stock_price : ℝ) (dividend_yield : ℝ) (dividend_earned : ℝ) :
  stock_price = 64 →
  dividend_yield = 1623 →
  dividend_earned = 1900 →
  ∃ (investment : ℝ), abs (investment - 117.00) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_stock_investment_calculation_l2301_230159


namespace NUMINAMATH_CALUDE_right_triangle_leg_lengths_l2301_230130

/-- 
Given a right triangle where the height from the right angle vertex 
divides the hypotenuse into segments of lengths a and b, 
the lengths of the legs are √(a(a+b)) and √(b(a+b)).
-/
theorem right_triangle_leg_lengths 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) : 
  ∃ (leg1 leg2 : ℝ), 
    leg1 = Real.sqrt (a * (a + b)) ∧ 
    leg2 = Real.sqrt (b * (a + b)) ∧
    leg1^2 + leg2^2 = (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_lengths_l2301_230130


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l2301_230158

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, |x| + x^2 < 0) ↔ (∀ x : ℝ, |x| + x^2 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l2301_230158


namespace NUMINAMATH_CALUDE_segment_mapping_l2301_230183

theorem segment_mapping (a : ℝ) : ∃ (x y : ℝ), 
  (∃ (AB A'B' : ℝ), AB = 3 ∧ A'B' = 6 ∧
  (∀ (P D P' D' : ℝ), 
    (P - D = x ∧ P' - D' = 2*x) →
    (x = a → x + y = 3*a))) :=
by sorry

end NUMINAMATH_CALUDE_segment_mapping_l2301_230183


namespace NUMINAMATH_CALUDE_fermat_numbers_coprime_l2301_230144

theorem fermat_numbers_coprime (m n : ℕ) (h : m ≠ n) :
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fermat_numbers_coprime_l2301_230144


namespace NUMINAMATH_CALUDE_smallest_with_200_divisors_l2301_230169

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Checks if n can be written as m * 10^k where 10 is not a divisor of m -/
def has_form (n m k : ℕ) : Prop :=
  n = m * (10 ^ k) ∧ ¬(10 ∣ m)

theorem smallest_with_200_divisors :
  ∃ (n m k : ℕ),
    (∀ i < n, num_divisors i < 200) ∧
    num_divisors n = 200 ∧
    has_form n m k ∧
    m + k = 18 := by sorry

end NUMINAMATH_CALUDE_smallest_with_200_divisors_l2301_230169


namespace NUMINAMATH_CALUDE_school_supplies_cost_l2301_230175

/-- Calculates the total cost of school supplies with discounts applied --/
theorem school_supplies_cost 
  (haley_paper_price : ℝ) 
  (haley_paper_quantity : ℕ)
  (sister_paper_price : ℝ)
  (sister_paper_quantity : ℕ)
  (paper_discount : ℝ)
  (haley_pen_price : ℝ)
  (haley_pen_quantity : ℕ)
  (sister_pen_price : ℝ)
  (sister_pen_quantity : ℕ)
  (pen_discount : ℝ)
  (h1 : haley_paper_price = 3.75)
  (h2 : haley_paper_quantity = 2)
  (h3 : sister_paper_price = 4.50)
  (h4 : sister_paper_quantity = 3)
  (h5 : paper_discount = 0.5)
  (h6 : haley_pen_price = 1.45)
  (h7 : haley_pen_quantity = 5)
  (h8 : sister_pen_price = 1.65)
  (h9 : sister_pen_quantity = 7)
  (h10 : pen_discount = 0.25)
  : ℝ := by
  sorry

end NUMINAMATH_CALUDE_school_supplies_cost_l2301_230175


namespace NUMINAMATH_CALUDE_no_roots_composition_l2301_230197

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b * x + c

/-- Theorem: If f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_roots_composition (b c : ℝ) 
  (h : ∀ x : ℝ, f b c x ≠ x) : 
  ∀ x : ℝ, f b c (f b c x) ≠ x := by
  sorry

end NUMINAMATH_CALUDE_no_roots_composition_l2301_230197


namespace NUMINAMATH_CALUDE_red_tiles_181_implies_total_2116_l2301_230112

/-- Represents a square floor tiled with congruent square tiles -/
structure SquareFloor :=
  (side : ℕ)

/-- Calculates the number of red tiles on a square floor -/
def red_tiles (floor : SquareFloor) : ℕ :=
  4 * floor.side - 2

/-- Calculates the total number of tiles on a square floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side * floor.side

/-- Theorem stating that a square floor with 181 red tiles has 2116 total tiles -/
theorem red_tiles_181_implies_total_2116 :
  ∀ (floor : SquareFloor), red_tiles floor = 181 → total_tiles floor = 2116 :=
by
  sorry

end NUMINAMATH_CALUDE_red_tiles_181_implies_total_2116_l2301_230112


namespace NUMINAMATH_CALUDE_first_native_is_liar_and_path_is_incorrect_l2301_230105

-- Define the types of natives
inductive NativeType
| Truthful
| Liar

-- Define a native
structure Native where
  type : NativeType

-- Define the claim about being a Liar
def claimToBeLiar (n : Native) : Prop :=
  match n.type with
  | NativeType.Truthful => False
  | NativeType.Liar => False

-- Define the first native's report about the second native's claim
def firstNativeReport (first : Native) (second : Native) : Prop :=
  claimToBeLiar second

-- Define the correctness of the path indication
def correctPathIndication (n : Native) : Prop :=
  match n.type with
  | NativeType.Truthful => True
  | NativeType.Liar => False

-- Theorem statement
theorem first_native_is_liar_and_path_is_incorrect 
  (first : Native) (second : Native) :
  firstNativeReport first second →
  first.type = NativeType.Liar ∧ ¬(correctPathIndication first) := by
  sorry

end NUMINAMATH_CALUDE_first_native_is_liar_and_path_is_incorrect_l2301_230105


namespace NUMINAMATH_CALUDE_hydropower_station_calculations_l2301_230156

/-- A hydropower station with given parameters -/
structure HydropowerStation where
  generator_power : ℝ
  generator_voltage : ℝ
  line_resistance : ℝ
  power_loss_percentage : ℝ
  user_voltage : ℝ

/-- Calculates the current in the transmission line -/
def transmission_line_current (station : HydropowerStation) : ℝ :=
  sorry

/-- Calculates the turns ratio of the step-up transformer -/
def step_up_transformer_ratio (station : HydropowerStation) : ℚ :=
  sorry

/-- Calculates the turns ratio of the step-down transformer -/
def step_down_transformer_ratio (station : HydropowerStation) : ℚ :=
  sorry

/-- Theorem stating the correct calculations for the hydropower station -/
theorem hydropower_station_calculations 
  (station : HydropowerStation) 
  (h1 : station.generator_power = 24.5)
  (h2 : station.generator_voltage = 350)
  (h3 : station.line_resistance = 4)
  (h4 : station.power_loss_percentage = 0.05)
  (h5 : station.user_voltage = 220) :
  transmission_line_current station = 17.5 ∧ 
  step_up_transformer_ratio station = 1 / 4 ∧
  step_down_transformer_ratio station = 133 / 22 :=
by sorry

end NUMINAMATH_CALUDE_hydropower_station_calculations_l2301_230156


namespace NUMINAMATH_CALUDE_jamie_balls_l2301_230142

theorem jamie_balls (R : ℕ) : 
  (R - 6) + 2 * R + 32 = 74 → R = 16 := by
sorry

end NUMINAMATH_CALUDE_jamie_balls_l2301_230142


namespace NUMINAMATH_CALUDE_sibling_of_five_sevenths_unique_parent_one_over_2008_descendant_of_one_l2301_230155

-- Define the child relation
def is_child (x y : ℝ) : Prop :=
  (y = x + 1) ∨ (y = x / (x + 1))

-- Define the sibling relation
def is_sibling (x y : ℝ) : Prop :=
  ∃ z, is_child z x ∧ y = z + 1

-- Define the descendant relation
def is_descendant (x y : ℝ) : Prop :=
  ∃ n : ℕ, ∃ f : ℕ → ℝ,
    f 0 = x ∧ f n = y ∧
    ∀ i < n, is_child (f i) (f (i + 1))

theorem sibling_of_five_sevenths :
  is_sibling (5/7) (7/2) :=
sorry

theorem unique_parent (x y z : ℝ) (hx : x > 0) (hz : z > 0) :
  is_child x y → is_child z y → x = z :=
sorry

theorem one_over_2008_descendant_of_one :
  is_descendant 1 (1/2008) :=
sorry

end NUMINAMATH_CALUDE_sibling_of_five_sevenths_unique_parent_one_over_2008_descendant_of_one_l2301_230155


namespace NUMINAMATH_CALUDE_ellipse_properties_l2301_230125

/-- Definition of an ellipse C with given parameters -/
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Definition of an equilateral triangle with given side length -/
def EquilateralTriangle (side : ℝ) (p1 p2 p3 : ℝ × ℝ) :=
  ‖p1 - p2‖ = side ∧ ‖p2 - p3‖ = side ∧ ‖p3 - p1‖ = side

/-- Theorem about the properties of a specific ellipse -/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (F1 F2 B : ℝ × ℝ) (h3 : EquilateralTriangle 2 B F1 F2) :
  ∃ (C : Set (ℝ × ℝ)) (e : ℝ) (l1 l2 : ℝ → ℝ),
    C = Ellipse 2 (Real.sqrt 3) ∧
    e = (1 : ℝ) / 2 ∧
    (∀ x, l1 x = (Real.sqrt 5 * x - Real.sqrt 5) / 2) ∧
    (∀ x, l2 x = (-Real.sqrt 5 * x + Real.sqrt 5) / 2) ∧
    (∃ P Q : ℝ × ℝ, P ∈ C ∧ Q ∈ C ∧
      (P.2 = l1 P.1 ∨ P.2 = l2 P.1) ∧
      (Q.2 = l1 Q.1 ∨ Q.2 = l2 Q.1) ∧
      ((P.1 - 2) * (Q.2 + 1) = (P.2) * (Q.1 + 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2301_230125


namespace NUMINAMATH_CALUDE_current_library_books_l2301_230192

def library_books (initial : ℕ) (first_purchase : ℕ) (second_purchase : ℕ) (donation : ℕ) : ℕ :=
  initial + first_purchase + second_purchase - donation

theorem current_library_books :
  library_books 500 300 400 200 = 1000 := by sorry

end NUMINAMATH_CALUDE_current_library_books_l2301_230192


namespace NUMINAMATH_CALUDE_ellipse_max_value_l2301_230119

theorem ellipse_max_value (x y : ℝ) :
  (x^2 / 6 + y^2 / 4 = 1) →
  (∃ (max : ℝ), ∀ (a b : ℝ), a^2 / 6 + b^2 / 4 = 1 → x + 2*y ≤ max ∧ max = Real.sqrt 22) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_max_value_l2301_230119


namespace NUMINAMATH_CALUDE_star_five_three_l2301_230126

/-- Define the binary operation ⋆ -/
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

/-- Theorem: When a = 5 and b = 3, a ⋆ b = 4 -/
theorem star_five_three : star 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_star_five_three_l2301_230126


namespace NUMINAMATH_CALUDE_marcos_strawberry_weight_l2301_230178

/-- Marco and his dad went strawberry picking. This theorem proves the weight of Marco's strawberries. -/
theorem marcos_strawberry_weight
  (total_weight : ℕ)
  (dads_weight : ℕ)
  (h1 : total_weight = 40)
  (h2 : dads_weight = 32)
  : total_weight - dads_weight = 8 := by
  sorry

end NUMINAMATH_CALUDE_marcos_strawberry_weight_l2301_230178


namespace NUMINAMATH_CALUDE_initial_volume_proof_l2301_230153

/-- Given a solution with initial volume V and 5% alcohol concentration,
    adding 2.5 liters of alcohol and 7.5 liters of water results in a
    9% alcohol concentration. Prove that V must be 40 liters. -/
theorem initial_volume_proof (V : ℝ) : 
  (0.05 * V + 2.5) / (V + 10) = 0.09 → V = 40 := by sorry

end NUMINAMATH_CALUDE_initial_volume_proof_l2301_230153


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2301_230115

-- Define the variables and conditions
variable (a b c : ℝ)
variable (h1 : 2 * a + b = c)
variable (h2 : c ≠ 0)

-- Theorem 1
theorem problem_1 : (2 * a + b - c - 1)^2023 = -1 := by sorry

-- Theorem 2
theorem problem_2 : (10 * c) / (4 * a + 2 * b) = 5 := by sorry

-- Theorem 3
theorem problem_3 : (2 * a + b) * 3 = c + 4 * a + 2 * b := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2301_230115


namespace NUMINAMATH_CALUDE_journey_distance_on_foot_l2301_230179

theorem journey_distance_on_foot 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (speed_on_foot : ℝ) 
  (speed_on_bicycle : ℝ) 
  (h1 : total_distance = 80) 
  (h2 : total_time = 7) 
  (h3 : speed_on_foot = 8) 
  (h4 : speed_on_bicycle = 16) :
  ∃ (distance_on_foot : ℝ),
    distance_on_foot = 32 ∧
    ∃ (distance_on_bicycle : ℝ),
      distance_on_foot + distance_on_bicycle = total_distance ∧
      distance_on_foot / speed_on_foot + distance_on_bicycle / speed_on_bicycle = total_time :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_on_foot_l2301_230179


namespace NUMINAMATH_CALUDE_zoo_visitors_saturday_l2301_230173

/-- The number of people who visited the zoo on Friday -/
def friday_visitors : ℕ := 1250

/-- The ratio of Saturday visitors to Friday visitors -/
def saturday_ratio : ℕ := 3

/-- The number of people who visited the zoo on Saturday -/
def saturday_visitors : ℕ := friday_visitors * saturday_ratio

theorem zoo_visitors_saturday : saturday_visitors = 3750 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_saturday_l2301_230173


namespace NUMINAMATH_CALUDE_product_equals_64_l2301_230157

theorem product_equals_64 : 
  (1/2 : ℚ) * 4 * (1/8) * 16 * (1/32) * 64 * (1/128) * 256 * (1/512) * 1024 * (1/2048) * 4096 = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_64_l2301_230157


namespace NUMINAMATH_CALUDE_trees_in_yard_l2301_230176

/-- Given a yard of length 275 meters with trees planted at equal distances,
    one tree at each end, and 11 meters between consecutive trees,
    prove that there are 26 trees in total. -/
theorem trees_in_yard (yard_length : ℕ) (tree_distance : ℕ) : 
  yard_length = 275 → 
  tree_distance = 11 → 
  (yard_length - tree_distance) % tree_distance = 0 →
  (yard_length - tree_distance) / tree_distance + 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l2301_230176


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2301_230127

theorem quadratic_equation_roots (x : ℝ) :
  (x^2 - 2*x - 1 = 0) ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2301_230127


namespace NUMINAMATH_CALUDE_quadratic_radical_range_l2301_230154

theorem quadratic_radical_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 3 - x) ↔ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_range_l2301_230154


namespace NUMINAMATH_CALUDE_berry_difference_l2301_230162

/-- The number of strawberries in a box -/
def strawberries_per_box : ℕ := 12

/-- The cost of a box of strawberries in dollars -/
def strawberry_box_cost : ℕ := 2

/-- The number of blueberries in a box -/
def blueberries_per_box : ℕ := 48

/-- The cost of a box of blueberries in dollars -/
def blueberry_box_cost : ℕ := 3

/-- The amount Sareen can spend in dollars -/
def sareen_budget : ℕ := 12

/-- The number of strawberries Sareen can buy -/
def m : ℕ := (sareen_budget / strawberry_box_cost) * strawberries_per_box

/-- The number of blueberries Sareen can buy -/
def n : ℕ := (sareen_budget / blueberry_box_cost) * blueberries_per_box

theorem berry_difference : n - m = 120 := by
  sorry

end NUMINAMATH_CALUDE_berry_difference_l2301_230162


namespace NUMINAMATH_CALUDE_collinear_vectors_l2301_230102

/-- Given vectors a and b in ℝ², if a + b is collinear with a, then the second component of a is 1. -/
theorem collinear_vectors (k : ℝ) : 
  let a : Fin 2 → ℝ := ![1, k]
  let b : Fin 2 → ℝ := ![2, 2]
  (∃ (t : ℝ), (a + b) = t • a) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l2301_230102


namespace NUMINAMATH_CALUDE_alexandrov_theorem_l2301_230149

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a face of a polyhedron -/
structure Face where
  -- Add necessary fields here
  mk ::

/-- Represents a planar angle in a polyhedron -/
def PlanarAngle : Type := ℝ

/-- Represents a dihedral angle in a polyhedron -/
def DihedralAngle : Type := ℝ

/-- Check if two polyhedra have correspondingly equal faces -/
def has_equal_faces (P Q : ConvexPolyhedron) : Prop :=
  sorry

/-- Check if two polyhedra have equal corresponding planar angles -/
def has_equal_planar_angles (P Q : ConvexPolyhedron) : Prop :=
  sorry

/-- Check if two polyhedra have equal corresponding dihedral angles -/
def has_equal_dihedral_angles (P Q : ConvexPolyhedron) : Prop :=
  sorry

/-- Alexandrov's Theorem -/
theorem alexandrov_theorem (P Q : ConvexPolyhedron) :
  has_equal_faces P Q → has_equal_planar_angles P Q → has_equal_dihedral_angles P Q :=
by
  sorry

end NUMINAMATH_CALUDE_alexandrov_theorem_l2301_230149


namespace NUMINAMATH_CALUDE_number_problem_l2301_230101

theorem number_problem : ∃ x : ℝ, x^2 + 95 = (x - 20)^2 ∧ x = 7.625 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2301_230101


namespace NUMINAMATH_CALUDE_odd_function_constant_term_zero_l2301_230165

def f (a b c x : ℝ) : ℝ := a * x^3 - b * x + c

theorem odd_function_constant_term_zero (a b c : ℝ) :
  (∀ x, f a b c x = -f a b c (-x)) → c = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_constant_term_zero_l2301_230165


namespace NUMINAMATH_CALUDE_f_max_values_l2301_230182

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1)^2 / (x * (x^2 - 1))

theorem f_max_values (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = a ∧ f y = a ∧ f z = a) ↔ a < -4 ∨ a > 4 :=
sorry

end NUMINAMATH_CALUDE_f_max_values_l2301_230182


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l2301_230131

def vector_a : Fin 3 → ℝ := ![1, 1, 2]
def vector_b : Fin 3 → ℝ := ![2, -1, 2]

theorem cosine_of_angle_between_vectors :
  let a := vector_a
  let b := vector_b
  let dot_product := (a 0) * (b 0) + (a 1) * (b 1) + (a 2) * (b 2)
  let magnitude_a := Real.sqrt ((a 0)^2 + (a 1)^2 + (a 2)^2)
  let magnitude_b := Real.sqrt ((b 0)^2 + (b 1)^2 + (b 2)^2)
  dot_product / (magnitude_a * magnitude_b) = (5 * Real.sqrt 6) / 18 :=
by sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l2301_230131


namespace NUMINAMATH_CALUDE_fruits_per_person_correct_l2301_230184

/-- The number of fruits each person gets when evenly distributed -/
def fruits_per_person (
  kim_strawberry_multiplier : ℕ)
  (strawberries_per_basket : ℕ)
  (kim_blueberry_baskets : ℕ)
  (blueberries_per_kim_basket : ℕ)
  (brother_strawberry_baskets : ℕ)
  (brother_blackberry_baskets : ℕ)
  (blackberries_per_basket : ℕ)
  (parents_blackberry_difference : ℕ)
  (parents_extra_blueberry_baskets : ℕ)
  (parents_extra_blueberries_per_basket : ℕ)
  (family_size : ℕ) : ℕ :=
  let total_strawberries := 
    (kim_strawberry_multiplier * brother_strawberry_baskets + brother_strawberry_baskets) * strawberries_per_basket
  let total_blueberries := 
    kim_blueberry_baskets * blueberries_per_kim_basket + 
    (kim_blueberry_baskets + parents_extra_blueberry_baskets) * (blueberries_per_kim_basket + parents_extra_blueberries_per_basket)
  let total_blackberries := 
    brother_blackberry_baskets * blackberries_per_basket + 
    (brother_blackberry_baskets * blackberries_per_basket - parents_blackberry_difference)
  let total_fruits := total_strawberries + total_blueberries + total_blackberries
  total_fruits / family_size

theorem fruits_per_person_correct : 
  fruits_per_person 8 15 5 40 3 4 30 75 4 15 4 = 316 := by sorry

end NUMINAMATH_CALUDE_fruits_per_person_correct_l2301_230184


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2301_230136

theorem cubic_roots_sum (p q r : ℝ) : 
  (p^3 - 5*p^2 + 9*p - 7 = 0) →
  (q^3 - 5*q^2 + 9*q - 7 = 0) →
  (r^3 - 5*r^2 + 9*r - 7 = 0) →
  ∃ (u v : ℝ), ((p+q)^3 + u*(p+q)^2 + v*(p+q) + (-13) = 0) ∧
               ((q+r)^3 + u*(q+r)^2 + v*(q+r) + (-13) = 0) ∧
               ((r+p)^3 + u*(r+p)^2 + v*(r+p) + (-13) = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2301_230136


namespace NUMINAMATH_CALUDE_socks_cost_theorem_l2301_230151

def flat_rate : ℝ := 5
def shipping_rate : ℝ := 0.2
def shirt_price : ℝ := 12
def shirt_quantity : ℕ := 3
def shorts_price : ℝ := 15
def shorts_quantity : ℕ := 2
def swim_trunks_price : ℝ := 14
def swim_trunks_quantity : ℕ := 1
def total_bill : ℝ := 102

def known_items_cost : ℝ := 
  shirt_price * shirt_quantity + 
  shorts_price * shorts_quantity + 
  swim_trunks_price * swim_trunks_quantity

theorem socks_cost_theorem (socks_price : ℝ) : 
  (known_items_cost + socks_price > 50 → 
    known_items_cost + socks_price + shipping_rate * (known_items_cost + socks_price) = total_bill) →
  (known_items_cost + socks_price ≤ 50 → 
    known_items_cost + socks_price + flat_rate = total_bill) →
  socks_price = 5 := by
sorry

end NUMINAMATH_CALUDE_socks_cost_theorem_l2301_230151


namespace NUMINAMATH_CALUDE_max_value_cubic_expression_l2301_230164

theorem max_value_cubic_expression (a b : ℝ) (h : a^2 + b^2 = 1) :
  ∃ (max : ℝ), max = 1/4 ∧ ∀ (x y : ℝ), x^2 + y^2 = 1 → x^3 * y - y^3 * x ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_cubic_expression_l2301_230164


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l2301_230123

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 30 ∧ initial_mean = 180 ∧ incorrect_value = 135 ∧ correct_value = 155 →
  (n * initial_mean + (correct_value - incorrect_value)) / n = 180.67 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l2301_230123


namespace NUMINAMATH_CALUDE_rachel_age_proof_l2301_230117

/-- Rachel's age in years -/
def rachel_age : ℕ := 12

/-- Rachel's grandfather's age in years -/
def grandfather_age (r : ℕ) : ℕ := 7 * r

/-- Rachel's mother's age in years -/
def mother_age (r : ℕ) : ℕ := grandfather_age r / 2

/-- Rachel's father's age in years -/
def father_age (r : ℕ) : ℕ := mother_age r + 5

theorem rachel_age_proof :
  rachel_age = 12 ∧
  grandfather_age rachel_age = 7 * rachel_age ∧
  mother_age rachel_age = grandfather_age rachel_age / 2 ∧
  father_age rachel_age = mother_age rachel_age + 5 ∧
  father_age rachel_age = rachel_age + 35 ∧
  father_age 25 = 60 :=
by sorry

end NUMINAMATH_CALUDE_rachel_age_proof_l2301_230117


namespace NUMINAMATH_CALUDE_triangle_equation_l2301_230118

/-- A non-isosceles triangle with side lengths a, b, c opposite to angles A, B, C respectively,
    where A, B, C form an arithmetic sequence. -/
structure NonIsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  nonIsosceles : a ≠ b ∧ b ≠ c ∧ a ≠ c
  oppositeAngles : angleA.cos = (b^2 + c^2 - a^2) / (2*b*c) ∧
                   angleB.cos = (a^2 + c^2 - b^2) / (2*a*c) ∧
                   angleC.cos = (a^2 + b^2 - c^2) / (2*a*b)
  arithmeticSequence : ∃ (d : ℝ), angleB = angleA + d ∧ angleC = angleB + d

/-- The main theorem stating the equation holds for non-isosceles triangles with angles
    in arithmetic sequence. -/
theorem triangle_equation (t : NonIsoscelesTriangle) :
  1 / (t.a - t.b) + 1 / (t.c - t.b) = 3 / (t.a - t.b + t.c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_equation_l2301_230118


namespace NUMINAMATH_CALUDE_first_cut_ratio_l2301_230137

/-- Proves the ratio of the first cut rope to the initial rope length is 1/2 -/
theorem first_cut_ratio (initial_length : ℝ) (final_piece_length : ℝ) : 
  initial_length = 100 → 
  final_piece_length = 5 → 
  (initial_length / 2) / initial_length = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_first_cut_ratio_l2301_230137


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l2301_230103

/-- The axis of symmetry of a parabola y = (x - h)^2 + k is the line x = h -/
theorem parabola_axis_of_symmetry (h k : ℝ) :
  let f : ℝ → ℝ := λ x => (x - h)^2 + k
  ∀ x, f (h + x) = f (h - x) :=
by sorry

/-- The axis of symmetry of the parabola y = (x - 1)^2 + 3 is the line x = 1 -/
theorem specific_parabola_axis_of_symmetry :
  let f : ℝ → ℝ := λ x => (x - 1)^2 + 3
  ∀ x, f (1 + x) = f (1 - x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l2301_230103


namespace NUMINAMATH_CALUDE_angle_sum_equality_l2301_230139

theorem angle_sum_equality (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h4 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_equality_l2301_230139


namespace NUMINAMATH_CALUDE_tank_dimension_l2301_230171

theorem tank_dimension (x : ℝ) : 
  x > 0 ∧ 
  (2 * (x * 5 + x * 2 + 5 * 2)) * 20 = 1240 → 
  x = 3 :=
by sorry

end NUMINAMATH_CALUDE_tank_dimension_l2301_230171


namespace NUMINAMATH_CALUDE_smallest_x_with_remainders_l2301_230107

theorem smallest_x_with_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 3 = 2 ∧ 
  x % 4 = 3 ∧ 
  x % 5 = 4 ∧ 
  ∀ y : ℕ, y > 0 → y % 3 = 2 → y % 4 = 3 → y % 5 = 4 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_x_with_remainders_l2301_230107


namespace NUMINAMATH_CALUDE_mouse_cost_l2301_230138

theorem mouse_cost (mouse_cost keyboard_cost total_cost : ℝ) : 
  keyboard_cost = 3 * mouse_cost →
  total_cost = mouse_cost + keyboard_cost →
  total_cost = 64 →
  mouse_cost = 16 := by
sorry

end NUMINAMATH_CALUDE_mouse_cost_l2301_230138


namespace NUMINAMATH_CALUDE_picture_arrangements_l2301_230113

/-- The number of people in the initial group -/
def initial_group_size : ℕ := 4

/-- The number of people combined into one unit -/
def combined_unit_size : ℕ := 2

/-- The effective number of units to arrange -/
def effective_units : ℕ := initial_group_size - combined_unit_size + 1

theorem picture_arrangements :
  (effective_units).factorial = 6 := by
  sorry

end NUMINAMATH_CALUDE_picture_arrangements_l2301_230113


namespace NUMINAMATH_CALUDE_cube_root_of_product_l2301_230100

theorem cube_root_of_product (a b c : ℕ) :
  (2^9 * 5^3 * 7^6 : ℝ)^(1/3) = 1960 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l2301_230100


namespace NUMINAMATH_CALUDE_urea_formation_moles_l2301_230180

-- Define the chemical species
inductive ChemicalSpecies
| CarbonDioxide
| Ammonia
| Urea
| Water

-- Define a structure for chemical reactions
structure ChemicalReaction where
  reactants : List (ChemicalSpecies × ℚ)
  products : List (ChemicalSpecies × ℚ)

-- Define the urea formation reaction
def ureaFormationReaction : ChemicalReaction :=
  { reactants := [(ChemicalSpecies.CarbonDioxide, 1), (ChemicalSpecies.Ammonia, 2)]
  , products := [(ChemicalSpecies.Urea, 1), (ChemicalSpecies.Water, 1)] }

-- Define a function to calculate the moles of product formed
def molesOfProductFormed (reaction : ChemicalReaction) (limitingReactant : ChemicalSpecies) (molesOfLimitingReactant : ℚ) (product : ChemicalSpecies) : ℚ :=
  sorry -- Implementation details omitted

-- Theorem statement
theorem urea_formation_moles :
  molesOfProductFormed ureaFormationReaction ChemicalSpecies.CarbonDioxide 1 ChemicalSpecies.Urea = 1 :=
sorry

end NUMINAMATH_CALUDE_urea_formation_moles_l2301_230180


namespace NUMINAMATH_CALUDE_relay_race_assignments_l2301_230104

theorem relay_race_assignments (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  (n.factorial / (n - k).factorial : ℕ) = 32760 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_assignments_l2301_230104


namespace NUMINAMATH_CALUDE_divisibility_property_l2301_230196

theorem divisibility_property (n : ℕ) : n ≥ 1 ∧ n ∣ (3^n + 1) ∧ n ∣ (11^n + 1) ↔ n = 1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2301_230196


namespace NUMINAMATH_CALUDE_P_equals_set_l2301_230150

def P : Set ℝ := {x | x^2 = 1}

theorem P_equals_set : P = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_P_equals_set_l2301_230150


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_three_halves_l2301_230152

theorem sqrt_expression_equals_three_halves :
  (Real.sqrt 8 - Real.sqrt (1/2)) / Real.sqrt 2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_three_halves_l2301_230152


namespace NUMINAMATH_CALUDE_max_ab_min_3x_4y_max_f_l2301_230110

-- Part 1
theorem max_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 1) :
  a * b ≤ 1 / 16 := by sorry

-- Part 2
theorem min_3x_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 := by sorry

-- Part 3
theorem max_f (x : ℝ) (h : x < 5 / 4) :
  4 * x - 2 + 1 / (4 * x - 5) ≤ 1 := by sorry

end NUMINAMATH_CALUDE_max_ab_min_3x_4y_max_f_l2301_230110


namespace NUMINAMATH_CALUDE_three_digit_powers_of_three_l2301_230116

theorem three_digit_powers_of_three (n : ℕ) : 
  (∃ k, 100 ≤ 3^k ∧ 3^k ≤ 999) ∧ (∀ m, 100 ≤ 3^m ∧ 3^m ≤ 999 → m = n ∨ m = n+1) :=
sorry

end NUMINAMATH_CALUDE_three_digit_powers_of_three_l2301_230116


namespace NUMINAMATH_CALUDE_concat_reverse_divisible_by_99_l2301_230193

def is_valid_permutation (p : List Nat) : Prop :=
  p.length = 10 ∧ 
  p.head? ≠ some 0 ∧ 
  (∀ i, i ∈ p → i < 10) ∧
  (∀ i, i < 10 → i ∈ p)

def concat_with_reverse (p : List Nat) : List Nat :=
  p ++ p.reverse

def to_number (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem concat_reverse_divisible_by_99 (p : List Nat) 
  (h : is_valid_permutation p) : 
  99 ∣ to_number (concat_with_reverse p) := by
  sorry

end NUMINAMATH_CALUDE_concat_reverse_divisible_by_99_l2301_230193


namespace NUMINAMATH_CALUDE_peanuts_in_jar_l2301_230186

theorem peanuts_in_jar (initial_peanuts : ℕ) : 
  (initial_peanuts : ℚ) - (1/4 : ℚ) * initial_peanuts - 29 = 82 → 
  initial_peanuts = 148 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_jar_l2301_230186


namespace NUMINAMATH_CALUDE_anika_maddie_age_ratio_l2301_230174

def anika_age : ℕ := 30
def future_years : ℕ := 15
def future_average_age : ℕ := 50

theorem anika_maddie_age_ratio :
  ∃ (maddie_age : ℕ),
    (anika_age + future_years + maddie_age + future_years) / 2 = future_average_age ∧
    anika_age * 4 = maddie_age * 3 := by
  sorry

end NUMINAMATH_CALUDE_anika_maddie_age_ratio_l2301_230174


namespace NUMINAMATH_CALUDE_max_product_constraint_max_product_achievable_l2301_230128

theorem max_product_constraint (x y : ℕ+) (h : 7 * x + 4 * y = 150) : x * y ≤ 200 := by
  sorry

theorem max_product_achievable : ∃ (x y : ℕ+), 7 * x + 4 * y = 150 ∧ x * y = 200 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_max_product_achievable_l2301_230128


namespace NUMINAMATH_CALUDE_triangle_area_proof_l2301_230172

/-- Given a triangle DEF with inradius r, circumradius R, and angles D, E, F,
    prove that if r = 10, R = 25, and 2cos(E) = cos(D) + cos(F),
    then the area of the triangle is 225√51/5 -/
theorem triangle_area_proof (D E F : ℝ) (r R : ℝ) :
  r = 10 →
  R = 25 →
  2 * Real.cos E = Real.cos D + Real.cos F →
  ∃ (d e f : ℝ),
    d > 0 ∧ e > 0 ∧ f > 0 ∧
    e^2 = d^2 + f^2 - 2*d*f*(Real.cos E) ∧
    Real.cos D = (f^2 + e^2 - d^2) / (2*f*e) ∧
    Real.cos F = (d^2 + e^2 - f^2) / (2*d*e) ∧
    (d + e + f) / 2 * r = 225 * Real.sqrt 51 / 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l2301_230172


namespace NUMINAMATH_CALUDE_inequality_solution_sum_l2301_230141

theorem inequality_solution_sum (m n : ℝ) : 
  (∀ x, x ∈ Set.Ioo m n ↔ (m * x - 1) / (x + 3) > 0) →
  m + n = -10/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sum_l2301_230141


namespace NUMINAMATH_CALUDE_triangle_sides_theorem_l2301_230132

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_sides_theorem (x : ℕ+) :
  triangle_exists 8 11 (x.val ^ 2) ↔ x.val = 2 ∨ x.val = 3 ∨ x.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_theorem_l2301_230132


namespace NUMINAMATH_CALUDE_batsman_average_is_37_l2301_230161

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  averageIncrease : ℕ
  lastInningScore : ℕ

/-- Calculates the new average score after the latest inning -/
def newAverage (b : Batsman) : ℚ :=
  (b.totalRuns + b.lastInningScore) / b.innings

/-- Theorem: Given the conditions, prove that the new average is 37 -/
theorem batsman_average_is_37 (b : Batsman)
    (h1 : b.innings = 17)
    (h2 : b.lastInningScore = 85)
    (h3 : b.averageIncrease = 3)
    (h4 : newAverage b = (b.totalRuns / (b.innings - 1) + b.averageIncrease)) :
    newAverage b = 37 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_is_37_l2301_230161


namespace NUMINAMATH_CALUDE_tan_75_degrees_l2301_230140

theorem tan_75_degrees : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  -- We define tan 75° as tan(90° - 15°)
  have h1 : Real.tan (75 * π / 180) = Real.tan ((90 - 15) * π / 180) := by sorry
  
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_tan_75_degrees_l2301_230140


namespace NUMINAMATH_CALUDE_average_age_increase_l2301_230133

theorem average_age_increase (initial_count : ℕ) (replaced_count : ℕ) (age1 age2 : ℕ) (women_avg_age : ℚ) : 
  initial_count = 7 →
  replaced_count = 2 →
  age1 = 18 →
  age2 = 22 →
  women_avg_age = 30.5 →
  (2 * women_avg_age - (age1 + age2 : ℚ)) / initial_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l2301_230133


namespace NUMINAMATH_CALUDE_square_cut_perimeter_l2301_230187

theorem square_cut_perimeter (square_side : ℝ) (total_perimeter : ℝ) :
  square_side = 4 →
  total_perimeter = 25 →
  ∃ (rect1_length rect1_width rect2_length rect2_width : ℝ),
    rect1_length * rect1_width + rect2_length * rect2_width = square_side * square_side ∧
    2 * (rect1_length + rect1_width) + 2 * (rect2_length + rect2_width) = total_perimeter :=
by sorry

end NUMINAMATH_CALUDE_square_cut_perimeter_l2301_230187


namespace NUMINAMATH_CALUDE_trig_identity_equivalence_l2301_230166

theorem trig_identity_equivalence (x : ℝ) :
  (2 * (Real.cos (4 * x) - Real.sin x * Real.cos (3 * x)) = Real.sin (4 * x) + Real.sin (2 * x)) ↔
  (∃ k : ℤ, x = (π / 16) * (4 * ↑k + 1)) :=
sorry

end NUMINAMATH_CALUDE_trig_identity_equivalence_l2301_230166


namespace NUMINAMATH_CALUDE_units_digit_of_sum_factorials_l2301_230177

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sum_factorials :
  units_digit (sum_of_factorials 50) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_factorials_l2301_230177


namespace NUMINAMATH_CALUDE_smallest_positive_integer_l2301_230106

theorem smallest_positive_integer : ∀ n : ℕ, n > 0 → n ≥ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_l2301_230106


namespace NUMINAMATH_CALUDE_min_sum_with_condition_min_sum_equality_l2301_230145

theorem min_sum_with_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  a + b ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_sum_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  a + b = 3 + 2 * Real.sqrt 2 ↔ a = 1 + Real.sqrt 2 ∧ b = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_condition_min_sum_equality_l2301_230145


namespace NUMINAMATH_CALUDE_sin_2x_given_cos_l2301_230190

theorem sin_2x_given_cos (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) : 
  Real.sin (2 * x) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_given_cos_l2301_230190


namespace NUMINAMATH_CALUDE_paul_dog_food_needed_l2301_230111

/-- The amount of dog food needed per day for a given weight in pounds -/
def dogFoodNeeded (weight : ℕ) : ℕ := weight / 10

/-- The total weight of Paul's dogs in pounds -/
def totalDogWeight : ℕ := 20 + 40 + 10 + 30 + 50

/-- Theorem: Paul needs 15 pounds of dog food per day for his five dogs -/
theorem paul_dog_food_needed : dogFoodNeeded totalDogWeight = 15 := by
  sorry

end NUMINAMATH_CALUDE_paul_dog_food_needed_l2301_230111


namespace NUMINAMATH_CALUDE_bob_questions_theorem_l2301_230114

/-- Represents the number of questions Bob creates in each hour -/
def questions_per_hour : Fin 3 → ℕ
  | 0 => 13
  | 1 => 13 * 2
  | 2 => 13 * 2 * 2

/-- The total number of questions Bob creates in three hours -/
def total_questions : ℕ := (questions_per_hour 0) + (questions_per_hour 1) + (questions_per_hour 2)

theorem bob_questions_theorem :
  total_questions = 91 := by
  sorry

end NUMINAMATH_CALUDE_bob_questions_theorem_l2301_230114


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l2301_230160

theorem quadratic_real_roots_range (m : ℝ) :
  (∃ x : ℝ, x^2 - x - m = 0) ↔ m ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l2301_230160


namespace NUMINAMATH_CALUDE_midpoint_triangle_perimeter_l2301_230148

/-- A right prism with regular pentagonal bases -/
structure RightPrism :=
  (height : ℝ)
  (base_side_length : ℝ)

/-- Midpoint of an edge -/
structure Midpoint :=
  (edge : String)

/-- Triangle formed by three midpoints -/
structure MidpointTriangle :=
  (p1 : Midpoint)
  (p2 : Midpoint)
  (p3 : Midpoint)

/-- Calculate the perimeter of the midpoint triangle -/
def perimeter (prism : RightPrism) (triangle : MidpointTriangle) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the midpoint triangle -/
theorem midpoint_triangle_perimeter 
  (prism : RightPrism) 
  (triangle : MidpointTriangle) 
  (h1 : prism.height = 25) 
  (h2 : prism.base_side_length = 15) 
  (h3 : triangle.p1 = Midpoint.mk "AB") 
  (h4 : triangle.p2 = Midpoint.mk "BC") 
  (h5 : triangle.p3 = Midpoint.mk "CD") : 
  perimeter prism triangle = 15 + 2 * Real.sqrt 212.5 :=
sorry

end NUMINAMATH_CALUDE_midpoint_triangle_perimeter_l2301_230148


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l2301_230199

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l2301_230199


namespace NUMINAMATH_CALUDE_lucy_shells_count_l2301_230198

/-- The number of shells Lucy initially had -/
def initial_shells : ℕ := 68

/-- The number of additional shells Lucy found -/
def additional_shells : ℕ := 21

/-- The total number of shells Lucy has now -/
def total_shells : ℕ := initial_shells + additional_shells

theorem lucy_shells_count : total_shells = 89 := by
  sorry

end NUMINAMATH_CALUDE_lucy_shells_count_l2301_230198

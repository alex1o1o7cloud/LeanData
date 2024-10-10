import Mathlib

namespace range_of_m_range_of_x_l3973_397366

/-- Given m > 0, p: (x+2)(x-6) ≤ 0, and q: 2-m ≤ x ≤ 2+m -/
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0

def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

/-- If p is a necessary condition for q, then 0 < m ≤ 4 -/
theorem range_of_m (m : ℝ) (h : m > 0) :
  (∀ x, q m x → p x) → 0 < m ∧ m ≤ 4 := by sorry

/-- Given m = 2, if ¬p ∨ ¬q is false, then 0 ≤ x ≤ 4 -/
theorem range_of_x (x : ℝ) :
  ¬(¬(p x) ∨ ¬(q 2 x)) → 0 ≤ x ∧ x ≤ 4 := by sorry

end range_of_m_range_of_x_l3973_397366


namespace pentagon_angle_sum_l3973_397374

theorem pentagon_angle_sum (A B C D E : ℝ) (x y : ℝ) : 
  A = 34 → 
  B = 70 → 
  C = 30 → 
  D = 90 → 
  A + B + C + D + E = 540 → 
  E = 360 - x → 
  180 - y = 120 →
  x + y = 134 := by sorry

end pentagon_angle_sum_l3973_397374


namespace student_count_proof_l3973_397369

def total_students (group1 group2 group3 group4 : ℕ) : ℕ :=
  group1 + group2 + group3 + group4

theorem student_count_proof :
  let group1 : ℕ := 5
  let group2 : ℕ := 8
  let group3 : ℕ := 7
  let group4 : ℕ := 4
  total_students group1 group2 group3 group4 = 24 := by
  sorry

end student_count_proof_l3973_397369


namespace G_difference_l3973_397359

/-- G is defined as the infinite repeating decimal 0.737373... -/
def G : ℚ := 73 / 99

/-- The difference between the denominator and numerator of G when expressed as a fraction in lowest terms -/
def difference : ℕ := 99 - 73

theorem G_difference : difference = 26 := by sorry

end G_difference_l3973_397359


namespace triangle_acute_angled_l3973_397339

theorem triangle_acute_angled (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (eq : a^3 + b^3 = c^3) :
  c^2 < a^2 + b^2 := by
sorry

end triangle_acute_angled_l3973_397339


namespace square_root_problem_l3973_397385

theorem square_root_problem (a b : ℝ) 
  (h1 : (2 * a + 1) = 9)
  (h2 : (5 * a + 2 * b - 2) = 16) :
  (3 * a - 4 * b) = 16 := by
sorry

end square_root_problem_l3973_397385


namespace train_length_l3973_397324

/-- Calculates the length of a train given its speed, the speed of a vehicle it overtakes, and the time it takes to overtake. -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) : 
  train_speed = 100 →
  motorbike_speed = 64 →
  overtake_time = 40 →
  (train_speed - motorbike_speed) * overtake_time * (1000 / 3600) = 400 :=
by sorry

end train_length_l3973_397324


namespace pencil_color_fractions_l3973_397310

theorem pencil_color_fractions (L : ℝ) (h1 : L = 9.333333333333332) : 
  let black_fraction : ℝ := 1/8
  let remaining_after_black : ℝ := L - black_fraction * L
  let blue_fraction_of_remaining : ℝ := 7/12
  let white_fraction_of_remaining : ℝ := 1 - blue_fraction_of_remaining
  white_fraction_of_remaining = 5/12 := by
sorry

end pencil_color_fractions_l3973_397310


namespace arithmetic_sequence_property_l3973_397312

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Theorem: In an arithmetic sequence, if a_4 + a_6 + a_8 + a_10 + a_12 = 120, then 2a_10 - a_12 = 24 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) : 
  2 * a 10 - a 12 = 24 := by
  sorry

end arithmetic_sequence_property_l3973_397312


namespace connor_date_cost_l3973_397362

/-- Calculates the total cost of Connor's movie date --/
def movie_date_cost (ticket_price : ℚ) (combo_price : ℚ) (candy_price : ℚ) : ℚ :=
  2 * ticket_price + combo_price + 2 * candy_price

/-- Theorem stating that the total cost of Connor's movie date is $36.00 --/
theorem connor_date_cost :
  movie_date_cost 10 11 2.5 = 36 :=
by sorry

end connor_date_cost_l3973_397362


namespace complementary_angles_proof_l3973_397364

theorem complementary_angles_proof (A B : Real) : 
  A + B = 90 →  -- Angles A and B are complementary
  A = 4 * B →   -- Measure of angle A is 4 times angle B
  A = 72 ∧ B = 18 := by
sorry

end complementary_angles_proof_l3973_397364


namespace happy_valley_kennel_arrangements_l3973_397349

/-- The number of ways to arrange animals in cages -/
def arrange_animals (chickens dogs cats rabbits : ℕ) : ℕ :=
  (Nat.factorial 4) * 
  (Nat.factorial chickens) * 
  (Nat.factorial dogs) * 
  (Nat.factorial cats) * 
  (Nat.factorial rabbits)

/-- Theorem stating the correct number of arrangements for the given problem -/
theorem happy_valley_kennel_arrangements :
  arrange_animals 4 3 5 2 = 414720 := by
  sorry

end happy_valley_kennel_arrangements_l3973_397349


namespace negation_of_exists_proposition_l3973_397344

theorem negation_of_exists_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end negation_of_exists_proposition_l3973_397344


namespace theo_has_winning_strategy_l3973_397371

/-- Game state representing the current turn number and cumulative score -/
structure GameState where
  turn : ℕ
  score : ℕ

/-- Player type -/
inductive Player
| Anatole
| Theo

/-- Game move representing the chosen number and resulting turn score -/
structure GameMove where
  number : ℕ
  turn_score : ℕ

/-- Represents a strategy for a player -/
def Strategy := GameState → GameMove

/-- Checks if a move is valid according to game rules -/
def is_valid_move (p : ℕ) (prev_move : Option GameMove) (current_move : GameMove) : Prop :=
  match prev_move with
  | none => current_move.number > 0
  | some prev => current_move.number > prev.number

/-- Checks if a player wins with a given move -/
def is_winning_move (p : ℕ) (state : GameState) (move : GameMove) : Prop :=
  (p ∣ (move.turn_score * (state.score + state.turn * move.turn_score)))

/-- Theorem stating that Theo has a winning strategy -/
theorem theo_has_winning_strategy (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2) :
  ∃ (theo_strategy : Strategy),
    ∀ (anatole_strategy : Strategy),
      ∃ (final_state : GameState),
        final_state.turn < p - 1 ∧
        is_winning_move p final_state (theo_strategy final_state) :=
  sorry

end theo_has_winning_strategy_l3973_397371


namespace cuboid_max_volume_l3973_397306

/-- The maximum volume of a cuboid with a total edge length of 60 units is 125 cubic units. -/
theorem cuboid_max_volume :
  ∀ x y z : ℝ,
  x > 0 → y > 0 → z > 0 →
  4 * (x + y + z) = 60 →
  x * y * z ≤ 125 :=
by sorry

end cuboid_max_volume_l3973_397306


namespace right_triangle_geometric_mean_l3973_397395

theorem right_triangle_geometric_mean (a c : ℝ) (h₁ : 0 < a) (h₂ : 0 < c) :
  (c * c = a * c) → (a = (c * (Real.sqrt 5 - 1)) / 2) :=
by sorry

end right_triangle_geometric_mean_l3973_397395


namespace nonagon_diagonal_intersection_probability_l3973_397340

/-- A regular nonagon is a 9-sided polygon with all sides and angles equal. -/
def RegularNonagon : Type := Unit

/-- A diagonal of a regular nonagon is a line segment connecting two non-adjacent vertices. -/
def Diagonal (n : RegularNonagon) : Type := Unit

/-- The set of all diagonals in a regular nonagon. -/
def AllDiagonals (n : RegularNonagon) : Set (Diagonal n) := sorry

/-- Two diagonals intersect if they cross each other inside the nonagon. -/
def Intersect (n : RegularNonagon) (d1 d2 : Diagonal n) : Prop := sorry

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes. -/
def Probability (n : RegularNonagon) (event : Set (Diagonal n × Diagonal n)) : ℚ := sorry

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  Probability n {p : Diagonal n × Diagonal n | Intersect n p.1 p.2} = 14/39 := by sorry

end nonagon_diagonal_intersection_probability_l3973_397340


namespace midpoint_coordinate_sum_l3973_397365

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, 16) and (-2, -8) is 7. -/
theorem midpoint_coordinate_sum : 
  let p1 : ℝ × ℝ := (8, 16)
  let p2 : ℝ × ℝ := (-2, -8)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 + midpoint.2 = 7 := by sorry

end midpoint_coordinate_sum_l3973_397365


namespace modulus_of_Z_l3973_397331

/-- The modulus of the complex number Z = 1 / (i - 1) is equal to √2/2 -/
theorem modulus_of_Z : Complex.abs (1 / (Complex.I - 1)) = Real.sqrt 2 / 2 := by sorry

end modulus_of_Z_l3973_397331


namespace perpendicular_transitivity_l3973_397347

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)

-- Define the statement
theorem perpendicular_transitivity 
  (m n : Line) (α β : Plane) 
  (hm_ne_n : m ≠ n) 
  (hα_ne_β : α ≠ β) 
  (hm_perp_α : perp m α) 
  (hm_perp_β : perp m β) 
  (hn_perp_α : perp n α) : 
  perp n β := by sorry

end perpendicular_transitivity_l3973_397347


namespace x_range_l3973_397393

theorem x_range (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -2) : x > 1 / 3 := by
  sorry

end x_range_l3973_397393


namespace log_equation_solution_l3973_397321

theorem log_equation_solution (x : ℝ) :
  0 < x ∧ x ≠ 1 ∧ x < 10 →
  (1 + 2 * (Real.log 2 / Real.log x) * (Real.log (10 - x) / Real.log 4) = 2 / (Real.log x / Real.log 4)) ↔
  (x = 2 ∨ x = 8) :=
by sorry

end log_equation_solution_l3973_397321


namespace vector_on_line_k_value_l3973_397396

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def line_through (a b : V) : ℝ → V :=
  λ t => a + t • (b - a)

theorem vector_on_line_k_value
  (a b : V) (ha_ne_b : a ≠ b) (k : ℝ) :
  (∃ t : ℝ, line_through a b t = k • a + (5/7 : ℝ) • b) →
  k = 5/7 := by
  sorry

end vector_on_line_k_value_l3973_397396


namespace sum_of_integers_l3973_397357

theorem sum_of_integers (x y : ℕ+) (h1 : x - y = 10) (h2 : x * y = 56) : x + y = 18 := by
  sorry

end sum_of_integers_l3973_397357


namespace five_congruent_subtriangles_impossible_l3973_397304

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : a > 0
  positive_b : b > 0
  positive_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define a subdivision of a triangle into five smaller triangles
structure SubdividedTriangle where
  main : Triangle
  sub1 : Triangle
  sub2 : Triangle
  sub3 : Triangle
  sub4 : Triangle
  sub5 : Triangle

-- Theorem statement
theorem five_congruent_subtriangles_impossible (t : SubdividedTriangle) :
  ¬(t.sub1 = t.sub2 ∧ t.sub2 = t.sub3 ∧ t.sub3 = t.sub4 ∧ t.sub4 = t.sub5) :=
by sorry

end five_congruent_subtriangles_impossible_l3973_397304


namespace cornelia_asian_countries_l3973_397383

theorem cornelia_asian_countries (total : ℕ) (europe : ℕ) (south_america : ℕ) 
  (h1 : total = 42)
  (h2 : europe = 20)
  (h3 : south_america = 10)
  (h4 : (total - europe - south_america) % 2 = 0) :
  (total - europe - south_america) / 2 = 6 := by
sorry

end cornelia_asian_countries_l3973_397383


namespace clothes_transport_expenditure_l3973_397386

/-- Calculates the monthly amount spent on clothes and transport given the yearly savings --/
def monthly_clothes_transport (yearly_savings : ℕ) : ℕ :=
  let monthly_savings := yearly_savings / 12
  let monthly_salary := monthly_savings * 5
  monthly_salary / 5

/-- Theorem stating that given the conditions in the problem, 
    the monthly amount spent on clothes and transport is 4038 --/
theorem clothes_transport_expenditure :
  monthly_clothes_transport 48456 = 4038 := by
  sorry

#eval monthly_clothes_transport 48456

end clothes_transport_expenditure_l3973_397386


namespace exactly_one_even_iff_not_all_odd_or_two_even_l3973_397354

def exactly_one_even (a b c : ℕ) : Prop :=
  (Even a ∧ Odd b ∧ Odd c) ∨
  (Odd a ∧ Even b ∧ Odd c) ∨
  (Odd a ∧ Odd b ∧ Even c)

def all_odd_or_two_even (a b c : ℕ) : Prop :=
  (Odd a ∧ Odd b ∧ Odd c) ∨
  (Even a ∧ Even b) ∨
  (Even a ∧ Even c) ∨
  (Even b ∧ Even c)

theorem exactly_one_even_iff_not_all_odd_or_two_even (a b c : ℕ) :
  exactly_one_even a b c ↔ ¬(all_odd_or_two_even a b c) :=
sorry

end exactly_one_even_iff_not_all_odd_or_two_even_l3973_397354


namespace max_value_sqrt_sum_l3973_397380

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 7) :
  ∃ (M : ℝ), M = 6 * Real.sqrt 5 ∧ 
  Real.sqrt (3 * x + 4) + Real.sqrt (3 * y + 4) + Real.sqrt (3 * z + 4) ≤ M ∧
  ∃ (x' y' z' : ℝ), x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 7 ∧
    Real.sqrt (3 * x' + 4) + Real.sqrt (3 * y' + 4) + Real.sqrt (3 * z' + 4) = M :=
by
  sorry

end max_value_sqrt_sum_l3973_397380


namespace base9_246_to_base10_l3973_397397

/-- Converts a three-digit number from base 9 to base 10 -/
def base9ToBase10 (d2 d1 d0 : Nat) : Nat :=
  d2 * 9^2 + d1 * 9^1 + d0 * 9^0

/-- The base 10 representation of 246 in base 9 is 204 -/
theorem base9_246_to_base10 : base9ToBase10 2 4 6 = 204 := by
  sorry

end base9_246_to_base10_l3973_397397


namespace empire_state_building_height_l3973_397315

/-- The height of the Empire State Building to the top floor -/
def height_to_top_floor : ℝ := 1454 - 204

/-- The total height of the Empire State Building -/
def total_height : ℝ := 1454

/-- The height of the antenna spire -/
def antenna_height : ℝ := 204

theorem empire_state_building_height : height_to_top_floor = 1250 := by
  sorry

end empire_state_building_height_l3973_397315


namespace opposite_of_gold_is_olive_l3973_397308

-- Define the colors
inductive Color
  | Aqua | Maroon | Olive | Purple | Silver | Gold | Black

-- Define the cube faces
structure CubeFace where
  color : Color

-- Define the cube
structure Cube where
  faces : List CubeFace
  gold_face : CubeFace
  opposite_face : CubeFace

-- Define the cross pattern
structure CrossPattern where
  squares : List CubeFace

-- Function to fold the cross pattern into a cube
def fold_cross_to_cube (cross : CrossPattern) : Cube :=
  sorry

-- Theorem: The face opposite to Gold is Olive
theorem opposite_of_gold_is_olive (cross : CrossPattern) 
  (cube : Cube := fold_cross_to_cube cross) : 
  cube.gold_face.color = Color.Gold → cube.opposite_face.color = Color.Olive :=
sorry

end opposite_of_gold_is_olive_l3973_397308


namespace tan_value_problem_l3973_397353

theorem tan_value_problem (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π) 
  (h3 : Real.sin θ + Real.cos θ = 1/5) : 
  Real.tan θ = -4/3 := by
  sorry

end tan_value_problem_l3973_397353


namespace copying_result_correct_l3973_397316

/-- Represents the copying cost and discount structure -/
structure CopyingCost where
  cost_per_5_pages : ℚ  -- Cost in cents for 5 pages
  budget : ℚ           -- Budget in dollars
  discount_rate : ℚ    -- Discount rate after 1000 pages
  discount_threshold : ℕ -- Number of pages after which discount applies

/-- Calculates the total number of pages that can be copied and the total cost with discount -/
def calculate_copying_result (c : CopyingCost) : ℕ × ℚ :=
  sorry

/-- Theorem stating the correctness of the calculation -/
theorem copying_result_correct (c : CopyingCost) :
  c.cost_per_5_pages = 10 ∧ 
  c.budget = 50 ∧ 
  c.discount_rate = 0.1 ∧
  c.discount_threshold = 1000 →
  calculate_copying_result c = (2500, 47) :=
sorry

end copying_result_correct_l3973_397316


namespace value_of_b_l3973_397363

theorem value_of_b (b : ℚ) (h : b + b/4 = 3) : b = 12/5 := by
  sorry

end value_of_b_l3973_397363


namespace factorial_quotient_l3973_397368

theorem factorial_quotient : Nat.factorial 50 / Nat.factorial 47 = 117600 := by
  sorry

end factorial_quotient_l3973_397368


namespace stamps_per_page_l3973_397326

theorem stamps_per_page (a b c : ℕ) (ha : a = 945) (hb : b = 1260) (hc : c = 630) :
  Nat.gcd a (Nat.gcd b c) = 315 := by
  sorry

end stamps_per_page_l3973_397326


namespace g_6_equals_666_l3973_397388

def g (x : ℝ) : ℝ := 3*x^4 - 19*x^3 + 31*x^2 - 27*x - 72

theorem g_6_equals_666 : g 6 = 666 := by
  sorry

end g_6_equals_666_l3973_397388


namespace ellipse_foci_y_axis_m_range_l3973_397350

/-- 
Given an equation of the form x²/(4-m) + y²/(m-3) = 1 representing an ellipse with foci on the y-axis,
prove that the range of m is (7/2, 4).
-/
theorem ellipse_foci_y_axis_m_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2/(4-m) + y^2/(m-3) = 1) ∧ 
  (∀ (x y : ℝ), x^2/(4-m) + y^2/(m-3) = 1 → (0 : ℝ) < 4-m ∧ (0 : ℝ) < m-3 ∧ m-3 < 4-m) 
  → 7/2 < m ∧ m < 4 :=
sorry

end ellipse_foci_y_axis_m_range_l3973_397350


namespace intersection_of_M_and_N_l3973_397342

def M : Set ℝ := {x | 4 ≤ x ∧ x ≤ 7}
def N : Set ℝ := {3, 5, 8}

theorem intersection_of_M_and_N : M ∩ N = {5} := by
  sorry

end intersection_of_M_and_N_l3973_397342


namespace product_digit_sum_equals_800_l3973_397389

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Represents a number with n repeated digits of 7 -/
def repeated_sevens (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem product_digit_sum_equals_800 :
  sum_of_digits (8 * repeated_sevens 788) = 800 := by sorry

end product_digit_sum_equals_800_l3973_397389


namespace smallest_m_for_integral_solutions_l3973_397378

theorem smallest_m_for_integral_solutions : 
  ∃ (m : ℕ), m > 0 ∧ 
  (∃ (x : ℤ), 18 * x^2 - m * x + 252 = 0) ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < m → ¬∃ (y : ℤ), 18 * y^2 - k * y + 252 = 0) ∧ 
  m = 162 := by
sorry

end smallest_m_for_integral_solutions_l3973_397378


namespace cosine_sum_problem_l3973_397319

theorem cosine_sum_problem (α : Real) 
  (h : Real.sin (π / 2 + α) = 1 / 3) : 
  Real.cos (2 * α) + Real.cos α = -4 / 9 := by
  sorry

end cosine_sum_problem_l3973_397319


namespace parabola_shift_l3973_397311

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 3

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := original_parabola (x + 2) + 2

-- Theorem statement
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = 3 * x^2 + 6 * x - 1 :=
by
  sorry

end parabola_shift_l3973_397311


namespace book_price_increase_l3973_397356

theorem book_price_increase (initial_price decreased_price final_price : ℝ) 
  (h1 : initial_price = 400)
  (h2 : decreased_price = initial_price * (1 - 0.15))
  (h3 : final_price = 476) :
  (final_price - decreased_price) / decreased_price = 0.4 := by
sorry

end book_price_increase_l3973_397356


namespace pet_shop_ducks_l3973_397314

theorem pet_shop_ducks (total : ℕ) (cats : ℕ) (ducks : ℕ) (parrots : ℕ) : 
  cats = 56 →
  ducks = total / 12 →
  ducks = (ducks + parrots) / 4 →
  total = cats + ducks + parrots →
  ducks = 7 := by
sorry

end pet_shop_ducks_l3973_397314


namespace imaginary_part_of_complex_fraction_l3973_397328

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I)
  Complex.im z = -1 := by sorry

end imaginary_part_of_complex_fraction_l3973_397328


namespace ad_rate_per_square_foot_l3973_397329

-- Define the problem parameters
def num_companies : ℕ := 3
def ads_per_company : ℕ := 10
def ad_length : ℕ := 12
def ad_width : ℕ := 5
def total_paid : ℕ := 108000

-- Define the theorem
theorem ad_rate_per_square_foot :
  let total_area : ℕ := num_companies * ads_per_company * ad_length * ad_width
  let rate_per_square_foot : ℚ := total_paid / total_area
  rate_per_square_foot = 60 := by
  sorry

end ad_rate_per_square_foot_l3973_397329


namespace rational_numbers_countable_l3973_397392

theorem rational_numbers_countable : ∃ f : ℚ → ℕ+, Function.Bijective f := by
  sorry

end rational_numbers_countable_l3973_397392


namespace consecutive_odd_integers_expressions_l3973_397382

theorem consecutive_odd_integers_expressions (p q : ℤ) 
  (h1 : ∃ k : ℤ, p = 2*k + 1 ∧ q = 2*k + 3) : 
  Odd (2*p + 5*q) ∧ Odd (5*p - 2*q) ∧ Odd (2*p*q + 5) := by
  sorry

end consecutive_odd_integers_expressions_l3973_397382


namespace adult_tickets_sold_l3973_397361

/-- Given the prices of adult and child tickets, the total number of tickets sold,
    and the total revenue, prove the number of adult tickets sold. -/
theorem adult_tickets_sold
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : total_revenue = 5100)
  : ∃ (adult_tickets : ℕ),
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_revenue ∧
    adult_tickets = 500 := by
  sorry

end adult_tickets_sold_l3973_397361


namespace cubic_sum_inequality_l3973_397317

theorem cubic_sum_inequality (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_of_squares : a^2 + b^2 + c^2 + d^2 = 4) : 
  a^3 + b^3 + c^3 + d^3 + a*b*c + b*c*d + c*d*a + d*a*b ≤ 8 := by
sorry

end cubic_sum_inequality_l3973_397317


namespace min_value_expression_l3973_397334

theorem min_value_expression (x y z : ℝ) (h1 : x * y ≠ 0) (h2 : x + y ≠ 0) :
  ((y + z) / x + 2)^2 + (z / y + 2)^2 + (z / (x + y) - 1)^2 ≥ 5 ∧
  ∃ (x y z : ℝ), x * y ≠ 0 ∧ x + y ≠ 0 ∧
    ((y + z) / x + 2)^2 + (z / y + 2)^2 + (z / (x + y) - 1)^2 = 5 :=
by sorry

end min_value_expression_l3973_397334


namespace books_gotten_rid_of_l3973_397367

def initial_stock : ℕ := 27
def shelves_used : ℕ := 3
def books_per_shelf : ℕ := 7

theorem books_gotten_rid_of : 
  initial_stock - (shelves_used * books_per_shelf) = 6 := by
sorry

end books_gotten_rid_of_l3973_397367


namespace arithmetic_calculation_l3973_397373

theorem arithmetic_calculation : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end arithmetic_calculation_l3973_397373


namespace juniper_has_six_bones_l3973_397341

/-- Calculates the number of bones Juniper has remaining after her master doubles 
    her initial number of bones and the neighbor's dog steals two bones. -/
def junipersBones (initialBones : ℕ) : ℕ :=
  2 * initialBones - 2

/-- Theorem stating that Juniper has 6 bones remaining after the events. -/
theorem juniper_has_six_bones : junipersBones 4 = 6 := by
  sorry

end juniper_has_six_bones_l3973_397341


namespace alla_boris_meeting_point_l3973_397323

/-- The number of lanterns along the alley -/
def total_lanterns : ℕ := 400

/-- Alla's position when the first observation is made -/
def alla_position : ℕ := 55

/-- Boris's position when the first observation is made -/
def boris_position : ℕ := 321

/-- The meeting point of Alla and Boris -/
def meeting_point : ℕ := 163

/-- Theorem stating that Alla and Boris will meet at the calculated meeting point -/
theorem alla_boris_meeting_point :
  ∀ (alla_start boris_start : ℕ),
  alla_start = 1 ∧ boris_start = total_lanterns ∧
  alla_position > alla_start ∧ boris_position < boris_start ∧
  (alla_position - alla_start) / (total_lanterns - alla_position - (boris_start - boris_position)) =
  (meeting_point - alla_start) / (boris_start - meeting_point) :=
by sorry

end alla_boris_meeting_point_l3973_397323


namespace age_ratio_proof_l3973_397330

def arun_future_age : ℕ := 26
def years_to_future : ℕ := 6
def deepak_current_age : ℕ := 15

theorem age_ratio_proof :
  let arun_current_age := arun_future_age - years_to_future
  (arun_current_age : ℚ) / deepak_current_age = 4 / 3 := by
  sorry

end age_ratio_proof_l3973_397330


namespace max_black_pieces_l3973_397375

/-- Represents a piece color -/
inductive Color
| Black
| White

/-- Represents the state of the circle -/
def CircleState := List Color

/-- Applies the rule to place new pieces between existing ones -/
def applyRule (state : CircleState) : CircleState :=
  sorry

/-- Removes the original pieces from the circle -/
def removeOriginal (state : CircleState) : CircleState :=
  sorry

/-- Counts the number of black pieces in the circle -/
def countBlack (state : CircleState) : Nat :=
  sorry

/-- The main theorem stating that the maximum number of black pieces is 4 -/
theorem max_black_pieces (initial : CircleState) : 
  initial.length = 5 → 
  ∀ (n : Nat), countBlack (removeOriginal (applyRule initial)) ≤ 4 :=
sorry

end max_black_pieces_l3973_397375


namespace a_annual_income_l3973_397351

/-- Proves that A's annual income is 403200 given the specified conditions -/
theorem a_annual_income (c_income : ℕ) (h1 : c_income = 12000) : ∃ (a_income b_income : ℕ),
  (a_income : ℚ) / b_income = 5 / 2 ∧
  b_income = c_income + c_income * 12 / 100 ∧
  a_income * 12 = 403200 :=
by sorry

end a_annual_income_l3973_397351


namespace total_sharks_l3973_397360

theorem total_sharks (newport_sharks : ℕ) (dana_point_sharks : ℕ) : 
  newport_sharks = 22 → 
  dana_point_sharks = 4 * newport_sharks → 
  newport_sharks + dana_point_sharks = 110 := by
sorry

end total_sharks_l3973_397360


namespace john_annual_oil_change_cost_l3973_397387

/-- Calculates the annual cost of oil changes for a driver named John. -/
theorem john_annual_oil_change_cost :
  ∀ (miles_per_month : ℕ) 
    (miles_per_oil_change : ℕ) 
    (free_oil_changes_per_year : ℕ) 
    (cost_per_oil_change : ℕ),
  miles_per_month = 1000 →
  miles_per_oil_change = 3000 →
  free_oil_changes_per_year = 1 →
  cost_per_oil_change = 50 →
  (12 * miles_per_month / miles_per_oil_change - free_oil_changes_per_year) * cost_per_oil_change = 150 :=
by
  sorry

#check john_annual_oil_change_cost

end john_annual_oil_change_cost_l3973_397387


namespace blue_marbles_count_l3973_397346

theorem blue_marbles_count (yellow green black : ℕ) (total : ℕ) (prob_black : ℚ) :
  yellow = 12 →
  green = 5 →
  black = 1 →
  prob_black = 1 / 28 →
  total = yellow + green + black + (total - yellow - green - black) →
  prob_black = black / total →
  (total - yellow - green - black) = 10 := by
  sorry

end blue_marbles_count_l3973_397346


namespace video_subscription_cost_l3973_397345

def monthly_cost : ℚ := 14
def num_people : ℕ := 2
def months_in_year : ℕ := 12

theorem video_subscription_cost :
  (monthly_cost / num_people) * months_in_year = 84 := by
sorry

end video_subscription_cost_l3973_397345


namespace product_of_integers_l3973_397379

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 20)
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x * y = 99 := by
  sorry

end product_of_integers_l3973_397379


namespace eight_digit_even_increasing_numbers_l3973_397305

theorem eight_digit_even_increasing_numbers (n : ℕ) (k : ℕ) : 
  n = 8 ∧ k = 4 → (n + k - 1).choose (k - 1) = 165 := by
  sorry

end eight_digit_even_increasing_numbers_l3973_397305


namespace max_sum_solution_l3973_397309

theorem max_sum_solution : ∃ (a b : ℕ), 
  (2 * a * b + 3 * b = b^2 + 6 * a + 6) ∧ 
  (∀ (x y : ℕ), (2 * x * y + 3 * y = y^2 + 6 * x + 6) → (x + y ≤ a + b)) ∧
  a = 5 ∧ b = 9 := by
  sorry

end max_sum_solution_l3973_397309


namespace lattice_fifth_number_ninth_row_l3973_397335

/-- Given a lattice with 7 numbers in each row, continuing for 9 rows,
    the fifth number in the 9th row is 60. -/
theorem lattice_fifth_number_ninth_row :
  ∀ (lattice : ℕ → ℕ → ℕ),
    (∀ row col, col ≤ 7 → lattice row col = row * col) →
    lattice 9 5 = 60 := by
sorry

end lattice_fifth_number_ninth_row_l3973_397335


namespace other_root_of_quadratic_l3973_397318

/-- Given that x = 1 is a root of the quadratic equation x^2 + bx - 2 = 0,
    prove that the other root is -2 -/
theorem other_root_of_quadratic (b : ℝ) : 
  (1^2 + b*1 - 2 = 0) → ∃ x : ℝ, x ≠ 1 ∧ x^2 + b*x - 2 = 0 ∧ x = -2 :=
by sorry

end other_root_of_quadratic_l3973_397318


namespace monkey_climb_l3973_397320

/-- Proves that a monkey slips back 2 feet per hour when climbing a 17 ft tree in 15 hours, 
    climbing 3 ft and slipping back a constant distance each hour. -/
theorem monkey_climb (tree_height : ℝ) (total_hours : ℕ) (climb_rate : ℝ) (slip_back : ℝ) : 
  tree_height = 17 →
  total_hours = 15 →
  climb_rate = 3 →
  (total_hours - 1 : ℝ) * (climb_rate - slip_back) + climb_rate = tree_height →
  slip_back = 2 := by
  sorry

#check monkey_climb

end monkey_climb_l3973_397320


namespace steve_socks_l3973_397333

theorem steve_socks (total_socks : ℕ) (matching_pairs : ℕ) (mismatching_socks : ℕ) : 
  total_socks = 48 → matching_pairs = 11 → mismatching_socks = total_socks - 2 * matching_pairs → mismatching_socks = 26 := by
  sorry

end steve_socks_l3973_397333


namespace z_in_third_quadrant_implies_a_range_l3973_397358

-- Define the complex number z
def z (a : ℝ) : ℂ := (2 + a * Complex.I) * (a - Complex.I)

-- Define the condition for z to be in the third quadrant
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- State the theorem
theorem z_in_third_quadrant_implies_a_range (a : ℝ) :
  in_third_quadrant (z a) → -Real.sqrt 2 < a ∧ a < 0 := by sorry

end z_in_third_quadrant_implies_a_range_l3973_397358


namespace four_number_puzzle_l3973_397300

theorem four_number_puzzle :
  ∀ (a b c d : ℕ),
    a + b + c + d = 243 →
    ∃ (x : ℚ),
      (a + 8 : ℚ) = x ∧
      (b - 8 : ℚ) = x ∧
      (c * 8 : ℚ) = x ∧
      (d / 8 : ℚ) = x →
    (max (max a b) (max c d)) * (min (min a b) (min c d)) = 576 := by
  sorry

end four_number_puzzle_l3973_397300


namespace double_base_exponent_l3973_397313

theorem double_base_exponent (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (2 * a)^(2 * b) = a^(2 * b) * y^(2 * b) → y = 2 := by
  sorry

end double_base_exponent_l3973_397313


namespace complete_square_equivalence_l3973_397384

theorem complete_square_equivalence :
  ∀ x : ℝ, 3 * x^2 - 6 * x + 2 = 0 ↔ (x - 1)^2 = 1/3 :=
by sorry

end complete_square_equivalence_l3973_397384


namespace sample_average_l3973_397355

theorem sample_average (x : ℝ) : 
  (1 + 3 + 2 + 5 + x) / 5 = 3 → x = 4 := by
  sorry

end sample_average_l3973_397355


namespace function_bound_implies_parameter_range_l3973_397394

-- Define the function f
def f (a d x : ℝ) : ℝ := a * x^3 + x^2 + x + d

-- State the theorem
theorem function_bound_implies_parameter_range :
  ∀ (a d : ℝ),
  (∀ x : ℝ, |x| ≤ 1 → |f a d x| ≤ 1) →
  (a ∈ Set.Icc (-2) 0 ∧ d ∈ Set.Icc (-2) 0) :=
by sorry

end function_bound_implies_parameter_range_l3973_397394


namespace smallest_five_digit_divisible_by_smallest_primes_l3973_397325

/-- The five smallest prime numbers -/
def smallest_primes : List Nat := [2, 3, 5, 7, 11]

/-- A number is five-digit if it's between 10000 and 99999 inclusive -/
def is_five_digit (n : Nat) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem smallest_five_digit_divisible_by_smallest_primes :
  ∃ (n : Nat), is_five_digit n ∧ 
    (∀ p ∈ smallest_primes, n % p = 0) ∧
    (∀ m : Nat, is_five_digit m ∧ (∀ p ∈ smallest_primes, m % p = 0) → n ≤ m) ∧
    n = 11550 := by
  sorry

end smallest_five_digit_divisible_by_smallest_primes_l3973_397325


namespace road_trip_cost_sharing_l3973_397302

/-- A road trip cost-sharing scenario -/
theorem road_trip_cost_sharing
  (alice_paid bob_paid carlos_paid : ℤ)
  (h_alice : alice_paid = 90)
  (h_bob : bob_paid = 150)
  (h_carlos : carlos_paid = 210)
  (h_split_evenly : alice_paid + bob_paid + carlos_paid = 3 * ((alice_paid + bob_paid + carlos_paid) / 3)) :
  let total := alice_paid + bob_paid + carlos_paid
  let share := total / 3
  let alice_owes := share - alice_paid
  let bob_owes := share - bob_paid
  alice_owes - bob_owes = 60 := by
sorry

end road_trip_cost_sharing_l3973_397302


namespace number_of_rats_l3973_397332

/-- Given a total of 70 animals where the number of rats is 6 times the number of chihuahuas,
    prove that the number of rats is 60. -/
theorem number_of_rats (total : ℕ) (chihuahuas : ℕ) (rats : ℕ) 
    (h1 : total = 70)
    (h2 : total = chihuahuas + rats)
    (h3 : rats = 6 * chihuahuas) : 
  rats = 60 := by
  sorry

end number_of_rats_l3973_397332


namespace sara_gave_dan_28_pears_l3973_397343

/-- The number of pears Sara initially picked -/
def initial_pears : ℕ := 35

/-- The number of pears Sara has left -/
def remaining_pears : ℕ := 7

/-- The number of pears Sara gave to Dan -/
def pears_given_to_dan : ℕ := initial_pears - remaining_pears

theorem sara_gave_dan_28_pears : pears_given_to_dan = 28 := by
  sorry

end sara_gave_dan_28_pears_l3973_397343


namespace extreme_values_imply_a_b_values_inequality_implies_m_range_l3973_397352

noncomputable def f (a b x : ℝ) : ℝ := 2 * a * x - b / x + Real.log x

def g (m x : ℝ) : ℝ := x^2 - 2 * m * x + m

def has_extreme_values (f : ℝ → ℝ) (x₁ x₂ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ (Set.Ioo (x₁ - ε) (x₁ + ε) ∪ Set.Ioo (x₂ - ε) (x₂ + ε)),
    f x ≤ f x₁ ∧ f x ≤ f x₂

theorem extreme_values_imply_a_b_values (a b : ℝ) :
  has_extreme_values (f a b) 1 (1/2) → a = -1/3 ∧ b = -1/3 :=
sorry

theorem inequality_implies_m_range (a b m : ℝ) :
  (a = -1/3 ∧ b = -1/3) →
  (∀ x₁ ∈ Set.Icc (1/2) 2, ∃ x₂ ∈ Set.Icc (1/2) 2, g m x₁ ≥ f a b x₂ - Real.log x₂) →
  m ≤ (3 + Real.sqrt 51) / 6 :=
sorry

end extreme_values_imply_a_b_values_inequality_implies_m_range_l3973_397352


namespace base_conversion_1729_to_base_5_l3973_397336

theorem base_conversion_1729_to_base_5 :
  ∃ (a b c d e : ℕ),
    1729 = a * 5^4 + b * 5^3 + c * 5^2 + d * 5^1 + e * 5^0 ∧
    a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 0 ∧ e = 4 :=
by sorry

end base_conversion_1729_to_base_5_l3973_397336


namespace school_distance_proof_l3973_397398

/-- The time in hours it takes to drive to school during rush hour -/
def rush_hour_time : ℚ := 18 / 60

/-- The time in hours it takes to drive to school with no traffic -/
def no_traffic_time : ℚ := 12 / 60

/-- The speed increase in mph when there's no traffic -/
def speed_increase : ℚ := 20

/-- The distance to school in miles -/
def distance_to_school : ℚ := 12

theorem school_distance_proof :
  ∃ (rush_hour_speed : ℚ),
    rush_hour_speed * rush_hour_time = distance_to_school ∧
    (rush_hour_speed + speed_increase) * no_traffic_time = distance_to_school := by
  sorry

#check school_distance_proof

end school_distance_proof_l3973_397398


namespace seongmin_completion_time_l3973_397390

/-- The number of days it takes Seongmin to complete the task alone -/
def seongmin_days : ℚ := 32

/-- The fraction of work Jinwoo and Seongmin complete together in 8 days -/
def work_together : ℚ := 7/12

/-- The number of days Jinwoo and Seongmin work together -/
def days_together : ℚ := 8

/-- The number of days Jinwoo works alone to complete the remaining work -/
def jinwoo_alone_days : ℚ := 10

theorem seongmin_completion_time :
  let total_work : ℚ := 1
  let work_rate_together : ℚ := work_together / days_together
  let jinwoo_alone_work : ℚ := total_work - work_together
  let jinwoo_work_rate : ℚ := jinwoo_alone_work / jinwoo_alone_days
  let seongmin_work_rate : ℚ := work_rate_together - jinwoo_work_rate
  seongmin_days = total_work / seongmin_work_rate :=
by sorry

end seongmin_completion_time_l3973_397390


namespace coin_value_theorem_l3973_397337

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Calculates the total value of coins in cents given the number of quarters and dimes -/
def total_value (quarters dimes : ℕ) : ℕ := quarter_value * quarters + dime_value * dimes

/-- Calculates the total value of coins in cents if quarters and dimes were swapped -/
def swapped_value (quarters dimes : ℕ) : ℕ := dime_value * quarters + quarter_value * dimes

theorem coin_value_theorem (quarters dimes : ℕ) :
  quarters + dimes = 30 →
  swapped_value quarters dimes = total_value quarters dimes + 150 →
  total_value quarters dimes = 450 := by
  sorry

end coin_value_theorem_l3973_397337


namespace annette_caitlin_weight_l3973_397327

/-- The combined weight of Annette and Caitlin given the conditions -/
theorem annette_caitlin_weight :
  ∀ (annette caitlin sara : ℝ),
  caitlin + sara = 87 →
  annette = sara + 8 →
  annette + caitlin = 95 :=
by
  sorry

end annette_caitlin_weight_l3973_397327


namespace complement_intersection_equals_l3973_397307

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection_equals : (U \ (A ∩ B)) = {1, 2, 4, 5} := by sorry

end complement_intersection_equals_l3973_397307


namespace dot_product_in_triangle_l3973_397301

/-- Given a triangle ABC where AB = (2, 3) and AC = (3, 4), prove that the dot product of AB and BC is 5. -/
theorem dot_product_in_triangle (A B C : ℝ × ℝ) : 
  B - A = (2, 3) → C - A = (3, 4) → (B - A) • (C - B) = 5 := by sorry

end dot_product_in_triangle_l3973_397301


namespace assignment_schemes_l3973_397322

/-- Represents the number of teachers and schools -/
def n : ℕ := 4

/-- The total number of assignment schemes -/
def total_schemes : ℕ := n^n

/-- The number of schemes where exactly one school is not assigned any teachers -/
def one_school_empty : ℕ := n * (n - 1)^(n - 1)

/-- The number of schemes where a certain school is assigned 2 teachers -/
def two_teachers_one_school : ℕ := Nat.choose n 2 * (n - 1)^(n - 2)

/-- The number of schemes where exactly two schools are not assigned any teachers -/
def two_schools_empty : ℕ := Nat.choose n 2 * (Nat.choose n 2 / 2 + n) * 2

theorem assignment_schemes :
  total_schemes = 256 ∧
  one_school_empty = 144 ∧
  two_teachers_one_school = 54 ∧
  two_schools_empty = 84 := by
  sorry

end assignment_schemes_l3973_397322


namespace triangle_side_length_l3973_397338

theorem triangle_side_length 
  (A B C : Real) 
  (AB BC AC : Real) :
  A = π / 3 →
  Real.tan B = 1 / 2 →
  AB = 2 * Real.sqrt 3 + 1 →
  BC = Real.sqrt 15 :=
by sorry

end triangle_side_length_l3973_397338


namespace liz_jump_shots_liz_jump_shots_correct_l3973_397370

theorem liz_jump_shots (initial_deficit : ℕ) (free_throws : ℕ) (three_pointers : ℕ) 
  (opponent_points : ℕ) (final_deficit : ℕ) : ℕ :=
  let free_throw_points := free_throws * 1
  let three_pointer_points := three_pointers * 3
  let total_deficit := initial_deficit + opponent_points
  let points_needed := total_deficit - final_deficit
  let jump_shot_points := points_needed - free_throw_points - three_pointer_points
  jump_shot_points / 2

theorem liz_jump_shots_correct :
  liz_jump_shots 20 5 3 10 8 = 4 := by sorry

end liz_jump_shots_liz_jump_shots_correct_l3973_397370


namespace rationalize_denominator_l3973_397391

theorem rationalize_denominator : 45 / Real.sqrt 45 = 3 * Real.sqrt 5 := by
  sorry

end rationalize_denominator_l3973_397391


namespace g_of_5_l3973_397348

/-- Given a function g : ℝ → ℝ satisfying g(x) + 3g(2 - x) = 4x^2 - 5x + 1 for all x ∈ ℝ,
    prove that g(5) = -5/4 -/
theorem g_of_5 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 5 * x + 1) :
  g 5 = -5/4 := by
  sorry

end g_of_5_l3973_397348


namespace quadratic_root_implies_m_value_l3973_397376

/-- Given a quadratic equation mx^2 + x - m^2 + 1 = 0 with -1 as a root, m must equal 1 -/
theorem quadratic_root_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, m*x^2 + x - m^2 + 1 = 0 → x = -1) → m = 1 :=
by sorry

end quadratic_root_implies_m_value_l3973_397376


namespace problem_1_l3973_397372

theorem problem_1 (a : ℚ) (h : a = 4/5) :
  -24.7 * a + 1.3 * a - (33/5) * a = -24 := by
  sorry

end problem_1_l3973_397372


namespace total_amount_theorem_l3973_397303

def calculate_selling_price (purchase_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  purchase_price * (1 - loss_percentage / 100)

def total_amount_received (price1 price2 price3 : ℚ) (loss1 loss2 loss3 : ℚ) : ℚ :=
  calculate_selling_price price1 loss1 +
  calculate_selling_price price2 loss2 +
  calculate_selling_price price3 loss3

theorem total_amount_theorem (price1 price2 price3 loss1 loss2 loss3 : ℚ) :
  price1 = 600 ∧ price2 = 800 ∧ price3 = 1000 ∧
  loss1 = 20 ∧ loss2 = 25 ∧ loss3 = 30 →
  total_amount_received price1 price2 price3 loss1 loss2 loss3 = 1780 :=
by sorry

end total_amount_theorem_l3973_397303


namespace smallest_circle_radius_l3973_397381

/-- A regular hexagon with side length 2 units -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- A circle in the context of our problem -/
structure Circle :=
  (center : Fin 6)  -- Vertex of the hexagon (0 to 5)
  (radius : ℝ)

/-- Three circles touching each other externally -/
def touching_circles (h : RegularHexagon) (c₁ c₂ c₃ : Circle) : Prop :=
  (c₁.center = 0 ∧ c₂.center = 1 ∧ c₃.center = 2) ∧  -- Centers at A, B, C
  (c₁.radius + c₂.radius = h.side_length) ∧
  (c₁.radius + c₃.radius = h.side_length * Real.sqrt 3) ∧
  (c₂.radius + c₃.radius = h.side_length)

theorem smallest_circle_radius 
  (h : RegularHexagon) 
  (c₁ c₂ c₃ : Circle) 
  (touch : touching_circles h c₁ c₂ c₃) :
  min c₁.radius (min c₂.radius c₃.radius) = 2 - Real.sqrt 3 :=
sorry

end smallest_circle_radius_l3973_397381


namespace f_max_min_on_interval_l3973_397399

-- Define the function f(x)
def f (x : ℝ) := x^3 - 12*x

-- Define the interval
def interval : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 16 ∧ min = -16 := by
  sorry

end f_max_min_on_interval_l3973_397399


namespace antons_number_l3973_397377

/-- Checks if two numbers match in exactly one digit place -/
def matchesOneDigit (a b : Nat) : Prop :=
  (a % 10 = b % 10 ∧ a / 10 ≠ b / 10) ∨
  (a / 10 % 10 = b / 10 % 10 ∧ a % 10 ≠ b % 10 ∧ a / 100 ≠ b / 100) ∨
  (a / 100 = b / 100 ∧ a % 100 ≠ b % 100)

theorem antons_number (x : Nat) :
  x ≥ 100 ∧ x < 1000 ∧
  matchesOneDigit x 109 ∧
  matchesOneDigit x 704 ∧
  matchesOneDigit x 124 →
  x = 729 := by
sorry

end antons_number_l3973_397377

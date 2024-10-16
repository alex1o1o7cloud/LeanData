import Mathlib

namespace NUMINAMATH_CALUDE_maria_yearly_distance_l505_50581

/-- Represents a pedometer with a maximum step count before resetting --/
structure Pedometer where
  max_steps : ℕ
  current_steps : ℕ

/-- Represents a postman's walking data for a year --/
structure PostmanYearlyData where
  pedometer : Pedometer
  flips : ℕ
  final_reading : ℕ
  steps_per_mile : ℕ

/-- Calculate the total distance walked by a postman in a year --/
def calculate_yearly_distance (data : PostmanYearlyData) : ℕ :=
  sorry

/-- Maria's yearly walking data --/
def maria_data : PostmanYearlyData :=
  { pedometer := { max_steps := 99999, current_steps := 0 },
    flips := 50,
    final_reading := 25000,
    steps_per_mile := 1500 }

theorem maria_yearly_distance :
  calculate_yearly_distance maria_data = 3350 :=
sorry

end NUMINAMATH_CALUDE_maria_yearly_distance_l505_50581


namespace NUMINAMATH_CALUDE_sum_of_roots_for_f_l505_50598

def f (x : ℝ) : ℝ := (3*x)^2 + 2*(3*x) + 1

theorem sum_of_roots_for_f (z : ℝ) : 
  (∃ z₁ z₂, f z₁ = 13 ∧ f z₂ = 13 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = -2/9) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_for_f_l505_50598


namespace NUMINAMATH_CALUDE_parallelogram_sides_l505_50508

theorem parallelogram_sides (x y : ℝ) : 
  (2*x + 3 = 9) ∧ (8*y - 1 = 7) → x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_sides_l505_50508


namespace NUMINAMATH_CALUDE_inequality_proof_l505_50531

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l505_50531


namespace NUMINAMATH_CALUDE_rare_coin_collection_l505_50584

theorem rare_coin_collection (initial_gold : ℕ) (initial_silver : ℕ) : 
  initial_gold = initial_silver / 3 →
  (initial_gold + 15 : ℕ) = initial_silver / 2 →
  initial_gold + initial_silver + 15 = 135 :=
by sorry

end NUMINAMATH_CALUDE_rare_coin_collection_l505_50584


namespace NUMINAMATH_CALUDE_victors_friend_bought_two_decks_l505_50570

/-- The number of decks Victor's friend bought given the conditions of the problem -/
def victors_friend_decks (deck_cost : ℕ) (victors_decks : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent - deck_cost * victors_decks) / deck_cost

/-- Theorem stating that Victor's friend bought 2 decks -/
theorem victors_friend_bought_two_decks :
  victors_friend_decks 8 6 64 = 2 := by
  sorry

end NUMINAMATH_CALUDE_victors_friend_bought_two_decks_l505_50570


namespace NUMINAMATH_CALUDE_smallest_result_is_24_l505_50517

def S : Finset ℕ := {2, 3, 5, 7, 11, 13}

def isConsecutive (a b : ℕ) : Prop := b = a + 1 ∨ a = b + 1

def validTriple (a b c : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ¬isConsecutive a b ∧ ¬isConsecutive b c ∧ ¬isConsecutive a c

def process (a b c : ℕ) : Finset ℕ :=
  {a * (b + c), b * (a + c), c * (a + b)}

theorem smallest_result_is_24 :
  ∀ a b c : ℕ, validTriple a b c →
    ∃ x ∈ process a b c, x ≥ 24 ∧ ∀ y ∈ process a b c, y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_result_is_24_l505_50517


namespace NUMINAMATH_CALUDE_apple_stack_count_l505_50550

def pyramid_stack (base_length : ℕ) (base_width : ℕ) : ℕ :=
  let layers := List.range (base_length - 1)
  let regular_layers := layers.map (λ i => (base_length - i) * (base_width - i))
  let top_layer := 2
  regular_layers.sum + top_layer

theorem apple_stack_count : pyramid_stack 6 9 = 156 := by
  sorry

end NUMINAMATH_CALUDE_apple_stack_count_l505_50550


namespace NUMINAMATH_CALUDE_triangle_max_area_l505_50536

/-- Given a triangle ABC with the following properties:
    1. (cos A / sin B) + (cos B / sin A) = 2
    2. The perimeter of the triangle is 12
    The maximum possible area of the triangle is 36(3 - 2√2) -/
theorem triangle_max_area (A B C : ℝ) (h1 : (Real.cos A / Real.sin B) + (Real.cos B / Real.sin A) = 2)
  (h2 : A + B + C = π) (h3 : Real.sin A > 0) (h4 : Real.sin B > 0) (h5 : Real.sin C > 0)
  (a b c : ℝ) (h6 : a + b + c = 12) (h7 : a > 0) (h8 : b > 0) (h9 : c > 0)
  (h10 : a / Real.sin A = b / Real.sin B) (h11 : b / Real.sin B = c / Real.sin C) :
  (1/2) * a * b * Real.sin C ≤ 36 * (3 - 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l505_50536


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l505_50562

theorem adult_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) : 
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_receipts ∧ 
    adult_tickets = 40 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l505_50562


namespace NUMINAMATH_CALUDE_solution_set_l505_50515

theorem solution_set (x y z : ℝ) : 
  x^2 = y^2 + z^2 ∧ 
  x^2024 = y^2024 + z^2024 ∧ 
  x^2025 = y^2025 + z^2025 →
  ((y = x ∧ z = 0) ∨ (y = -x ∧ z = 0) ∨ (y = 0 ∧ z = x) ∨ (y = 0 ∧ z = -x)) := by
sorry

end NUMINAMATH_CALUDE_solution_set_l505_50515


namespace NUMINAMATH_CALUDE_sum_of_real_solutions_l505_50588

theorem sum_of_real_solutions (a b : ℝ) (ha : a > 1) (hb : b > 0) :
  ∃ x : ℝ, Real.sqrt (a - Real.sqrt (a + b + x)) = x ∧
  x = Real.sqrt ((2 * a + 1 + Real.sqrt (4 * a + 1 + 4 * b)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_real_solutions_l505_50588


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l505_50509

/-- A geometric sequence with positive first term and a_2 * a_4 = 25 has a_3 = 5 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h1 : a 1 > 0) 
  (h2 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) 
  (h3 : a 2 * a 4 = 25) : a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l505_50509


namespace NUMINAMATH_CALUDE_right_triangle_from_medians_l505_50591

theorem right_triangle_from_medians (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    x^2 = (16 * b^2 - 4 * a^2) / 15 ∧
    y^2 = (16 * a^2 - 4 * b^2) / 15 ∧
    x^2 + y^2 = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_from_medians_l505_50591


namespace NUMINAMATH_CALUDE_divisibility_of_all_ones_number_l505_50565

/-- A positive integer whose decimal representation contains only ones -/
def AllOnesNumber (n : ℕ+) : Prop :=
  ∃ k : ℕ+, n.val = (10^k.val - 1) / 9

theorem divisibility_of_all_ones_number (N : ℕ+) 
  (h_all_ones : AllOnesNumber N) 
  (h_div_7 : 7 ∣ N.val) : 
  13 ∣ N.val := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_all_ones_number_l505_50565


namespace NUMINAMATH_CALUDE_ellipse_properties_l505_50572

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A hyperbola formed from an ellipse -/
structure Hyperbola (e : Ellipse) where
  is_equilateral : Bool

/-- A triangle formed by the left focus, right focus, and two points on the ellipse -/
structure Triangle (e : Ellipse) where
  perimeter : ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ := sorry

/-- The maximum area of a triangle formed by the foci and two points on the ellipse -/
def max_triangle_area (e : Ellipse) (t : Triangle e) : ℝ := sorry

theorem ellipse_properties (e : Ellipse) (h : Hyperbola e) (t : Triangle e) :
  h.is_equilateral = true → t.perimeter = 8 →
    eccentricity e = Real.sqrt 2 / 2 ∧ max_triangle_area e t = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l505_50572


namespace NUMINAMATH_CALUDE_point_positions_l505_50522

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation x - y + m = 0 -/
def line_equation (p : Point) (m : ℝ) : ℝ := p.x - p.y + m

/-- Two points are on opposite sides of the line if the product of their line equations is negative -/
def opposite_sides (a b : Point) (m : ℝ) : Prop :=
  line_equation a m * line_equation b m < 0

theorem point_positions (m : ℝ) : 
  let a : Point := ⟨2, 1⟩
  let b : Point := ⟨1, 3⟩
  opposite_sides a b m ↔ -1 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_positions_l505_50522


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l505_50555

theorem opposite_of_negative_fraction :
  -(-(4/5 : ℚ)) = 4/5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l505_50555


namespace NUMINAMATH_CALUDE_even_decreasing_inequality_l505_50526

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_decreasing_on_positive (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f y < f x

-- State the theorem
theorem even_decreasing_inequality (h1 : is_even f) (h2 : is_decreasing_on_positive f) :
  f (1/2) > f (-2/3) ∧ f (-2/3) > f (3/4) := by sorry

end NUMINAMATH_CALUDE_even_decreasing_inequality_l505_50526


namespace NUMINAMATH_CALUDE_expression_evaluation_l505_50585

/-- Proves that the given expression evaluates to -3/2 when x = -1/2 and y = 3 -/
theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := 3
  3 * (2 * x^2 * y - x * y^2) - 2 * (-2 * y^2 * x + x^2 * y) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l505_50585


namespace NUMINAMATH_CALUDE_ratio_equality_l505_50506

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (2*x - z) = (x + y) / (2*z) ∧ (x + y) / (2*z) = x / y) :
  x / y = 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l505_50506


namespace NUMINAMATH_CALUDE_inequality_proof_l505_50527

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : -1 < b ∧ b < 0) :
  a * b < a * b^2 ∧ a * b^2 < a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l505_50527


namespace NUMINAMATH_CALUDE_integer_triangle_exists_l505_50520

/-- A triangle with integer side lengths forming an arithmetic progression and integer area -/
structure IntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  area : ℕ
  arith_prog : b - a = c - b
  area_formula : area^2 = (a + b + c) * (b + c - a) * (a + c - b) * (a + b - c) / 16

/-- The existence of a specific integer triangle with sides 3, 4, 5 -/
theorem integer_triangle_exists : ∃ (t : IntegerTriangle), t.a = 3 ∧ t.b = 4 ∧ t.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_integer_triangle_exists_l505_50520


namespace NUMINAMATH_CALUDE_perfect_number_examples_mn_value_S_is_perfect_number_min_value_a_plus_b_l505_50582

/-- Definition of a perfect number -/
def is_perfect_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

/-- Statement 1: 29 and 13 are perfect numbers, while 48 and 28 are not -/
theorem perfect_number_examples :
  is_perfect_number 29 ∧ is_perfect_number 13 ∧ ¬is_perfect_number 48 ∧ ¬is_perfect_number 28 :=
sorry

/-- Statement 2: Given a^2 - 4a + 8 = (a - m)^2 + n^2, prove that mn = ±4 -/
theorem mn_value (a m n : ℝ) (h : a^2 - 4*a + 8 = (a - m)^2 + n^2) :
  m * n = 4 ∨ m * n = -4 :=
sorry

/-- Statement 3: Given S = a^2 + 4ab + 5b^2 - 12b + k, prove that S is a perfect number when k = 36 -/
theorem S_is_perfect_number (a b : ℤ) :
  is_perfect_number (a^2 + 4*a*b + 5*b^2 - 12*b + 36) :=
sorry

/-- Statement 4: Given -a^2 + 5a + b - 7 = 0, prove that the minimum value of a + b is 3 -/
theorem min_value_a_plus_b (a b : ℝ) (h : -a^2 + 5*a + b - 7 = 0) :
  a + b ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_perfect_number_examples_mn_value_S_is_perfect_number_min_value_a_plus_b_l505_50582


namespace NUMINAMATH_CALUDE_hostel_provisions_theorem_l505_50544

/-- The number of days provisions last for the initial group -/
def initial_days : ℕ := 50

/-- The number of days provisions last with 20 fewer people -/
def extended_days : ℕ := 250

/-- The number of fewer people in the extended scenario -/
def fewer_people : ℕ := 20

/-- The function to calculate the daily consumption rate given the number of people and days -/
def daily_consumption_rate (people : ℕ) (days : ℕ) : ℚ :=
  1 / (people.cast * days.cast)

theorem hostel_provisions_theorem (initial_girls : ℕ) :
  (daily_consumption_rate initial_girls initial_days =
   daily_consumption_rate (initial_girls + fewer_people) extended_days) →
  initial_girls = 25 := by
  sorry

end NUMINAMATH_CALUDE_hostel_provisions_theorem_l505_50544


namespace NUMINAMATH_CALUDE_inequality_range_of_a_l505_50504

theorem inequality_range_of_a (b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |x^2 - a*x| + b < 0) ↔
  ((b ≥ -1 ∧ b < 0 ∧ a ∈ Set.Ioo (1 + b) (2 * Real.sqrt (-b))) ∨
   (b < -1 ∧ a ∈ Set.Ioo (1 + b) (1 - b))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_of_a_l505_50504


namespace NUMINAMATH_CALUDE_tetra_edge_is_2sqrt3_l505_50513

/-- Configuration of five mutually tangent spheres with a circumscribed tetrahedron -/
structure SphereTetConfig where
  /-- Radius of each sphere -/
  r : ℝ
  /-- Centers of the four bottom spheres -/
  bottom_centers : Fin 4 → ℝ × ℝ × ℝ
  /-- Center of the top sphere -/
  top_center : ℝ × ℝ × ℝ
  /-- Vertices of the tetrahedron -/
  tetra_vertices : Fin 4 → ℝ × ℝ × ℝ

/-- The spheres are mutually tangent and properly configured -/
def is_valid_config (cfg : SphereTetConfig) : Prop :=
  cfg.r = 2 ∧
  ∀ i j, i ≠ j → dist (cfg.bottom_centers i) (cfg.bottom_centers j) = 4 ∧
  ∀ i, dist (cfg.bottom_centers i) cfg.top_center = 4 ∧
  cfg.top_center.2 = 2 ∧
  cfg.tetra_vertices 0 = cfg.top_center ∧
  ∀ i : Fin 3, cfg.tetra_vertices (i + 1) = cfg.bottom_centers i

/-- The edge length of the tetrahedron -/
def tetra_edge_length (cfg : SphereTetConfig) : ℝ :=
  dist (cfg.tetra_vertices 0) (cfg.tetra_vertices 1)

/-- Main theorem: The edge length of the tetrahedron is 2√3 -/
theorem tetra_edge_is_2sqrt3 (cfg : SphereTetConfig) (h : is_valid_config cfg) :
  tetra_edge_length cfg = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tetra_edge_is_2sqrt3_l505_50513


namespace NUMINAMATH_CALUDE_liar_count_l505_50521

/-- Represents a district in the town -/
inductive District
| A
| B
| Γ
| Δ

/-- Structure representing the town -/
structure Town where
  knights : Nat
  liars : Nat
  affirmativeAnswers : District → Nat

/-- The conditions of the problem -/
def townConditions (t : Town) : Prop :=
  t.affirmativeAnswers District.A +
  t.affirmativeAnswers District.B +
  t.affirmativeAnswers District.Γ +
  t.affirmativeAnswers District.Δ = 500 ∧
  t.knights * 4 = 200 ∧
  t.affirmativeAnswers District.A = t.knights + 95 ∧
  t.affirmativeAnswers District.B = t.knights + 115 ∧
  t.affirmativeAnswers District.Γ = t.knights + 157 ∧
  t.affirmativeAnswers District.Δ = t.knights + 133 ∧
  t.liars * 3 + t.knights = 500

theorem liar_count (t : Town) (h : townConditions t) : t.liars = 100 := by
  sorry

end NUMINAMATH_CALUDE_liar_count_l505_50521


namespace NUMINAMATH_CALUDE_fourth_sample_is_nineteen_l505_50576

/-- Represents a systematic sampling scenario in a class. -/
structure SystematicSample where
  total_students : ℕ
  sample_size : ℕ
  known_samples : List ℕ

/-- Calculates the interval for systematic sampling. -/
def sampling_interval (s : SystematicSample) : ℕ :=
  s.total_students / s.sample_size

/-- Checks if a number is part of the systematic sample. -/
def is_in_sample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k, n = k * sampling_interval s + (s.known_samples.head? ).getD 0

/-- The main theorem stating that 19 must be the fourth sample in the given scenario. -/
theorem fourth_sample_is_nineteen (s : SystematicSample)
    (h1 : s.total_students = 56)
    (h2 : s.sample_size = 4)
    (h3 : s.known_samples = [5, 33, 47])
    (h4 : ∀ n, is_in_sample s n → n ∈ [5, 19, 33, 47]) :
    is_in_sample s 19 :=
  sorry

#check fourth_sample_is_nineteen

end NUMINAMATH_CALUDE_fourth_sample_is_nineteen_l505_50576


namespace NUMINAMATH_CALUDE_f_2019_l505_50561

/-- The function f(n) represents the original number of the last person to leave the line. -/
def f (n : ℕ) : ℕ :=
  let m := Nat.sqrt n
  if n ≤ m * m + m then m * m + 1
  else m * m + m + 1

/-- Theorem stating that f(2019) = 1981 -/
theorem f_2019 : f 2019 = 1981 := by
  sorry

end NUMINAMATH_CALUDE_f_2019_l505_50561


namespace NUMINAMATH_CALUDE_square_diagonals_equal_l505_50546

-- Define the necessary structures
structure Rectangle where
  diagonals_equal : Bool

structure Square extends Rectangle

-- Define the theorem
theorem square_diagonals_equal (h1 : ∀ r : Rectangle, r.diagonals_equal) 
  (h2 : Square → Rectangle) : 
  ∀ s : Square, (h2 s).diagonals_equal :=
by
  sorry


end NUMINAMATH_CALUDE_square_diagonals_equal_l505_50546


namespace NUMINAMATH_CALUDE_place_value_ratios_l505_50548

theorem place_value_ratios : 
  ∀ (d : ℕ), d > 0 → d < 10 →
  (d * 10000) / (d * 1000) = 10 ∧ 
  (d * 100000) / (d * 100) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratios_l505_50548


namespace NUMINAMATH_CALUDE_range_of_m_l505_50577

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 1

def set_A : Set ℝ := {a | -1 < a ∧ a < 1}

def set_B (m : ℝ) : Set ℝ := {a | m < a ∧ a < m + 3}

theorem range_of_m (m : ℝ) :
  (∀ x, x ∈ set_A → x ∈ set_B m) ∧
  (∃ x, x ∈ set_B m ∧ x ∉ set_A) →
  -2 ≤ m ∧ m ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l505_50577


namespace NUMINAMATH_CALUDE_negation_of_proposition_l505_50587

variable (a : ℝ)

theorem negation_of_proposition (p : Prop) :
  (¬ (∃ x : ℝ, x^2 + 2*a*x + a ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2*a*x + a > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l505_50587


namespace NUMINAMATH_CALUDE_erics_chickens_l505_50539

/-- The number of chickens on Eric's farm -/
def num_chickens : ℕ := 4

/-- The number of eggs each chicken lays per day -/
def eggs_per_chicken_per_day : ℕ := 3

/-- The number of days Eric collected eggs -/
def days_collected : ℕ := 3

/-- The total number of eggs collected -/
def total_eggs_collected : ℕ := 36

theorem erics_chickens :
  num_chickens * eggs_per_chicken_per_day * days_collected = total_eggs_collected :=
by sorry

end NUMINAMATH_CALUDE_erics_chickens_l505_50539


namespace NUMINAMATH_CALUDE_yunhwan_water_consumption_l505_50594

/-- Yunhwan's yearly water consumption in liters -/
def yearly_water_consumption (monthly_consumption : ℝ) (months_per_year : ℕ) : ℝ :=
  monthly_consumption * months_per_year

/-- Proof that Yunhwan's yearly water consumption is 2194.56 liters -/
theorem yunhwan_water_consumption : 
  yearly_water_consumption 182.88 12 = 2194.56 := by
  sorry

end NUMINAMATH_CALUDE_yunhwan_water_consumption_l505_50594


namespace NUMINAMATH_CALUDE_factorial_sum_mod_20_l505_50528

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_sum_mod_20 : (factorial 1 + factorial 2 + factorial 3 + factorial 4 + factorial 5 + factorial 6) % 20 = 13 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_mod_20_l505_50528


namespace NUMINAMATH_CALUDE_vector_norm_difference_l505_50511

theorem vector_norm_difference (a b : ℝ × ℝ) 
  (h1 : a + b = (2, 3)) 
  (h2 : a - b = (-2, 1)) : 
  ‖a‖^2 - ‖b‖^2 = -1 := by sorry

end NUMINAMATH_CALUDE_vector_norm_difference_l505_50511


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l505_50533

theorem imaginary_part_of_z_is_zero : 
  ∀ z : ℂ, z * (Complex.I + 1) = 2 / (Complex.I - 1) → Complex.im z = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l505_50533


namespace NUMINAMATH_CALUDE_max_cables_cut_theorem_l505_50557

/-- Represents a computer network -/
structure ComputerNetwork where
  num_computers : ℕ
  num_cables : ℕ
  num_clusters : ℕ

/-- Calculates the maximum number of cables that can be cut -/
def max_cables_cut (network : ComputerNetwork) : ℕ :=
  network.num_cables - (network.num_clusters - 1)

/-- Theorem stating the maximum number of cables that can be cut -/
theorem max_cables_cut_theorem (network : ComputerNetwork) 
  (h1 : network.num_computers = 200)
  (h2 : network.num_cables = 345)
  (h3 : network.num_clusters = 8) :
  max_cables_cut network = 153 := by
  sorry

#eval max_cables_cut ⟨200, 345, 8⟩

end NUMINAMATH_CALUDE_max_cables_cut_theorem_l505_50557


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l505_50578

theorem sum_of_x_and_y_equals_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 20) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l505_50578


namespace NUMINAMATH_CALUDE_twentieth_number_in_twentieth_row_l505_50549

/-- Calculates the first number in a given row of the triangular sequence -/
def first_number_in_row (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the kth number in the nth row of the triangular sequence -/
def number_in_sequence (n k : ℕ) : ℕ := first_number_in_row n + (k - 1)

/-- The 20th number in the 20th row of the triangular sequence is 381 -/
theorem twentieth_number_in_twentieth_row :
  number_in_sequence 20 20 = 381 := by sorry

end NUMINAMATH_CALUDE_twentieth_number_in_twentieth_row_l505_50549


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l505_50552

theorem unique_number_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    x = p * q * r ∧ 
    (∀ s : ℕ, Nat.Prime s ∧ s ∣ x → s = p ∨ s = q ∨ s = r)) →
  7 ∣ x →
  x = 728 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_l505_50552


namespace NUMINAMATH_CALUDE_tetrahedral_die_expected_steps_l505_50571

def expected_steps (n : Nat) : ℚ :=
  match n with
  | 1 => 1
  | 2 => 5/4
  | 3 => 25/16
  | 4 => 125/64
  | _ => 0

theorem tetrahedral_die_expected_steps :
  let total_expectation := 1 + (expected_steps 1 + expected_steps 2 + expected_steps 3 + expected_steps 4) / 4
  total_expectation = 625/256 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedral_die_expected_steps_l505_50571


namespace NUMINAMATH_CALUDE_square_area_error_percentage_l505_50583

/-- If the measured side length of a square is 102.5% of its actual side length,
    then the percentage of error in the calculated area of the square is 5.0625%. -/
theorem square_area_error_percentage (a : ℝ) (h : a > 0) :
  let measured_side := 1.025 * a
  let actual_area := a ^ 2
  let calculated_area := measured_side ^ 2
  (calculated_area - actual_area) / actual_area * 100 = 5.0625 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_percentage_l505_50583


namespace NUMINAMATH_CALUDE_equality_from_fraction_equation_l505_50558

theorem equality_from_fraction_equation (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  1 / (3 * a) + 2 / (3 * b) = 3 / (a + 2 * b) → a = b := by
  sorry

end NUMINAMATH_CALUDE_equality_from_fraction_equation_l505_50558


namespace NUMINAMATH_CALUDE_fraction_of_loss_example_l505_50502

/-- Calculates the fraction of loss given the cost price and selling price -/
def fractionOfLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice

/-- Theorem: The fraction of loss for an item with cost price 18 and selling price 17 is 1/18 -/
theorem fraction_of_loss_example : fractionOfLoss 18 17 = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_loss_example_l505_50502


namespace NUMINAMATH_CALUDE_unit_vector_xy_plane_l505_50569

theorem unit_vector_xy_plane (u : ℝ × ℝ × ℝ) : 
  let (x, y, z) := u
  (x^2 + y^2 = 1 ∧ z = 0) →  -- u is a unit vector in the xy-plane
  (x + 3*y = Real.sqrt 30 / 2) →  -- angle with (1, 3, 0) is 30°
  (3*x - y = Real.sqrt 20 / 2) →  -- angle with (3, -1, 0) is 45°
  x = (3 * Real.sqrt 20 + Real.sqrt 30) / 20 :=
by sorry

end NUMINAMATH_CALUDE_unit_vector_xy_plane_l505_50569


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l505_50532

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬(is_divisible (y + 3) 18 ∧ is_divisible (y + 3) 70 ∧ 
                     is_divisible (y + 3) 25 ∧ is_divisible (y + 3) 21)) ∧
  (is_divisible (x + 3) 18 ∧ is_divisible (x + 3) 70 ∧ 
   is_divisible (x + 3) 25 ∧ is_divisible (x + 3) 21) ∧
  x = 3147 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l505_50532


namespace NUMINAMATH_CALUDE_interchange_difference_for_62_l505_50505

def is_two_digit_number (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def interchange_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem interchange_difference_for_62 :
  is_two_digit_number 62 ∧ digit_sum 62 = 8 →
  62 - interchange_digits 62 = 36 := by
  sorry

end NUMINAMATH_CALUDE_interchange_difference_for_62_l505_50505


namespace NUMINAMATH_CALUDE_ABD_collinear_l505_50554

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (a b : V)

-- Define the points
variable (A B C D : V)

-- Define the vector relationships
axiom AB_def : B - A = a + 2 • b
axiom BC_def : C - B = -5 • a + 6 • b
axiom CD_def : D - C = 7 • a - 2 • b

-- Theorem to prove
theorem ABD_collinear : ∃ (t : ℝ), D - A = t • (B - A) := by
  sorry

end NUMINAMATH_CALUDE_ABD_collinear_l505_50554


namespace NUMINAMATH_CALUDE_money_left_proof_l505_50501

def salary : ℚ := 150000.00000000003

def food_fraction : ℚ := 1 / 5
def rent_fraction : ℚ := 1 / 10
def clothes_fraction : ℚ := 3 / 5

def money_left : ℚ := salary - (salary * food_fraction + salary * rent_fraction + salary * clothes_fraction)

theorem money_left_proof : money_left = 15000 := by
  sorry

end NUMINAMATH_CALUDE_money_left_proof_l505_50501


namespace NUMINAMATH_CALUDE_positive_intervals_l505_50543

-- Define the expression
def f (x : ℝ) : ℝ := (x + 2) * (x - 2)

-- State the theorem
theorem positive_intervals (x : ℝ) : f x > 0 ↔ x < -2 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_intervals_l505_50543


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l505_50538

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l505_50538


namespace NUMINAMATH_CALUDE_square_difference_equals_736_l505_50574

theorem square_difference_equals_736 : (23 + 16)^2 - (23^2 + 16^2) = 736 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_736_l505_50574


namespace NUMINAMATH_CALUDE_largest_square_area_l505_50516

theorem largest_square_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2)
  (h_sum_areas : a^2 + b^2 + c^2 = 450) : c^2 = 225 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l505_50516


namespace NUMINAMATH_CALUDE_tan_value_from_ratio_l505_50556

theorem tan_value_from_ratio (α : Real) :
  (Real.sin α + 7 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = -5 →
  Real.tan α = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_ratio_l505_50556


namespace NUMINAMATH_CALUDE_unique_a_value_l505_50524

-- Define the set A
def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

-- State the theorem
theorem unique_a_value : ∃! a : ℝ, 1 ∈ A a ∧ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l505_50524


namespace NUMINAMATH_CALUDE_cost_of_paints_paint_cost_is_five_l505_50512

theorem cost_of_paints (classes : ℕ) (folders_per_class : ℕ) (pencils_per_class : ℕ) 
  (pencils_per_eraser : ℕ) (folder_cost : ℕ) (pencil_cost : ℕ) (eraser_cost : ℕ) 
  (total_spent : ℕ) : ℕ :=
  let total_folders := classes * folders_per_class
  let total_pencils := classes * pencils_per_class
  let total_erasers := total_pencils / pencils_per_eraser
  let folder_expense := total_folders * folder_cost
  let pencil_expense := total_pencils * pencil_cost
  let eraser_expense := total_erasers * eraser_cost
  let total_expense := folder_expense + pencil_expense + eraser_expense
  total_spent - total_expense

theorem paint_cost_is_five :
  cost_of_paints 6 1 3 6 6 2 1 80 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_paints_paint_cost_is_five_l505_50512


namespace NUMINAMATH_CALUDE_min_value_of_f_min_value_attained_l505_50535

/-- The function f(x) = (x^2 + 2) / √(x^2 + 1) has a minimum value of 2 for all real x -/
theorem min_value_of_f (x : ℝ) : (x^2 + 2) / Real.sqrt (x^2 + 1) ≥ 2 := by
  sorry

/-- The minimum value 2 is attained when x = 0 -/
theorem min_value_attained : ∃ x : ℝ, (x^2 + 2) / Real.sqrt (x^2 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_min_value_attained_l505_50535


namespace NUMINAMATH_CALUDE_rhombus_area_l505_50514

/-- The area of a rhombus with diagonals of 14 cm and 20 cm is 140 square centimeters. -/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 140 := by
sorry

end NUMINAMATH_CALUDE_rhombus_area_l505_50514


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l505_50545

theorem gcd_from_lcm_and_ratio (A B : ℕ+) : 
  Nat.lcm A B = 200 → A * 5 = B * 2 → Nat.gcd A B = 20 := by
  sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l505_50545


namespace NUMINAMATH_CALUDE_defective_probability_l505_50519

/-- The probability that both selected products are defective given that one is defective -/
theorem defective_probability (total : ℕ) (genuine : ℕ) (defective : ℕ) (selected : ℕ) 
  (h1 : total = genuine + defective)
  (h2 : total = 10)
  (h3 : genuine = 6)
  (h4 : defective = 4)
  (h5 : selected = 2) :
  (defective.choose 2 : ℚ) / (total.choose 2) / 
  ((defective.choose 1 * genuine.choose 1 + defective.choose 2 : ℚ) / total.choose 2) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_defective_probability_l505_50519


namespace NUMINAMATH_CALUDE_books_written_proof_l505_50529

def total_books (zig_books flo_books : ℕ) : ℕ :=
  zig_books + flo_books

theorem books_written_proof (zig_books flo_books : ℕ) 
  (h1 : zig_books = 60) 
  (h2 : zig_books = 4 * flo_books) : 
  total_books zig_books flo_books = 75 := by
  sorry

end NUMINAMATH_CALUDE_books_written_proof_l505_50529


namespace NUMINAMATH_CALUDE_log_equation_solution_l505_50586

theorem log_equation_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l505_50586


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l505_50547

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_1, a_2, and a_4 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℝ) : Prop :=
  a 2 ^ 2 = a 1 * a 4

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
  (h1 : arithmetic_seq a) (h2 : geometric_subseq a) : a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l505_50547


namespace NUMINAMATH_CALUDE_mrs_hilt_friends_l505_50566

/-- Mrs. Hilt's friends problem -/
theorem mrs_hilt_friends (friends_can_go : ℕ) (friends_cant_go : ℕ) 
  (h1 : friends_can_go = 8) (h2 : friends_cant_go = 7) : 
  friends_can_go + friends_cant_go = 15 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_friends_l505_50566


namespace NUMINAMATH_CALUDE_system_solution_unique_l505_50500

theorem system_solution_unique : 
  ∃! (x y : ℝ), 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l505_50500


namespace NUMINAMATH_CALUDE_cutting_process_ends_at_1998_l505_50525

/-- Represents a shape with points on its boundary -/
structure Shape :=
  (points : ℕ)

/-- Represents the state of the cutting process -/
structure CuttingState :=
  (shape : Shape)
  (cuts : ℕ)

/-- Checks if a shape is a polygon -/
def is_polygon (s : Shape) : Prop :=
  s.points ≤ 3

/-- Checks if a shape can become a polygon with further cutting -/
def can_become_polygon (s : Shape) : Prop :=
  s.points > 3

/-- Performs a cut on the shape -/
def cut (state : CuttingState) : CuttingState :=
  { shape := { points := state.shape.points - 1 },
    cuts := state.cuts + 1 }

/-- The main theorem to be proved -/
theorem cutting_process_ends_at_1998 :
  ∀ (initial_state : CuttingState),
    initial_state.shape.points = 1001 →
    ∀ (n : ℕ),
      n ≤ 1998 →
      ¬(is_polygon (cut^[n] initial_state).shape) ∧
      can_become_polygon (cut^[n] initial_state).shape →
      ¬(∃ (m : ℕ),
        m > 1998 ∧
        ¬(is_polygon (cut^[m] initial_state).shape) ∧
        can_become_polygon (cut^[m] initial_state).shape) :=
sorry

end NUMINAMATH_CALUDE_cutting_process_ends_at_1998_l505_50525


namespace NUMINAMATH_CALUDE_cold_production_time_proof_l505_50563

/-- The time (in minutes) it takes to produce each pot when the machine is cold. -/
def cold_production_time : ℝ := 6

/-- The time (in minutes) it takes to produce each pot when the machine is warm. -/
def warm_production_time : ℝ := 5

/-- The number of additional pots produced in the last hour compared to the first. -/
def additional_pots : ℕ := 2

/-- The number of minutes in an hour. -/
def minutes_per_hour : ℕ := 60

theorem cold_production_time_proof :
  cold_production_time = 6 ∧
  warm_production_time = 5 ∧
  additional_pots = 2 ∧
  minutes_per_hour / cold_production_time + additional_pots = minutes_per_hour / warm_production_time :=
by sorry

end NUMINAMATH_CALUDE_cold_production_time_proof_l505_50563


namespace NUMINAMATH_CALUDE_ellipse_I_equation_ellipse_II_equation_l505_50523

-- Part I
def ellipse_I (x y : ℝ) := x^2 / 2 + y^2 = 1

theorem ellipse_I_equation : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), ellipse_I x y ↔ 
    (x + 1)^2 + y^2 + ((x - 1)^2 + y^2).sqrt = 2 * a ∧
    a^2 - 1 = b^2 ∧
    x^2 / a^2 + y^2 / b^2 = 1) ∧
  ellipse_I (1/2) (Real.sqrt 14 / 4) :=
sorry

-- Part II
def ellipse_II (x y : ℝ) := x^2 / 4 + y^2 / 2 = 1

theorem ellipse_II_equation :
  ellipse_II (Real.sqrt 2) (-1) ∧
  ellipse_II (-1) (Real.sqrt 6 / 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_I_equation_ellipse_II_equation_l505_50523


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l505_50597

/-- A rectangle with perimeter 60 feet and area 130 square feet has a shorter side of approximately 5 feet -/
theorem rectangle_shorter_side (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≥ b)
  (h_perimeter : 2*a + 2*b = 60) (h_area : a*b = 130) :
  ∃ ε > 0, abs (b - 5) < ε :=
sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l505_50597


namespace NUMINAMATH_CALUDE_perfect_square_equation_l505_50599

theorem perfect_square_equation (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_equation_l505_50599


namespace NUMINAMATH_CALUDE_fruit_problem_solution_l505_50537

/-- Represents the solution to the fruit buying problem -/
def FruitSolution : Type := ℕ × ℕ × ℕ

/-- The total number of fruits bought -/
def total_fruits : ℕ := 100

/-- The total cost in copper coins -/
def total_cost : ℕ := 100

/-- The cost of a single peach in copper coins -/
def peach_cost : ℕ := 3

/-- The cost of a single plum in copper coins -/
def plum_cost : ℕ := 4

/-- The number of olives that can be bought for 1 copper coin -/
def olives_per_coin : ℕ := 7

/-- Checks if a given solution satisfies all conditions of the problem -/
def is_valid_solution (solution : FruitSolution) : Prop :=
  let (peaches, plums, olives) := solution
  peaches + plums + olives = total_fruits ∧
  peach_cost * peaches + plum_cost * plums + (olives / olives_per_coin) = total_cost

/-- The correct solution to the problem -/
def correct_solution : FruitSolution := (3, 20, 77)

/-- Theorem stating that the correct_solution is the unique valid solution -/
theorem fruit_problem_solution :
  is_valid_solution correct_solution ∧
  ∀ (other : FruitSolution), is_valid_solution other → other = correct_solution :=
sorry

end NUMINAMATH_CALUDE_fruit_problem_solution_l505_50537


namespace NUMINAMATH_CALUDE_twelve_students_pairs_l505_50534

/-- The number of unique pairs that can be formed from a group of n elements -/
def number_of_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of unique pairs from a group of 12 students is 66 -/
theorem twelve_students_pairs : number_of_pairs 12 = 66 := by
  sorry

end NUMINAMATH_CALUDE_twelve_students_pairs_l505_50534


namespace NUMINAMATH_CALUDE_cosine_value_l505_50503

theorem cosine_value (α : Real) 
  (h : Real.sin (π / 6 - α) = 5 / 13) : 
  Real.cos (π / 3 + α) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_l505_50503


namespace NUMINAMATH_CALUDE_soda_bottle_difference_l505_50595

theorem soda_bottle_difference (regular_soda : ℕ) (diet_soda : ℕ)
  (h1 : regular_soda = 81)
  (h2 : diet_soda = 60) :
  regular_soda - diet_soda = 21 := by
  sorry

end NUMINAMATH_CALUDE_soda_bottle_difference_l505_50595


namespace NUMINAMATH_CALUDE_rachel_apple_picking_l505_50560

theorem rachel_apple_picking (num_trees : ℕ) (total_picked : ℕ) (h1 : num_trees = 4) (h2 : total_picked = 28) :
  total_picked / num_trees = 7 := by
  sorry

end NUMINAMATH_CALUDE_rachel_apple_picking_l505_50560


namespace NUMINAMATH_CALUDE_congruence_systems_solutions_l505_50507

theorem congruence_systems_solutions :
  (∃ x : ℤ, x % 7 = 3 ∧ (6 * x) % 8 = 10) ∧
  (∀ x : ℤ, x % 7 = 3 ∧ (6 * x) % 8 = 10 → x % 56 = 3 ∨ x % 56 = 31) ∧
  (¬ ∃ x : ℤ, (3 * x) % 10 = 1 ∧ (4 * x) % 15 = 7) :=
by sorry

end NUMINAMATH_CALUDE_congruence_systems_solutions_l505_50507


namespace NUMINAMATH_CALUDE_notebook_distribution_ratio_l505_50564

/-- Given a class where notebooks are distributed equally among children,
    prove that the ratio of notebooks per child to the number of children is 1/8 -/
theorem notebook_distribution_ratio 
  (C : ℕ) -- number of children
  (N : ℕ) -- number of notebooks per child
  (h1 : C * N = 512) -- total notebooks distributed
  (h2 : (C / 2) * 16 = 512) -- if children halved, each gets 16
  : N / C = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_ratio_l505_50564


namespace NUMINAMATH_CALUDE_remainder_x14_minus_1_div_x_plus_1_l505_50575

theorem remainder_x14_minus_1_div_x_plus_1 (x : ℝ) : 
  (x^14 - 1) % (x + 1) = 0 := by sorry

end NUMINAMATH_CALUDE_remainder_x14_minus_1_div_x_plus_1_l505_50575


namespace NUMINAMATH_CALUDE_f_nonpositive_implies_k_geq_one_l505_50542

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := log x - k * x + 1

theorem f_nonpositive_implies_k_geq_one (k : ℝ) :
  (∀ x > 0, f k x ≤ 0) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_nonpositive_implies_k_geq_one_l505_50542


namespace NUMINAMATH_CALUDE_smallest_b_for_integer_solutions_l505_50589

theorem smallest_b_for_integer_solutions : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∀ (x : ℤ), x^2 + b*x = -21 → ∃ (y : ℤ), x = y) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ∃ (x : ℝ), x^2 + b'*x = -21 ∧ ¬∃ (y : ℤ), x = y) ∧
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_b_for_integer_solutions_l505_50589


namespace NUMINAMATH_CALUDE_specific_hexagon_perimeter_l505_50596

/-- A hexagon with specific side lengths and right angles -/
structure RightAngleHexagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EF : ℝ
  FA : ℝ
  right_angles : Bool

/-- The perimeter of a hexagon -/
def perimeter (h : RightAngleHexagon) : ℝ :=
  h.AB + h.BC + h.CD + h.DE + h.EF + h.FA

/-- Theorem: The perimeter of the specific hexagon is 6 -/
theorem specific_hexagon_perimeter :
  ∃ (h : RightAngleHexagon),
    h.AB = 1 ∧ h.BC = 1 ∧ h.CD = 2 ∧ h.DE = 1 ∧ h.EF = 1 ∧ h.right_angles = true ∧
    perimeter h = 6 := by
  sorry

end NUMINAMATH_CALUDE_specific_hexagon_perimeter_l505_50596


namespace NUMINAMATH_CALUDE_sqrt_a_minus_4_real_l505_50530

theorem sqrt_a_minus_4_real (a : ℝ) : (∃ x : ℝ, x^2 = a - 4) ↔ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_a_minus_4_real_l505_50530


namespace NUMINAMATH_CALUDE_exists_three_integers_with_cube_product_l505_50579

/-- A set of 9 distinct integers with prime factors at most 3 -/
def SetWithPrimeFactorsUpTo3 : Type :=
  { S : Finset ℕ // S.card = 9 ∧ ∀ n ∈ S, ∀ p : ℕ, Nat.Prime p → p ∣ n → p ≤ 3 }

/-- The theorem stating that there exist three distinct integers in S whose product is a perfect cube -/
theorem exists_three_integers_with_cube_product (S : SetWithPrimeFactorsUpTo3) :
  ∃ a b c : ℕ, a ∈ S.val ∧ b ∈ S.val ∧ c ∈ S.val ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  ∃ k : ℕ, a * b * c = k^3 :=
sorry

end NUMINAMATH_CALUDE_exists_three_integers_with_cube_product_l505_50579


namespace NUMINAMATH_CALUDE_five_chicks_per_hen_l505_50580

/-- Represents the poultry farm scenario --/
structure PoultryFarm where
  num_hens : ℕ
  hen_to_rooster_ratio : ℕ
  total_chickens : ℕ

/-- Calculates the number of chicks per hen --/
def chicks_per_hen (farm : PoultryFarm) : ℕ :=
  let num_roosters := farm.num_hens / farm.hen_to_rooster_ratio
  let num_adult_chickens := farm.num_hens + num_roosters
  let num_chicks := farm.total_chickens - num_adult_chickens
  num_chicks / farm.num_hens

/-- Theorem stating that for the given farm conditions, each hen has 5 chicks --/
theorem five_chicks_per_hen (farm : PoultryFarm) 
    (h1 : farm.num_hens = 12)
    (h2 : farm.hen_to_rooster_ratio = 3)
    (h3 : farm.total_chickens = 76) : 
  chicks_per_hen farm = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_chicks_per_hen_l505_50580


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_by_five_l505_50518

theorem least_addition_for_divisibility_by_five (n : ℕ) (h : n = 821562) :
  ∃ k : ℕ, k = 3 ∧ (n + k) % 5 = 0 ∧ ∀ m : ℕ, m < k → (n + m) % 5 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_by_five_l505_50518


namespace NUMINAMATH_CALUDE_trig_expressions_l505_50541

theorem trig_expressions (α : Real) (h : Real.tan α = 2) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 6/11 ∧
  (1/4) * Real.sin α ^ 2 + (1/3) * Real.sin α * Real.cos α + (1/2) * Real.cos α ^ 2 = 13/30 := by
  sorry

end NUMINAMATH_CALUDE_trig_expressions_l505_50541


namespace NUMINAMATH_CALUDE_nail_hammering_l505_50567

theorem nail_hammering (k : ℝ) (h1 : 0 < k) (h2 : k < 1) : 
  (4 : ℝ) / 7 + 4 / 7 * k + 4 / 7 * k^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_nail_hammering_l505_50567


namespace NUMINAMATH_CALUDE_exists_fixed_point_l505_50510

variable {α : Type*} [Finite α]

def IsIncreasing (f : Set α → Set α) : Prop :=
  ∀ X Y : Set α, X ⊆ Y → f X ⊆ f Y

theorem exists_fixed_point (f : Set α → Set α) (hf : IsIncreasing f) :
    ∃ H₀ : Set α, f H₀ = H₀ := by
  sorry

end NUMINAMATH_CALUDE_exists_fixed_point_l505_50510


namespace NUMINAMATH_CALUDE_derivative_of_even_function_is_odd_l505_50592

/-- A function f: ℝ → ℝ that is even, i.e., f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The derivative of a function f: ℝ → ℝ -/
def DerivativeOf (g f : ℝ → ℝ) : Prop :=
  ∀ x, HasDerivAt f (g x) x

theorem derivative_of_even_function_is_odd
  (f g : ℝ → ℝ) (hf : EvenFunction f) (hg : DerivativeOf g f) :
  ∀ x, g (-x) = -g x :=
sorry

end NUMINAMATH_CALUDE_derivative_of_even_function_is_odd_l505_50592


namespace NUMINAMATH_CALUDE_wizard_collection_value_l505_50551

def base7ToBase10 (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem wizard_collection_value :
  let crystal_ball := [3, 4, 2, 6]
  let wand := [0, 5, 6, 1]
  let book := [2, 0, 2]
  base7ToBase10 crystal_ball + base7ToBase10 wand + base7ToBase10 book = 2959 := by
  sorry

end NUMINAMATH_CALUDE_wizard_collection_value_l505_50551


namespace NUMINAMATH_CALUDE_eleven_divides_four_digit_palindromes_l505_50573

/-- A four-digit palindrome is a number of the form abba where a and b are digits. -/
def FourDigitPalindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem eleven_divides_four_digit_palindromes :
  ∀ n : ℕ, FourDigitPalindrome n → 11 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_eleven_divides_four_digit_palindromes_l505_50573


namespace NUMINAMATH_CALUDE_amy_bought_21_tickets_l505_50540

/-- Calculates the number of tickets Amy bought at the fair -/
def tickets_bought (initial_tickets total_tickets : ℕ) : ℕ :=
  total_tickets - initial_tickets

/-- Proves that Amy bought 21 tickets at the fair -/
theorem amy_bought_21_tickets :
  tickets_bought 33 54 = 21 := by
  sorry

end NUMINAMATH_CALUDE_amy_bought_21_tickets_l505_50540


namespace NUMINAMATH_CALUDE_angle_sum_quarter_range_l505_50593

-- Define acute and obtuse angles
def is_acute (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2
def is_obtuse (β : Real) : Prop := Real.pi / 2 < β ∧ β < Real.pi

-- Main theorem
theorem angle_sum_quarter_range (α β : Real) 
  (h_acute : is_acute α) (h_obtuse : is_obtuse β) :
  Real.pi / 8 < 0.25 * (α + β) ∧ 0.25 * (α + β) < 3 * Real.pi / 8 := by
  sorry

#check angle_sum_quarter_range

end NUMINAMATH_CALUDE_angle_sum_quarter_range_l505_50593


namespace NUMINAMATH_CALUDE_pizza_slice_price_l505_50568

theorem pizza_slice_price 
  (whole_pizza_price : ℝ)
  (slices_sold : ℕ)
  (whole_pizzas_sold : ℕ)
  (total_revenue : ℝ)
  (h1 : whole_pizza_price = 15)
  (h2 : slices_sold = 24)
  (h3 : whole_pizzas_sold = 3)
  (h4 : total_revenue = 117) :
  ∃ (price_per_slice : ℝ), 
    price_per_slice * slices_sold + whole_pizza_price * whole_pizzas_sold = total_revenue ∧ 
    price_per_slice = 3 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_price_l505_50568


namespace NUMINAMATH_CALUDE_total_dress_designs_is_40_l505_50559

/-- The number of color choices for a dress design. -/
def num_colors : ℕ := 4

/-- The number of pattern choices for a dress design. -/
def num_patterns : ℕ := 5

/-- The number of fabric type choices for a dress design. -/
def num_fabric_types : ℕ := 2

/-- The total number of possible dress designs. -/
def total_designs : ℕ := num_colors * num_patterns * num_fabric_types

/-- Theorem stating that the total number of possible dress designs is 40. -/
theorem total_dress_designs_is_40 : total_designs = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_is_40_l505_50559


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l505_50590

theorem quadratic_inequality_solution_set (a : ℝ) (ha : a > 0) :
  {x : ℝ | x^2 - 4*a*x - 5*a^2 < 0} = {x : ℝ | -a < x ∧ x < 5*a} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l505_50590


namespace NUMINAMATH_CALUDE_total_marks_for_exam_l505_50553

/-- Calculates the total marks given the number of candidates and average score -/
def totalMarks (numCandidates : ℕ) (averageScore : ℚ) : ℚ :=
  numCandidates * averageScore

/-- Proves that for 250 candidates with an average score of 42, the total marks is 10500 -/
theorem total_marks_for_exam : totalMarks 250 42 = 10500 := by
  sorry

#eval totalMarks 250 42

end NUMINAMATH_CALUDE_total_marks_for_exam_l505_50553

import Mathlib

namespace NUMINAMATH_CALUDE_boxes_with_neither_l2072_207295

-- Define the set universe
def U : Set Nat := {n | n ≤ 15}

-- Define the set of boxes with crayons
def C : Set Nat := {n ∈ U | n ≤ 9}

-- Define the set of boxes with markers
def M : Set Nat := {n ∈ U | n ≤ 5}

-- Define the set of boxes with both crayons and markers
def B : Set Nat := {n ∈ U | n ≤ 4}

theorem boxes_with_neither (hU : Fintype U) (hC : Fintype C) (hM : Fintype M) (hB : Fintype B) :
  Fintype.card U - (Fintype.card C + Fintype.card M - Fintype.card B) = 5 := by
  sorry


end NUMINAMATH_CALUDE_boxes_with_neither_l2072_207295


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l2072_207222

theorem arithmetic_sequence_solution (y : ℝ) (h : y > 0) :
  (2^2 + 5^2) / 2 = y^2 → y = Real.sqrt (29 / 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l2072_207222


namespace NUMINAMATH_CALUDE_joan_football_games_l2072_207206

/-- Given that Joan went to 4 football games this year and 13 games in total,
    prove that she went to 9 games last year. -/
theorem joan_football_games (games_this_year games_total : ℕ)
  (h1 : games_this_year = 4)
  (h2 : games_total = 13) :
  games_total - games_this_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_football_games_l2072_207206


namespace NUMINAMATH_CALUDE_sprinkles_remaining_l2072_207212

theorem sprinkles_remaining (initial_cans : ℕ) (remaining_cans : ℕ) : 
  initial_cans = 12 →
  remaining_cans = initial_cans / 2 - 3 →
  remaining_cans = 3 := by
sorry

end NUMINAMATH_CALUDE_sprinkles_remaining_l2072_207212


namespace NUMINAMATH_CALUDE_ryan_tokens_l2072_207259

def arcade_tokens (initial_tokens : ℕ) : ℕ :=
  let pacman_tokens := (2 * initial_tokens) / 3
  let remaining_after_pacman := initial_tokens - pacman_tokens
  let candy_crush_tokens := remaining_after_pacman / 2
  let remaining_after_candy_crush := remaining_after_pacman - candy_crush_tokens
  let skeball_tokens := min remaining_after_candy_crush 7
  let parents_bought := 10 * skeball_tokens
  remaining_after_candy_crush - skeball_tokens + parents_bought

theorem ryan_tokens : arcade_tokens 36 = 66 := by
  sorry

end NUMINAMATH_CALUDE_ryan_tokens_l2072_207259


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2072_207232

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (2 * i) / (1 - i)
  Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2072_207232


namespace NUMINAMATH_CALUDE_price_change_after_increase_and_discounts_l2072_207279

theorem price_change_after_increase_and_discounts :
  let initial_price : ℝ := 100
  let increased_price := initial_price * 1.5
  let price_after_first_discount := increased_price * 0.9
  let price_after_second_discount := price_after_first_discount * 0.85
  let final_price := price_after_second_discount * 0.8
  (final_price - initial_price) / initial_price = -0.082 :=
by sorry

end NUMINAMATH_CALUDE_price_change_after_increase_and_discounts_l2072_207279


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2072_207266

theorem consecutive_integers_product_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 272 → x + (x + 1) = 33 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2072_207266


namespace NUMINAMATH_CALUDE_number_of_friends_l2072_207211

-- Define the total number of stickers
def total_stickers : ℕ := 72

-- Define the number of stickers each friend receives
def stickers_per_friend : ℕ := 8

-- Theorem to prove the number of friends receiving stickers
theorem number_of_friends : total_stickers / stickers_per_friend = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_friends_l2072_207211


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2072_207203

-- Define the Cartesian coordinate system
def Cartesian := ℝ × ℝ

-- Define a point in the Cartesian coordinate system
def point : Cartesian := (1, -2)

-- Define the fourth quadrant
def fourth_quadrant (p : Cartesian) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_fourth_quadrant :
  fourth_quadrant point := by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2072_207203


namespace NUMINAMATH_CALUDE_complex_number_problem_l2072_207258

theorem complex_number_problem (z : ℂ) 
  (h1 : ∃ (r : ℝ), z + 2*I = r)
  (h2 : ∃ (s : ℝ), z / (2 - I) = s) :
  z = 4 - 2*I ∧ 
  ∀ (a : ℝ), (z - a*I)^2 ∈ {w : ℂ | 0 < w.re ∧ 0 < w.im} ↔ -6 < a ∧ a < -2 :=
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2072_207258


namespace NUMINAMATH_CALUDE_power_fraction_equality_l2072_207273

theorem power_fraction_equality : (10^20 : ℝ) / (50^10 : ℝ) = 2^10 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l2072_207273


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l2072_207200

theorem least_perimeter_triangle (a b c : ℕ) : 
  a = 40 → b = 48 → c > 0 → a + b > c → a + c > b → b + c > a → 
  (∀ x : ℕ, x > 0 → a + b > x → a + x > b → b + x > a → a + b + x ≥ a + b + c) →
  a + b + c = 97 := by sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l2072_207200


namespace NUMINAMATH_CALUDE_vision_assistance_l2072_207236

theorem vision_assistance (total : ℕ) (glasses_percent : ℚ) (contacts_percent : ℚ)
  (h_total : total = 40)
  (h_glasses : glasses_percent = 25 / 100)
  (h_contacts : contacts_percent = 40 / 100) :
  total - (total * glasses_percent).floor - (total * contacts_percent).floor = 14 := by
  sorry

end NUMINAMATH_CALUDE_vision_assistance_l2072_207236


namespace NUMINAMATH_CALUDE_thursday_max_attendance_l2072_207280

/-- Represents the days of the week --/
inductive Day
| monday
| tuesday
| wednesday
| thursday
| friday

/-- Represents a team member --/
inductive Member
| dave
| elena
| fiona
| george
| hannah

/-- Returns whether a member can attend on a given day --/
def canAttend (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.dave, Day.wednesday => false
  | Member.dave, Day.thursday => false
  | Member.elena, Day.monday => false
  | Member.elena, Day.thursday => false
  | Member.elena, Day.friday => false
  | Member.fiona, Day.monday => false
  | Member.fiona, Day.tuesday => false
  | Member.fiona, Day.friday => false
  | Member.george, Day.tuesday => false
  | Member.george, Day.wednesday => false
  | Member.george, Day.friday => false
  | Member.hannah, Day.monday => false
  | Member.hannah, Day.wednesday => false
  | Member.hannah, Day.thursday => false
  | _, _ => true

/-- Counts the number of members who can attend on a given day --/
def countAttendees (d : Day) : Nat :=
  (List.filter (fun m => canAttend m d) [Member.dave, Member.elena, Member.fiona, Member.george, Member.hannah]).length

/-- Theorem: Thursday has the maximum number of attendees --/
theorem thursday_max_attendance :
  ∀ d : Day, countAttendees Day.thursday ≥ countAttendees d :=
by sorry


end NUMINAMATH_CALUDE_thursday_max_attendance_l2072_207280


namespace NUMINAMATH_CALUDE_equation_solution_l2072_207294

theorem equation_solution : 
  {x : ℝ | Real.sqrt ((1 + Real.sqrt 2) ^ x) + Real.sqrt ((1 - Real.sqrt 2) ^ x) = 3} = {2, -2} :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2072_207294


namespace NUMINAMATH_CALUDE_no_subdivision_for_1986_plots_l2072_207223

theorem no_subdivision_for_1986_plots : ¬ ∃ (n : ℕ), 8 * n + 9 = 1986 := by
  sorry

end NUMINAMATH_CALUDE_no_subdivision_for_1986_plots_l2072_207223


namespace NUMINAMATH_CALUDE_twelve_pharmacies_not_enough_l2072_207250

/-- Represents a grid of streets -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a pharmacy on the grid -/
structure Pharmacy :=
  (row : Nat)
  (col : Nat)

/-- The maximum walking distance to a pharmacy -/
def max_walking_distance : Nat := 3

/-- Calculate the number of street segments covered by a pharmacy -/
def covered_segments (g : Grid) (p : Pharmacy) : Nat :=
  let coverage_side := 2 * max_walking_distance + 1
  min coverage_side g.rows * min coverage_side g.cols

/-- Calculate the total number of street segments in the grid -/
def total_segments (g : Grid) : Nat :=
  2 * g.rows * (g.cols - 1) + 2 * g.cols * (g.rows - 1)

/-- The main theorem to be proved -/
theorem twelve_pharmacies_not_enough :
  ∀ (pharmacies : List Pharmacy),
    pharmacies.length = 12 →
    ∃ (g : Grid),
      g.rows = 9 ∧ g.cols = 9 ∧
      (pharmacies.map (covered_segments g)).sum < total_segments g := by
  sorry


end NUMINAMATH_CALUDE_twelve_pharmacies_not_enough_l2072_207250


namespace NUMINAMATH_CALUDE_fraction_invariance_l2072_207244

theorem fraction_invariance (x y : ℝ) : 
  (2 * x) / (3 * x - y) = (2 * (3 * x)) / (3 * (3 * x) - (3 * y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_invariance_l2072_207244


namespace NUMINAMATH_CALUDE_translation_problem_l2072_207233

def translation (z w : ℂ) : ℂ := z + w

theorem translation_problem (t : ℂ → ℂ) :
  (∃ w : ℂ, ∀ z, t z = translation z w) →
  t (1 + 3*I) = 5 + 7*I →
  t (2 - 2*I) = 6 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_translation_problem_l2072_207233


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2072_207205

theorem complex_equation_solution (z : ℂ) :
  (1 - Complex.I) * z = 2 * Complex.I → z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2072_207205


namespace NUMINAMATH_CALUDE_weighted_distances_sum_l2072_207296

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  V : ℝ  -- Volume
  S : Fin 4 → ℝ  -- Face areas
  d : Fin 4 → ℝ  -- Distances from a point to each face
  k : ℝ  -- Constant ratio
  h_positive : V > 0
  S_positive : ∀ i, S i > 0
  d_positive : ∀ i, d i > 0
  k_positive : k > 0
  h_ratio : ∀ i : Fin 4, S i / (i.val + 1 : ℝ) = k

/-- The sum of weighted distances equals three times the volume divided by k -/
theorem weighted_distances_sum (p : TriangularPyramid) :
  (p.d 0) + 2 * (p.d 1) + 3 * (p.d 2) + 4 * (p.d 3) = 3 * p.V / p.k := by
  sorry

end NUMINAMATH_CALUDE_weighted_distances_sum_l2072_207296


namespace NUMINAMATH_CALUDE_average_problem_l2072_207237

theorem average_problem (y : ℝ) (h : (15 + 24 + 32 + y) / 4 = 26) : y = 33 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l2072_207237


namespace NUMINAMATH_CALUDE_fish_total_weight_l2072_207246

/-- The weight of a fish with specific weight relationships between its parts -/
def fish_weight (head body tail : ℝ) : Prop :=
  tail = 1 ∧ 
  head = tail + body / 2 ∧ 
  body = head + tail

theorem fish_total_weight : 
  ∀ (head body tail : ℝ), 
  fish_weight head body tail → 
  head + body + tail = 8 := by
  sorry

end NUMINAMATH_CALUDE_fish_total_weight_l2072_207246


namespace NUMINAMATH_CALUDE_cubic_function_monotonicity_l2072_207270

-- Define the function f(x) = ax^3
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3

-- State the theorem
theorem cubic_function_monotonicity (a : ℝ) (h : a ≠ 0) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (a > 0 → f a x₁ < f a x₂) ∧ (a < 0 → f a x₁ > f a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_monotonicity_l2072_207270


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2072_207255

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_theorem (a b c : ℝ) :
  (f a b c 0 = 0) →
  (∀ x, f a b c (x + 1) = f a b c x + x + 1) →
  (∀ x, f a b c x = x^2 / 2 + x / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2072_207255


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2072_207275

/-- Given a geometric sequence {a_n} with sum S_n of the first n terms,
    if a_3 + 2a_6 = 0, then S_3/S_6 = 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  (∀ n : ℕ, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum formula
  a 3 + 2 * a 6 = 0 →  -- given condition
  S 3 / S 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2072_207275


namespace NUMINAMATH_CALUDE_original_class_strength_l2072_207230

/-- Proves that the original strength of an adult class is 17 students given the conditions. -/
theorem original_class_strength (original_average : ℝ) (new_students : ℕ) (new_average : ℝ) (average_decrease : ℝ) :
  original_average = 40 →
  new_students = 17 →
  new_average = 32 →
  average_decrease = 4 →
  ∃ (x : ℕ), x = 17 ∧ 
    (x : ℝ) * original_average + (new_students : ℝ) * new_average = 
      ((x : ℝ) + (new_students : ℝ)) * (original_average - average_decrease) := by
  sorry

#check original_class_strength

end NUMINAMATH_CALUDE_original_class_strength_l2072_207230


namespace NUMINAMATH_CALUDE_union_of_a_and_b_l2072_207216

def U : Set Nat := {0, 1, 2, 3, 4}

theorem union_of_a_and_b (A B : Set Nat) 
  (h1 : U = {0, 1, 2, 3, 4})
  (h2 : (U \ A) = {1, 2})
  (h3 : B = {1, 3}) :
  A ∪ B = {0, 1, 3, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_a_and_b_l2072_207216


namespace NUMINAMATH_CALUDE_solve_commencement_addresses_l2072_207288

def commencement_addresses_problem (sandoval hawkins sloan : ℕ) : Prop :=
  sandoval = 12 ∧
  hawkins = sandoval / 2 ∧
  sloan > sandoval ∧
  sandoval + hawkins + sloan = 40 ∧
  sloan - sandoval = 10

theorem solve_commencement_addresses :
  ∃ (sandoval hawkins sloan : ℕ), commencement_addresses_problem sandoval hawkins sloan :=
by
  sorry

end NUMINAMATH_CALUDE_solve_commencement_addresses_l2072_207288


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_l2072_207293

/-- The sum of an infinite geometric series with first term 5/3 and common ratio -9/20 is 100/87 -/
theorem infinite_geometric_series_sum :
  let a : ℚ := 5/3
  let r : ℚ := -9/20
  let S := a / (1 - r)
  S = 100/87 := by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_l2072_207293


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2072_207267

theorem quadratic_inequality (x : ℝ) (h : x ∈ Set.Icc 0 1) :
  |x^2 - x + 1/8| ≤ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2072_207267


namespace NUMINAMATH_CALUDE_johns_per_sheet_price_l2072_207213

def johns_sitting_fee : ℝ := 125
def sams_sitting_fee : ℝ := 140
def sams_per_sheet : ℝ := 1.50
def num_sheets : ℝ := 12

theorem johns_per_sheet_price (johns_per_sheet : ℝ) : 
  johns_per_sheet * num_sheets + johns_sitting_fee = 
  sams_per_sheet * num_sheets + sams_sitting_fee → 
  johns_per_sheet = 2.75 := by
sorry

end NUMINAMATH_CALUDE_johns_per_sheet_price_l2072_207213


namespace NUMINAMATH_CALUDE_lamp_arrangement_count_l2072_207225

def number_of_lamps : ℕ := 10
def lamps_to_turn_off : ℕ := 3
def available_positions : ℕ := number_of_lamps - 2 - lamps_to_turn_off + 1

theorem lamp_arrangement_count : 
  Nat.choose available_positions lamps_to_turn_off = 20 := by
  sorry

end NUMINAMATH_CALUDE_lamp_arrangement_count_l2072_207225


namespace NUMINAMATH_CALUDE_log_problem_l2072_207227

theorem log_problem :
  let x := (Real.log 2 / Real.log 8) ^ (Real.log 8 / Real.log 2)
  Real.log x / Real.log 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l2072_207227


namespace NUMINAMATH_CALUDE_triangle_inequalities_l2072_207290

theorem triangle_inequalities (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a < b + c) (hbc : b < a + c) (hca : c < a + b) :
  (a + b + c = 2 → a^2 + b^2 + c^2 + 2*a*b*c < 2) ∧
  (a + b + c = 1 → a^2 + b^2 + c^2 + 4*a*b*c < 1/2) ∧
  (a + b + c = 1 → 5*(a^2 + b^2 + c^2) + 18*a*b*c > 7/3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l2072_207290


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l2072_207219

theorem complex_sum_magnitude (a b c : ℂ) :
  Complex.abs a = 2 →
  Complex.abs b = 2 →
  Complex.abs c = 2 →
  a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = 0 →
  Complex.abs (a + b + c) = 6 + 2 * Real.sqrt 6 ∨
  Complex.abs (a + b + c) = 6 - 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l2072_207219


namespace NUMINAMATH_CALUDE_inscribed_right_triangle_exists_l2072_207277

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the condition for a point to be inside a circle
def inside_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 < c.radius^2

-- Define the condition for a point to be on the circumference of a circle
def on_circumference (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define a right triangle
structure RightTriangle where
  A : Point
  B : Point
  C : Point
  right_angle : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem inscribed_right_triangle_exists (c : Circle) (A B : Point)
  (h_A : inside_circle A c) (h_B : inside_circle B c) :
  ∃ (C : Point), on_circumference C c ∧
    ∃ (t : RightTriangle), t.A = A ∧ t.B = B ∧ t.C = C :=
sorry

end NUMINAMATH_CALUDE_inscribed_right_triangle_exists_l2072_207277


namespace NUMINAMATH_CALUDE_least_multiple_17_greater_500_l2072_207260

theorem least_multiple_17_greater_500 : ∃ (n : ℕ), n * 17 = 510 ∧ 
  510 > 500 ∧ (∀ m : ℕ, m * 17 > 500 → m * 17 ≥ 510) := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_17_greater_500_l2072_207260


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l2072_207282

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 8 ways to distribute 7 indistinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l2072_207282


namespace NUMINAMATH_CALUDE_odd_even_function_problem_l2072_207221

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem odd_even_function_problem (f g : ℝ → ℝ) 
  (h_odd : IsOdd f) (h_even : IsEven g)
  (h1 : f (-3) + g 3 = 2) (h2 : f 3 + g (-3) = 4) : 
  g 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_function_problem_l2072_207221


namespace NUMINAMATH_CALUDE_clothes_washer_discount_l2072_207202

theorem clothes_washer_discount (original_price : ℝ) 
  (discount1 discount2 discount3 : ℝ) : 
  original_price = 500 →
  discount1 = 0.1 →
  discount2 = 0.2 →
  discount3 = 0.05 →
  (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)) / original_price = 0.684 := by
sorry

end NUMINAMATH_CALUDE_clothes_washer_discount_l2072_207202


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l2072_207241

/-- The number of combinations of k items chosen from n items -/
def combination (n k : ℕ) : ℕ := Nat.choose n k

/-- There are 10 available toppings -/
def num_toppings : ℕ := 10

/-- We want to choose 3 toppings -/
def toppings_to_choose : ℕ := 3

theorem pizza_toppings_combinations :
  combination num_toppings toppings_to_choose = 120 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l2072_207241


namespace NUMINAMATH_CALUDE_composite_product_ratio_l2072_207286

def first_twelve_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21]

def product_first_six : ℕ := (first_twelve_composites.take 6).prod

def product_next_six : ℕ := (first_twelve_composites.drop 6).prod

theorem composite_product_ratio : 
  (product_first_six : ℚ) / product_next_six = 2 / 245 := by sorry

end NUMINAMATH_CALUDE_composite_product_ratio_l2072_207286


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2072_207287

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 3}

theorem complement_of_A_in_U :
  Set.compl A = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2072_207287


namespace NUMINAMATH_CALUDE_average_geometric_sequence_l2072_207265

theorem average_geometric_sequence (z : ℝ) : 
  (z + 3*z + 9*z + 27*z + 81*z) / 5 = 24.2 * z := by
  sorry

end NUMINAMATH_CALUDE_average_geometric_sequence_l2072_207265


namespace NUMINAMATH_CALUDE_min_value_implies_a_l2072_207224

/-- Given a function f(x) = 4x + a/x where x > 0 and a > 0, 
    if f attains its minimum value at x = 2, then a = 16 -/
theorem min_value_implies_a (a : ℝ) (h_a : a > 0) :
  (∀ x > 0, 4 * x + a / x ≥ 4 * 2 + a / 2) →
  (∃ x > 0, 4 * x + a / x = 4 * 2 + a / 2) →
  a = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l2072_207224


namespace NUMINAMATH_CALUDE_group_size_proof_l2072_207257

theorem group_size_proof (total_collection : ℚ) (paise_per_rupee : ℕ) : 
  (total_collection = 32.49) →
  (paise_per_rupee = 100) →
  ∃ n : ℕ, (n * n = total_collection * paise_per_rupee) ∧ (n = 57) :=
by sorry

end NUMINAMATH_CALUDE_group_size_proof_l2072_207257


namespace NUMINAMATH_CALUDE_tank_capacity_ratio_l2072_207276

theorem tank_capacity_ratio : 
  let h_a : ℝ := 8
  let c_a : ℝ := 8
  let h_b : ℝ := 8
  let c_b : ℝ := 10
  let r_a : ℝ := c_a / (2 * Real.pi)
  let r_b : ℝ := c_b / (2 * Real.pi)
  let v_a : ℝ := Real.pi * r_a^2 * h_a
  let v_b : ℝ := Real.pi * r_b^2 * h_b
  v_a / v_b = 0.64
  := by sorry

end NUMINAMATH_CALUDE_tank_capacity_ratio_l2072_207276


namespace NUMINAMATH_CALUDE_hearty_beads_count_l2072_207254

/-- The number of beads Hearty has in total -/
def total_beads (blue_packages red_packages beads_per_package : ℕ) : ℕ :=
  (blue_packages + red_packages) * beads_per_package

/-- Proof that Hearty has 320 beads in total -/
theorem hearty_beads_count :
  total_beads 3 5 40 = 320 := by
  sorry

end NUMINAMATH_CALUDE_hearty_beads_count_l2072_207254


namespace NUMINAMATH_CALUDE_equivalent_equations_product_l2072_207209

/-- Given that the equation a^8xy - 2a^7y - 3a^6x = 2a^5(b^5 - 2) is equivalent to 
    (a^m*x - 2a^n)(a^p*y - 3a^3) = 2a^5*b^5 for some integers m, n, and p, 
    prove that m*n*p = 60 -/
theorem equivalent_equations_product (a b x y : ℝ) (m n p : ℤ) 
  (h1 : a^8*x*y - 2*a^7*y - 3*a^6*x = 2*a^5*(b^5 - 2))
  (h2 : (a^m*x - 2*a^n)*(a^p*y - 3*a^3) = 2*a^5*b^5) :
  m * n * p = 60 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_equations_product_l2072_207209


namespace NUMINAMATH_CALUDE_evaluate_expression_l2072_207248

theorem evaluate_expression : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2072_207248


namespace NUMINAMATH_CALUDE_arrangement_is_correct_l2072_207231

-- Define the metals and safes
inductive Metal
| Gold | Silver | Bronze | Platinum | Nickel

inductive Safe
| One | Two | Three | Four | Five

-- Define the arrangement as a function from Safe to Metal
def Arrangement := Safe → Metal

-- Define the statements on the safes
def statement1 (a : Arrangement) : Prop :=
  a Safe.Two = Metal.Gold ∨ a Safe.Three = Metal.Gold

def statement2 (a : Arrangement) : Prop :=
  a Safe.One = Metal.Silver

def statement3 (a : Arrangement) : Prop :=
  a Safe.Three ≠ Metal.Bronze

def statement4 (a : Arrangement) : Prop :=
  (a Safe.One = Metal.Nickel ∧ a Safe.Two = Metal.Gold) ∨
  (a Safe.Two = Metal.Nickel ∧ a Safe.Three = Metal.Gold) ∨
  (a Safe.Three = Metal.Nickel ∧ a Safe.Four = Metal.Gold) ∨
  (a Safe.Four = Metal.Nickel ∧ a Safe.Five = Metal.Gold)

def statement5 (a : Arrangement) : Prop :=
  (a Safe.One = Metal.Bronze ∧ a Safe.Two = Metal.Platinum) ∨
  (a Safe.Two = Metal.Bronze ∧ a Safe.Three = Metal.Platinum) ∨
  (a Safe.Three = Metal.Bronze ∧ a Safe.Four = Metal.Platinum) ∨
  (a Safe.Four = Metal.Bronze ∧ a Safe.Five = Metal.Platinum)

-- Define the correct arrangement
def correctArrangement : Arrangement :=
  fun s => match s with
  | Safe.One => Metal.Nickel
  | Safe.Two => Metal.Silver
  | Safe.Three => Metal.Bronze
  | Safe.Four => Metal.Platinum
  | Safe.Five => Metal.Gold

-- Theorem statement
theorem arrangement_is_correct (a : Arrangement) :
  (∃! s, a s = Metal.Gold ∧
    (s = Safe.One → statement1 a) ∧
    (s = Safe.Two → statement2 a) ∧
    (s = Safe.Three → statement3 a) ∧
    (s = Safe.Four → statement4 a) ∧
    (s = Safe.Five → statement5 a)) →
  (∀ s, a s = correctArrangement s) :=
sorry

end NUMINAMATH_CALUDE_arrangement_is_correct_l2072_207231


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2072_207299

theorem trigonometric_identity : 
  (Real.sin (20 * π / 180) / Real.cos (20 * π / 180)) + 
  (Real.sin (40 * π / 180) / Real.cos (40 * π / 180)) + 
  Real.tan (60 * π / 180) * Real.tan (20 * π / 180) * Real.tan (40 * π / 180) = 
  Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2072_207299


namespace NUMINAMATH_CALUDE_coin_jar_problem_l2072_207201

theorem coin_jar_problem (x : ℕ) : 
  (x : ℚ) * (1 + 5 + 10 + 25) / 100 = 20 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_coin_jar_problem_l2072_207201


namespace NUMINAMATH_CALUDE_representatives_selection_count_l2072_207281

def num_boys : ℕ := 5
def num_girls : ℕ := 4
def total_students : ℕ := num_boys + num_girls
def num_representatives : ℕ := 3

theorem representatives_selection_count :
  (Nat.choose total_students num_representatives) - (Nat.choose num_boys num_representatives) = 74 := by
  sorry

end NUMINAMATH_CALUDE_representatives_selection_count_l2072_207281


namespace NUMINAMATH_CALUDE_tribe_assignment_l2072_207278

-- Define the two tribes
inductive Tribe
| Triussa
| La

-- Define a person as having a tribe
structure Person where
  tribe : Tribe

-- Define the three people
def person1 : Person := sorry
def person2 : Person := sorry
def person3 : Person := sorry

-- Define what it means for a statement to be true
def isTrueStatement (p : Person) (s : Prop) : Prop :=
  (p.tribe = Tribe.Triussa ∧ s) ∨ (p.tribe = Tribe.La ∧ ¬s)

-- Define the statements made by each person
def statement1 : Prop := 
  (person1.tribe = Tribe.Triussa ∧ person2.tribe = Tribe.La ∧ person3.tribe = Tribe.La) ∨
  (person1.tribe = Tribe.La ∧ person2.tribe = Tribe.Triussa ∧ person3.tribe = Tribe.La) ∨
  (person1.tribe = Tribe.La ∧ person2.tribe = Tribe.La ∧ person3.tribe = Tribe.Triussa)

def statement2 : Prop := person3.tribe = Tribe.La

def statement3 : Prop := person1.tribe = Tribe.La

-- Theorem to prove
theorem tribe_assignment :
  isTrueStatement person1 statement1 ∧
  isTrueStatement person2 statement2 ∧
  isTrueStatement person3 statement3 →
  person1.tribe = Tribe.La ∧ person2.tribe = Tribe.La ∧ person3.tribe = Tribe.Triussa :=
sorry

end NUMINAMATH_CALUDE_tribe_assignment_l2072_207278


namespace NUMINAMATH_CALUDE_piper_wing_count_l2072_207289

/-- The number of commercial planes in the air exhibition -/
def num_planes : ℕ := 45

/-- The number of wings on each commercial plane -/
def wings_per_plane : ℕ := 2

/-- The total number of wings counted by Piper -/
def total_wings : ℕ := num_planes * wings_per_plane

theorem piper_wing_count : total_wings = 90 := by
  sorry

end NUMINAMATH_CALUDE_piper_wing_count_l2072_207289


namespace NUMINAMATH_CALUDE_cube_volume_l2072_207247

theorem cube_volume (a : ℝ) (h : 3 * a^2 - 8 * a - 12 = 0) : a^3 = 64 := by
  sorry

#check cube_volume

end NUMINAMATH_CALUDE_cube_volume_l2072_207247


namespace NUMINAMATH_CALUDE_f_unique_positive_zero_implies_a_range_l2072_207262

/-- The function f(x) = ax³ - 3x² + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The statement that f has only one zero point -/
def has_unique_zero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

theorem f_unique_positive_zero_implies_a_range (a : ℝ) :
  (has_unique_zero (f a)) ∧ (∃ x₀ > 0, f a x₀ = 0) → a < -2 :=
sorry

end NUMINAMATH_CALUDE_f_unique_positive_zero_implies_a_range_l2072_207262


namespace NUMINAMATH_CALUDE_sin_sum_zero_l2072_207285

theorem sin_sum_zero : 
  Real.sin (-1071 * π / 180) * Real.sin (99 * π / 180) + 
  Real.sin (-171 * π / 180) * Real.sin (-261 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_zero_l2072_207285


namespace NUMINAMATH_CALUDE_line_through_midpoint_parallel_to_PR_l2072_207283

/-- Given points P, Q, R in a 2D plane, prove that if a line y = mx + b is parallel to PR
    and passes through the midpoint of QR, then b = -4. -/
theorem line_through_midpoint_parallel_to_PR (P Q R : ℝ × ℝ) (m b : ℝ) : 
  P = (0, 0) →
  Q = (4, 0) →
  R = (1, 2) →
  (∀ x y : ℝ, y = m * x + b ↔ (∃ t : ℝ, (x, y) = ((1 - t) * P.1 + t * R.1, (1 - t) * P.2 + t * R.2))) →
  (m * ((Q.1 + R.1) / 2) + b = (Q.2 + R.2) / 2) →
  b = -4 := by
  sorry


end NUMINAMATH_CALUDE_line_through_midpoint_parallel_to_PR_l2072_207283


namespace NUMINAMATH_CALUDE_phone_not_answered_probability_l2072_207297

theorem phone_not_answered_probability 
  (p1 : ℝ) (p2 : ℝ) (p3 : ℝ) (p4 : ℝ)
  (h1 : p1 = 0.1) (h2 : p2 = 0.3) (h3 : p3 = 0.4) (h4 : p4 = 0.1) :
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_phone_not_answered_probability_l2072_207297


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_l2072_207271

/-- The number of different books in the 'crazy silly school' series -/
def num_books : ℕ := sorry

/-- The number of different movies in the 'crazy silly school' series -/
def num_movies : ℕ := 11

/-- The number of books you have read -/
def books_read : ℕ := 13

/-- The number of movies you have watched -/
def movies_watched : ℕ := 12

theorem crazy_silly_school_series :
  (books_read = movies_watched + 1) →
  (num_books = 12) :=
by sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_l2072_207271


namespace NUMINAMATH_CALUDE_buddy_cards_thursday_l2072_207253

def baseball_cards_problem (initial_cards : ℕ) (bought_wednesday : ℕ) : ℕ :=
  let tuesday_cards := initial_cards / 2
  let wednesday_cards := tuesday_cards + bought_wednesday
  let thursday_bought := tuesday_cards / 3
  wednesday_cards + thursday_bought

theorem buddy_cards_thursday (initial_cards : ℕ) (bought_wednesday : ℕ) 
  (h1 : initial_cards = 30) (h2 : bought_wednesday = 12) : 
  baseball_cards_problem initial_cards bought_wednesday = 32 := by
  sorry

end NUMINAMATH_CALUDE_buddy_cards_thursday_l2072_207253


namespace NUMINAMATH_CALUDE_pulley_system_force_l2072_207284

/-- The force required to move a load using a pulley system -/
def required_force (m : ℝ) (g : ℝ) : ℝ := 2 * m * g

/-- Theorem: The required force to move a 2 kg load with a pulley system is 20 N -/
theorem pulley_system_force :
  let m : ℝ := 2 -- mass of the load in kg
  let g : ℝ := 10 -- acceleration due to gravity in m/s²
  required_force m g = 20 := by
  sorry

#check pulley_system_force

end NUMINAMATH_CALUDE_pulley_system_force_l2072_207284


namespace NUMINAMATH_CALUDE_area_of_five_presentable_set_l2072_207239

/-- A complex number is five-presentable if it can be represented as w - 1/w for some complex number w with |w| = 5 -/
def FivePresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 5 ∧ z = w - 1 / w

/-- The set of all five-presentable complex numbers -/
def S : Set ℂ :=
  {z : ℂ | FivePresentable z}

/-- The area of a set in the complex plane -/
noncomputable def area (A : Set ℂ) : ℝ := sorry

theorem area_of_five_presentable_set :
  area S = 624 * Real.pi / 25 := by sorry

end NUMINAMATH_CALUDE_area_of_five_presentable_set_l2072_207239


namespace NUMINAMATH_CALUDE_final_price_calculation_final_price_is_841_32_l2072_207242

/-- Calculates the final price of a TV and sound system after discounts and tax --/
theorem final_price_calculation (tv_price sound_price : ℝ) 
  (tv_discount1 tv_discount2 sound_discount tax_rate : ℝ) : ℝ :=
  let tv_after_discounts := tv_price * (1 - tv_discount1) * (1 - tv_discount2)
  let sound_after_discount := sound_price * (1 - sound_discount)
  let total_before_tax := tv_after_discounts + sound_after_discount
  let tax_amount := total_before_tax * tax_rate
  let final_price := total_before_tax + tax_amount
  final_price

/-- Theorem stating that the final price is $841.32 given the specific conditions --/
theorem final_price_is_841_32 : 
  final_price_calculation 600 400 0.1 0.15 0.2 0.08 = 841.32 := by
  sorry

end NUMINAMATH_CALUDE_final_price_calculation_final_price_is_841_32_l2072_207242


namespace NUMINAMATH_CALUDE_equation_solution_l2072_207252

theorem equation_solution :
  ∃! x : ℚ, 2 * x + 3 = 500 - (4 * x + 5 * x) + 7 ∧ x = 504 / 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2072_207252


namespace NUMINAMATH_CALUDE_polar_bear_fish_consumption_l2072_207264

/-- Calculates the total number of fish buckets required for three polar bears for a week -/
theorem polar_bear_fish_consumption 
  (bear1_trout bear1_salmon : ℝ)
  (bear2_trout bear2_salmon : ℝ)
  (bear3_trout bear3_salmon : ℝ)
  (h1 : bear1_trout = 0.2)
  (h2 : bear1_salmon = 0.4)
  (h3 : bear2_trout = 0.3)
  (h4 : bear2_salmon = 0.5)
  (h5 : bear3_trout = 0.25)
  (h6 : bear3_salmon = 0.45)
  : (bear1_trout + bear1_salmon + bear2_trout + bear2_salmon + bear3_trout + bear3_salmon) * 7 = 14.7 := by
  sorry

#check polar_bear_fish_consumption

end NUMINAMATH_CALUDE_polar_bear_fish_consumption_l2072_207264


namespace NUMINAMATH_CALUDE_theater_ticket_price_l2072_207207

/-- Proves that the cost of an orchestra seat is $12 given the conditions of the theater problem -/
theorem theater_ticket_price
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (balcony_price : ℕ)
  (ticket_difference : ℕ)
  (h1 : total_tickets = 355)
  (h2 : total_revenue = 3320)
  (h3 : balcony_price = 8)
  (h4 : ticket_difference = 115) :
  ∃ (orchestra_price : ℕ),
    orchestra_price = 12 ∧
    ∃ (orchestra_tickets : ℕ),
      orchestra_tickets + (orchestra_tickets + ticket_difference) = total_tickets ∧
      orchestra_price * orchestra_tickets + balcony_price * (orchestra_tickets + ticket_difference) = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_price_l2072_207207


namespace NUMINAMATH_CALUDE_expression_simplification_l2072_207291

theorem expression_simplification : 
  (3 * 5 * 7) / (9 * 11 * 13) * (7 * 9 * 11 * 15) / (3 * 5 * 14) = 15 / 26 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2072_207291


namespace NUMINAMATH_CALUDE_expression_zero_iff_x_eq_two_l2072_207272

/-- The expression (x^2 - 4x + 4) / (3x - 9) equals zero if and only if x = 2 -/
theorem expression_zero_iff_x_eq_two (x : ℝ) : 
  (x^2 - 4*x + 4) / (3*x - 9) = 0 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_zero_iff_x_eq_two_l2072_207272


namespace NUMINAMATH_CALUDE_spirit_mixture_problem_l2072_207217

/-- Given three vessels a, b, and c with spirit concentrations of 45%, 30%, and 10% respectively,
    and a mixture of x litres from vessel a, 5 litres from vessel b, and 6 litres from vessel c
    resulting in a 26% spirit concentration, prove that x = 4 litres. -/
theorem spirit_mixture_problem (x : ℝ) :
  (0.45 * x + 0.30 * 5 + 0.10 * 6) / (x + 5 + 6) = 0.26 → x = 4 := by
  sorry

#check spirit_mixture_problem

end NUMINAMATH_CALUDE_spirit_mixture_problem_l2072_207217


namespace NUMINAMATH_CALUDE_isosceles_triangle_theorem_congruent_triangles_theorem_supplementary_angles_not_always_equal_supplements_of_equal_angles_are_equal_proposition_c_is_false_l2072_207229

-- Define the basic geometric concepts
def Triangle : Type := sorry
def Angle : Type := sorry
def Line : Type := sorry

-- Define the properties and relations
def equal_sides (t : Triangle) (s1 s2 : Nat) : Prop := sorry
def equal_angles (t : Triangle) (a1 a2 : Nat) : Prop := sorry
def congruent (t1 t2 : Triangle) : Prop := sorry
def corresponding_sides_equal (t1 t2 : Triangle) : Prop := sorry
def supplementary (a1 a2 : Angle) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def supplement_of (a1 a2 : Angle) : Prop := sorry

-- Theorem statements
theorem isosceles_triangle_theorem (t : Triangle) (s1 s2 a1 a2 : Nat) :
  equal_sides t s1 s2 → equal_angles t a1 a2 := sorry

theorem congruent_triangles_theorem (t1 t2 : Triangle) :
  congruent t1 t2 → corresponding_sides_equal t1 t2 := sorry

theorem supplementary_angles_not_always_equal :
  ∃ (a1 a2 : Angle), supplementary a1 a2 ∧ a1 ≠ a2 := sorry

theorem supplements_of_equal_angles_are_equal (a1 a2 a3 a4 : Angle) :
  a1 = a2 → supplement_of a1 a3 → supplement_of a2 a4 → a3 = a4 := sorry

-- The main theorem proving that proposition C is false while others are true
theorem proposition_c_is_false :
  (∀ (t : Triangle) (s1 s2 a1 a2 : Nat), equal_sides t s1 s2 → equal_angles t a1 a2) ∧
  (∀ (t1 t2 : Triangle), congruent t1 t2 → corresponding_sides_equal t1 t2) ∧
  (∃ (a1 a2 : Angle) (l1 l2 : Line), supplementary a1 a2 ∧ a1 ≠ a2 ∧ parallel l1 l2) ∧
  (∀ (a1 a2 a3 a4 : Angle), a1 = a2 → supplement_of a1 a3 → supplement_of a2 a4 → a3 = a4) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_theorem_congruent_triangles_theorem_supplementary_angles_not_always_equal_supplements_of_equal_angles_are_equal_proposition_c_is_false_l2072_207229


namespace NUMINAMATH_CALUDE_jenga_remaining_blocks_l2072_207261

/-- Represents a Jenga game state -/
structure JengaGame where
  initialBlocks : ℕ
  players : ℕ
  completeRounds : ℕ
  extraBlocksRemoved : ℕ

/-- Calculates the number of blocks remaining before the last player's turn -/
def remainingBlocks (game : JengaGame) : ℕ :=
  game.initialBlocks - (game.players * game.completeRounds + game.extraBlocksRemoved)

/-- Theorem stating the number of blocks remaining in the specific Jenga game scenario -/
theorem jenga_remaining_blocks :
  let game : JengaGame := {
    initialBlocks := 54,
    players := 5,
    completeRounds := 5,
    extraBlocksRemoved := 1
  }
  remainingBlocks game = 28 := by sorry

end NUMINAMATH_CALUDE_jenga_remaining_blocks_l2072_207261


namespace NUMINAMATH_CALUDE_unique_integer_solution_l2072_207268

theorem unique_integer_solution : ∃! x : ℤ, 
  (((2 * x > 70) ∧ (x < 100)) ∨ 
   ((2 * x > 70) ∧ (4 * x > 25)) ∨ 
   ((2 * x > 70) ∧ (x > 5)) ∨ 
   ((x < 100) ∧ (4 * x > 25)) ∨ 
   ((x < 100) ∧ (x > 5)) ∨ 
   ((4 * x > 25) ∧ (x > 5))) ∧
  (((2 * x ≤ 70) ∧ (x ≥ 100)) ∨ 
   ((2 * x ≤ 70) ∧ (4 * x ≤ 25)) ∨ 
   ((2 * x ≤ 70) ∧ (x ≤ 5)) ∨ 
   ((x ≥ 100) ∧ (4 * x ≤ 25)) ∨ 
   ((x ≥ 100) ∧ (x ≤ 5)) ∨ 
   ((4 * x ≤ 25) ∧ (x ≤ 5))) ∧
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l2072_207268


namespace NUMINAMATH_CALUDE_olga_sons_daughters_l2072_207215

/-- Represents the family structure of Grandma Olga -/
structure OlgaFamily where
  daughters : Nat
  sons : Nat
  grandchildren : Nat
  daughters_sons : Nat
  sons_daughters : Nat

/-- The theorem stating the number of daughters each of Grandma Olga's sons has -/
theorem olga_sons_daughters (family : OlgaFamily) :
  family.daughters = 3 →
  family.sons = 3 →
  family.daughters_sons = 6 →
  family.grandchildren = 33 →
  family.sons_daughters = 5 := by
  sorry

end NUMINAMATH_CALUDE_olga_sons_daughters_l2072_207215


namespace NUMINAMATH_CALUDE_new_circle_externally_tangent_l2072_207228

/-- Given circle equation -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- Center of the new circle -/
def new_center : ℝ × ℝ := (2, -2)

/-- Equation of the new circle -/
def new_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 2)^2 = 9

/-- Theorem stating that the new circle is externally tangent to the given circle -/
theorem new_circle_externally_tangent :
  ∃ (x y : ℝ), given_circle x y ∧ new_circle x y ∧
  (∀ (x' y' : ℝ), given_circle x' y' ∧ new_circle x' y' → (x, y) = (x', y')) :=
sorry

end NUMINAMATH_CALUDE_new_circle_externally_tangent_l2072_207228


namespace NUMINAMATH_CALUDE_gcd_12347_9876_l2072_207226

theorem gcd_12347_9876 : Nat.gcd 12347 9876 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12347_9876_l2072_207226


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2072_207204

theorem arithmetic_computation : (-5 * 3) - (7 * -2) + (-4 * -6) = 23 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2072_207204


namespace NUMINAMATH_CALUDE_union_covers_reals_implies_a_leq_neg_one_l2072_207269

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < 4}

-- State the theorem
theorem union_covers_reals_implies_a_leq_neg_one (a : ℝ) :
  A ∪ B a = Set.univ → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_implies_a_leq_neg_one_l2072_207269


namespace NUMINAMATH_CALUDE_age_difference_l2072_207218

theorem age_difference (a b c : ℕ) : 
  b = 12 →
  b = 2 * c →
  a + b + c = 32 →
  a = b + 2 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l2072_207218


namespace NUMINAMATH_CALUDE_portias_high_school_students_portias_high_school_students_proof_l2072_207208

theorem portias_high_school_students : ℕ → ℕ → Prop :=
  fun (portia_students lara_students : ℕ) =>
    (portia_students = 3 * lara_students) →
    (portia_students + lara_students = 2600) →
    (portia_students = 1950)

-- Proof
theorem portias_high_school_students_proof : 
  ∃ (portia_students lara_students : ℕ), 
    portias_high_school_students portia_students lara_students :=
by
  sorry

end NUMINAMATH_CALUDE_portias_high_school_students_portias_high_school_students_proof_l2072_207208


namespace NUMINAMATH_CALUDE_equation_solution_l2072_207240

theorem equation_solution : ∃ x : ℝ, (x + 2) / (2 * x - 1) = 1 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2072_207240


namespace NUMINAMATH_CALUDE_division_problem_l2072_207243

theorem division_problem (x y z total : ℚ) : 
  x / y = 5 / 7 →
  x / z = 5 / 11 →
  y = 150 →
  total = x + y + z →
  total = 493 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2072_207243


namespace NUMINAMATH_CALUDE_log_inequality_l2072_207256

/-- Given 0 < a < b < 1 < c, prove that log_b(c) < log_a(c) < a^c -/
theorem log_inequality (a b c : ℝ) (ha : 0 < a) (hab : a < b) (hb1 : b < 1) (hc : 1 < c) :
  Real.log c / Real.log b < Real.log c / Real.log a ∧ Real.log c / Real.log a < a^c := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2072_207256


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2072_207220

theorem polynomial_simplification (x : ℝ) :
  (2*x^6 + 3*x^5 + x^4 + 3*x^3 + 2*x + 15) - (x^6 + 4*x^5 + 2*x^3 - x^2 + 5) =
  x^6 - x^5 + x^4 + x^3 + x^2 + 2*x + 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2072_207220


namespace NUMINAMATH_CALUDE_forty_percent_of_number_l2072_207235

theorem forty_percent_of_number (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 15 → (40/100 : ℝ) * N = 180 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_number_l2072_207235


namespace NUMINAMATH_CALUDE_obtuse_triangle_partition_l2072_207249

/-- A triple of positive integers forming an obtuse triangle -/
structure ObtuseTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  h1 : a < b
  h2 : b < c
  h3 : a + b > c
  h4 : a * a + b * b < c * c

/-- The set of integers from 2 to 3n+1 -/
def triangleSet (n : ℕ+) : Set ℕ+ :=
  {k | 2 ≤ k ∧ k ≤ 3*n+1}

/-- A partition of the triangle set into n obtuse triples -/
def ObtusePartition (n : ℕ+) : Type :=
  { partition : Finset (Finset ℕ+) //
    partition.card = n ∧
    (∀ s ∈ partition, ∃ t : ObtuseTriple, (↑s : Set ℕ+) = {t.a, t.b, t.c}) ∧
    (⋃ (s ∈ partition), (↑s : Set ℕ+)) = triangleSet n }

/-- The main theorem -/
theorem obtuse_triangle_partition (n : ℕ+) :
  ∃ p : ObtusePartition n, True := by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_partition_l2072_207249


namespace NUMINAMATH_CALUDE_sales_price_calculation_l2072_207251

theorem sales_price_calculation (C S : ℝ) : 
  (1.20 * C = 24) →  -- Gross profit is $24
  (S = C + 1.20 * C) →  -- Sales price is cost plus gross profit
  (S = 44) :=  -- Prove that sales price is $44
by
  sorry

end NUMINAMATH_CALUDE_sales_price_calculation_l2072_207251


namespace NUMINAMATH_CALUDE_quadratic_function_m_value_l2072_207292

/-- A quadratic function of the form y = 3x^2 + 2(m-1)x + n -/
def quadratic_function (m n : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * (m - 1) * x + n

/-- The derivative of the quadratic function -/
def quadratic_derivative (m : ℝ) (x : ℝ) : ℝ := 6 * x + 2 * (m - 1)

theorem quadratic_function_m_value (m n : ℝ) :
  (∀ x < 1, quadratic_derivative m x < 0) →
  (∀ x ≥ 1, quadratic_derivative m x ≥ 0) →
  m = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_value_l2072_207292


namespace NUMINAMATH_CALUDE_triangle_side_sum_l2072_207245

/-- Given real numbers x, y, and z, if 1/|x^2+2yz|, 1/|y^2+2zx|, and 1/|z^2+2xy| 
    form the sides of a non-degenerate triangle, then xy + yz + zx = 0 -/
theorem triangle_side_sum (x y z : ℝ) 
  (h1 : 1 / |x^2 + 2*y*z| + 1 / |y^2 + 2*z*x| > 1 / |z^2 + 2*x*y|)
  (h2 : 1 / |y^2 + 2*z*x| + 1 / |z^2 + 2*x*y| > 1 / |x^2 + 2*y*z|)
  (h3 : 1 / |z^2 + 2*x*y| + 1 / |x^2 + 2*y*z| > 1 / |y^2 + 2*z*x|)
  (h4 : |x^2 + 2*y*z| ≠ 0)
  (h5 : |y^2 + 2*z*x| ≠ 0)
  (h6 : |z^2 + 2*x*y| ≠ 0) :
  x*y + y*z + z*x = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l2072_207245


namespace NUMINAMATH_CALUDE_brothers_puzzle_l2072_207214

-- Define the possible identities
inductive Identity : Type
| Tweedledee : Identity
| Tweedledum : Identity

-- Define the days of the week
inductive DayOfWeek : Type
| Sunday : DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek

-- Define the brothers
structure Brother :=
(identity : Identity)

-- Define the scenario
structure Scenario :=
(brother1 : Brother)
(brother2 : Brother)
(day : DayOfWeek)

-- Define the statements of the brothers
def statement1 (s : Scenario) : Prop :=
  s.brother1.identity = Identity.Tweedledee → s.brother2.identity = Identity.Tweedledum

def statement2 (s : Scenario) : Prop :=
  s.brother2.identity = Identity.Tweedledum → s.brother1.identity = Identity.Tweedledee

-- Theorem: The scenario must be on Sunday and identities cannot be determined
theorem brothers_puzzle (s : Scenario) :
  (statement1 s ∧ statement2 s) →
  (s.day = DayOfWeek.Sunday ∧
   ¬(s.brother1.identity ≠ s.brother2.identity)) :=
by sorry

end NUMINAMATH_CALUDE_brothers_puzzle_l2072_207214


namespace NUMINAMATH_CALUDE_class_exercise_result_l2072_207238

theorem class_exercise_result (x : ℝ) : 2 * ((2 * (3 * x + 2) + 3) - 2) = 2 * (2 * (3 * x + 2) + 1) := by
  sorry

end NUMINAMATH_CALUDE_class_exercise_result_l2072_207238


namespace NUMINAMATH_CALUDE_candy_bar_division_l2072_207234

theorem candy_bar_division (total_candy : ℝ) (num_bags : ℕ) 
  (h1 : total_candy = 15.5) 
  (h2 : num_bags = 5) : 
  total_candy / (num_bags : ℝ) = 3.1 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_division_l2072_207234


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l2072_207274

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (team_avg_age : ℝ) 
  (captain_age_diff : ℝ) 
  (remaining_avg_age_diff : ℝ) : 
  team_size = 15 → 
  team_avg_age = 28 → 
  captain_age_diff = 4 → 
  remaining_avg_age_diff = 2 → 
  (team_size : ℝ) * team_avg_age = 
    ((team_size - 2) : ℝ) * (team_avg_age - remaining_avg_age_diff) + 
    (team_avg_age + captain_age_diff) + 
    (team_size * team_avg_age - ((team_size - 2) : ℝ) * (team_avg_age - remaining_avg_age_diff) - (team_avg_age + captain_age_diff)) →
  team_avg_age = 28 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l2072_207274


namespace NUMINAMATH_CALUDE_line_segment_proportion_l2072_207298

theorem line_segment_proportion (a b c d : ℝ) :
  a = 1 →
  b = 2 →
  c = 3 →
  (a / b = c / d) →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_line_segment_proportion_l2072_207298


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposed_l2072_207210

/-- Represents a card color -/
inductive CardColor
| Red
| White
| Black

/-- Represents a person -/
inductive Person
| A
| B
| C

/-- Represents the distribution of cards to people -/
def Distribution := Person → CardColor

/-- The event "A receives the red card" -/
def event_A_red (d : Distribution) : Prop := d Person.A = CardColor.Red

/-- The event "B receives the red card" -/
def event_B_red (d : Distribution) : Prop := d Person.B = CardColor.Red

/-- The set of all possible distributions -/
def all_distributions : Set Distribution :=
  {d | ∀ c : CardColor, ∃! p : Person, d p = c}

theorem events_mutually_exclusive_but_not_opposed :
  (∀ d ∈ all_distributions, ¬(event_A_red d ∧ event_B_red d)) ∧
  (∃ d ∈ all_distributions, ¬event_A_red d ∧ ¬event_B_red d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposed_l2072_207210


namespace NUMINAMATH_CALUDE_triangle_angle_A_l2072_207263

theorem triangle_angle_A (a b : ℝ) (B : Real) (A : Real) : 
  a = 4 → 
  b = 4 * Real.sqrt 3 → 
  B = 60 * π / 180 →
  (a / Real.sin A = b / Real.sin B) →
  A = 30 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l2072_207263

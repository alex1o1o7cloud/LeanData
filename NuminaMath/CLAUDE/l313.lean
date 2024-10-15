import Mathlib

namespace NUMINAMATH_CALUDE_calculation_proof_l313_31363

theorem calculation_proof : 5 * (-2) + Real.pi ^ 0 + (-1) ^ 2023 - 2 ^ 3 = -18 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l313_31363


namespace NUMINAMATH_CALUDE_orange_apple_difference_l313_31328

def apples : ℕ := 14
def dozen : ℕ := 12
def oranges : ℕ := 2 * dozen

theorem orange_apple_difference : oranges - apples = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_apple_difference_l313_31328


namespace NUMINAMATH_CALUDE_complex_set_is_line_l313_31334

/-- The set of complex numbers z such that (3+4i)z is real forms a line in the complex plane. -/
theorem complex_set_is_line : 
  let S : Set ℂ := {z | ∃ r : ℝ, (3 + 4*I) * z = r}
  ∃ a b : ℝ, S = {z | z.re = a * z.im + b} :=
sorry

end NUMINAMATH_CALUDE_complex_set_is_line_l313_31334


namespace NUMINAMATH_CALUDE_fred_card_spending_l313_31354

-- Define the costs of each type of card
def football_pack_cost : ℝ := 2.73
def pokemon_pack_cost : ℝ := 4.01
def baseball_deck_cost : ℝ := 8.95

-- Define the number of packs/decks bought
def football_packs : ℕ := 2
def pokemon_packs : ℕ := 1
def baseball_decks : ℕ := 1

-- Define the total cost function
def total_cost : ℝ := 
  (football_pack_cost * football_packs) + 
  (pokemon_pack_cost * pokemon_packs) + 
  (baseball_deck_cost * baseball_decks)

-- Theorem statement
theorem fred_card_spending : total_cost = 18.42 := by
  sorry

end NUMINAMATH_CALUDE_fred_card_spending_l313_31354


namespace NUMINAMATH_CALUDE_cake_mix_tray_difference_l313_31392

theorem cake_mix_tray_difference :
  ∀ (tray1_capacity tray2_capacity : ℕ),
    tray1_capacity + tray2_capacity = 500 →
    tray2_capacity = 240 →
    tray1_capacity > tray2_capacity →
    tray1_capacity - tray2_capacity = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_mix_tray_difference_l313_31392


namespace NUMINAMATH_CALUDE_smallest_prime_factors_difference_l313_31339

theorem smallest_prime_factors_difference (n : Nat) (h : n = 296045) :
  ∃ (p q : Nat), Prime p ∧ Prime q ∧ p < q ∧ p ∣ n ∧ q ∣ n ∧
  (∀ r : Nat, Prime r → r ∣ n → p ≤ r) ∧
  (∀ r : Nat, Prime r → r ∣ n → r ≠ p → q ≤ r) ∧
  q - p = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factors_difference_l313_31339


namespace NUMINAMATH_CALUDE_inequality_proof_l313_31393

theorem inequality_proof (a b : ℝ) (m : ℤ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ m + (1 + b / a) ^ m ≥ 2^(m + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l313_31393


namespace NUMINAMATH_CALUDE_circplus_square_sum_diff_l313_31350

/-- Custom operation ⊕ for real numbers -/
def circplus (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating the equality for (x+y)^2 ⊕ (x-y)^2 -/
theorem circplus_square_sum_diff (x y : ℝ) : 
  circplus ((x + y)^2) ((x - y)^2) = 4 * (x^2 + y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_circplus_square_sum_diff_l313_31350


namespace NUMINAMATH_CALUDE_sphere_surface_area_l313_31342

theorem sphere_surface_area (V : ℝ) (r : ℝ) (A : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  A = 4 * Real.pi * r^2 → 
  A = 36 * Real.pi * 2^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l313_31342


namespace NUMINAMATH_CALUDE_complex_division_simplification_l313_31369

theorem complex_division_simplification :
  (2 - I) / (1 - I) = 3/2 + I/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l313_31369


namespace NUMINAMATH_CALUDE_geometry_statements_l313_31360

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def parallel (l1 l2 : Line) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def parallel_plane (p1 p2 : Plane) : Prop := sorry
def perpendicular_plane (p1 p2 : Plane) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry

theorem geometry_statements 
  (a b : Line) (α β : Plane) : 
  -- Statement 2
  (perpendicular a b ∧ perpendicular a α ∧ ¬contained_in b α → parallel b α) ∧
  -- Statement 3
  (perpendicular_plane α β ∧ perpendicular a α ∧ perpendicular b β → perpendicular a b) ∧
  -- Statement 1 (not necessarily true)
  ¬(parallel a b ∧ contained_in b α → parallel a α ∨ contained_in a α) ∧
  -- Statement 4 (not necessarily true)
  ¬(skew a b ∧ contained_in a α ∧ contained_in b β → parallel_plane α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_statements_l313_31360


namespace NUMINAMATH_CALUDE_probability_two_females_l313_31346

/-- The probability of selecting two females from a group of contestants -/
theorem probability_two_females (total : ℕ) (females : ℕ) (males : ℕ) :
  total = females + males →
  females = 5 →
  males = 3 →
  (Nat.choose females 2 : ℚ) / (Nat.choose total 2 : ℚ) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_females_l313_31346


namespace NUMINAMATH_CALUDE_incorrect_proposition_l313_31356

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (different : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem incorrect_proposition
  (α β : Plane) (m n : Line)
  (h1 : different α β)
  (h2 : m ≠ n)
  : ¬(parallel_lines m n ∧ intersect α β m →
      parallel_line_plane n α ∧ parallel_line_plane n β) :=
sorry

end NUMINAMATH_CALUDE_incorrect_proposition_l313_31356


namespace NUMINAMATH_CALUDE_weekend_ice_cream_total_l313_31338

/-- The total amount of ice cream consumed by 4 roommates over a weekend -/
def weekend_ice_cream_consumption (friday_total : ℝ) : ℝ :=
  let saturday_total := friday_total - (4 * 0.25)
  let sunday_total := 2 * saturday_total
  friday_total + saturday_total + sunday_total

/-- Theorem stating that the total ice cream consumption over the weekend is 10 pints -/
theorem weekend_ice_cream_total :
  weekend_ice_cream_consumption 3.25 = 10 := by
  sorry

#eval weekend_ice_cream_consumption 3.25

end NUMINAMATH_CALUDE_weekend_ice_cream_total_l313_31338


namespace NUMINAMATH_CALUDE_circle_circumference_l313_31390

/-- The circumference of a circle with diameter 2 yards is equal to π * 2 yards. -/
theorem circle_circumference (diameter : ℝ) (h : diameter = 2) :
  (diameter * π) = 2 * π := by sorry

end NUMINAMATH_CALUDE_circle_circumference_l313_31390


namespace NUMINAMATH_CALUDE_square_root_fraction_equality_l313_31323

theorem square_root_fraction_equality : 
  let x : ℝ := Real.sqrt (7 - 4 * Real.sqrt 3)
  (x^2 - 4*x + 5) / (x^2 - 4*x + 3) = 2 := by sorry

end NUMINAMATH_CALUDE_square_root_fraction_equality_l313_31323


namespace NUMINAMATH_CALUDE_equation_solutions_l313_31368

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧
    x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 2/3 ∧
    3*x₁*(x₁-1) = 2*x₁-2 ∧ 3*x₂*(x₂-1) = 2*x₂-2) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l313_31368


namespace NUMINAMATH_CALUDE_congruence_problem_l313_31307

theorem congruence_problem (x : ℤ) : 
  (5 * x + 3) % 18 = 1 → (3 * x + 8) % 18 = 14 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l313_31307


namespace NUMINAMATH_CALUDE_estevan_blanket_ratio_l313_31314

/-- The ratio of polka-dot blankets to total blankets before Estevan's birthday -/
theorem estevan_blanket_ratio :
  let total_blankets : ℕ := 24
  let new_polka_dot_blankets : ℕ := 2
  let total_polka_dot_blankets : ℕ := 10
  let initial_polka_dot_blankets : ℕ := total_polka_dot_blankets - new_polka_dot_blankets
  (initial_polka_dot_blankets : ℚ) / total_blankets = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_estevan_blanket_ratio_l313_31314


namespace NUMINAMATH_CALUDE_five_hundredth_barrel_is_four_l313_31312

/-- The labeling function for barrels based on their position in the sequence -/
def barrel_label (n : ℕ) : ℕ :=
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 4
  | 5 => 5
  | 6 => 10 - 6
  | 7 => 10 - 7
  | 0 => 10 - 8
  | _ => 0  -- This case should never occur due to properties of modulo

/-- The theorem stating that the 500th barrel is labeled 4 -/
theorem five_hundredth_barrel_is_four :
  barrel_label 500 = 4 := by
  sorry


end NUMINAMATH_CALUDE_five_hundredth_barrel_is_four_l313_31312


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l313_31349

def son_age : ℕ := 20
def man_age : ℕ := son_age + 22

def son_age_in_two_years : ℕ := son_age + 2
def man_age_in_two_years : ℕ := man_age + 2

theorem age_ratio_in_two_years :
  man_age_in_two_years / son_age_in_two_years = 2 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l313_31349


namespace NUMINAMATH_CALUDE_triangle_problem_l313_31302

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  (2 * Real.sin t.B * Real.cos t.A = Real.sin (t.A + t.C)) →  -- Given condition
  (t.BC = 2) →  -- Given condition
  (1/2 * t.AB * t.AC * Real.sin t.A = Real.sqrt 3) →  -- Area condition
  (t.A = Real.pi / 3 ∧ t.AB = 2) := by  -- Conclusion
sorry  -- Proof is omitted

end NUMINAMATH_CALUDE_triangle_problem_l313_31302


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_parallel_l313_31391

variable {V : Type*} [NormedAddCommGroup V] [Module ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem sufficient_not_necessary_parallel (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b : V, a + b = 0 → parallel a b) ∧
  (∃ a b : V, parallel a b ∧ a + b ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_parallel_l313_31391


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l313_31375

/-- Given a geometric sequence where the third term is 24 and the fourth term is 36, 
    the first term of the sequence is 32/3. -/
theorem geometric_sequence_first_term (a : ℚ) (r : ℚ) : 
  a * r^2 = 24 ∧ a * r^3 = 36 → a = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l313_31375


namespace NUMINAMATH_CALUDE_triangle_is_obtuse_l313_31311

theorem triangle_is_obtuse (a b c : ℝ) (h_a : a = 5) (h_b : b = 6) (h_c : c = 8) :
  c^2 > a^2 + b^2 := by
  sorry

#check triangle_is_obtuse

end NUMINAMATH_CALUDE_triangle_is_obtuse_l313_31311


namespace NUMINAMATH_CALUDE_smallest_k_for_720k_square_and_cube_l313_31394

theorem smallest_k_for_720k_square_and_cube :
  (∀ n : ℕ+, n < 1012500 → ¬(∃ a b : ℕ+, 720 * n = a^2 ∧ 720 * n = b^3)) ∧
  (∃ a b : ℕ+, 720 * 1012500 = a^2 ∧ 720 * 1012500 = b^3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_720k_square_and_cube_l313_31394


namespace NUMINAMATH_CALUDE_thief_hiding_speeds_l313_31382

/-- Configuration of roads, houses, and police movement --/
structure Configuration where
  road_distance : ℝ
  house_size : ℝ
  house_spacing : ℝ
  house_road_distance : ℝ
  police_speed : ℝ
  police_interval : ℝ

/-- Thief's movement relative to police --/
inductive ThiefMovement
  | Opposite
  | Same

/-- Proposition that the thief can stay hidden --/
def can_stay_hidden (config : Configuration) (thief_speed : ℝ) (direction : ThiefMovement) : Prop :=
  match direction with
  | ThiefMovement.Opposite => thief_speed = 2 * config.police_speed
  | ThiefMovement.Same => thief_speed = config.police_speed / 2

/-- Theorem stating the only two viable speeds for the thief --/
theorem thief_hiding_speeds (config : Configuration) 
  (h1 : config.road_distance = 30)
  (h2 : config.house_size = 10)
  (h3 : config.house_spacing = 20)
  (h4 : config.house_road_distance = 10)
  (h5 : config.police_interval = 90)
  (thief_speed : ℝ)
  (direction : ThiefMovement) :
  can_stay_hidden config thief_speed direction ↔ 
    (thief_speed = 2 * config.police_speed ∧ direction = ThiefMovement.Opposite) ∨
    (thief_speed = config.police_speed / 2 ∧ direction = ThiefMovement.Same) :=
  sorry

end NUMINAMATH_CALUDE_thief_hiding_speeds_l313_31382


namespace NUMINAMATH_CALUDE_base3_subtraction_l313_31325

/-- Represents a number in base 3 as a list of digits (least significant first) -/
def Base3 : Type := List Nat

/-- Converts a base 3 number to a natural number -/
def to_nat (b : Base3) : Nat :=
  b.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Adds two base 3 numbers -/
def add (a b : Base3) : Base3 :=
  sorry

/-- Subtracts two base 3 numbers -/
def sub (a b : Base3) : Base3 :=
  sorry

theorem base3_subtraction :
  let a : Base3 := [0, 1, 0]  -- 10₃
  let b : Base3 := [1, 0, 1, 1]  -- 1101₃
  let c : Base3 := [2, 0, 1, 2]  -- 2102₃
  let d : Base3 := [2, 1, 2]  -- 212₃
  let result : Base3 := [1, 0, 1, 1]  -- 1101₃
  sub (add (add a b) c) d = result := by
  sorry

end NUMINAMATH_CALUDE_base3_subtraction_l313_31325


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l313_31383

/-- An isosceles triangle with two sides of length 5 and one side of length 2 has perimeter 12 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 5 → b = 5 → c = 2 →
  (a = b ∨ a = c ∨ b = c) →  -- isosceles condition
  a + b + c = 12 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l313_31383


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l313_31370

theorem three_digit_number_problem : ∃ (a b c : ℕ), 
  (8 * a + 5 * b + c = 100) ∧ 
  (a + b + c = 20) ∧ 
  (a < 10) ∧ (b < 10) ∧ (c < 10) ∧
  (a * 100 + b * 10 + c = 866) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l313_31370


namespace NUMINAMATH_CALUDE_fish_and_shrimp_prices_l313_31336

/-- The regular price of fish per pound -/
def regular_fish_price : ℝ := 10

/-- The discounted price of fish per quarter-pound package -/
def discounted_fish_price : ℝ := 1.5

/-- The price of shrimp per half-pound -/
def shrimp_price : ℝ := 5

/-- The discount rate on fish -/
def discount_rate : ℝ := 0.6

theorem fish_and_shrimp_prices :
  (regular_fish_price * (1 - discount_rate) / 4 = discounted_fish_price) ∧
  (regular_fish_price = 2 * shrimp_price) :=
sorry

end NUMINAMATH_CALUDE_fish_and_shrimp_prices_l313_31336


namespace NUMINAMATH_CALUDE_carries_shopping_money_l313_31395

theorem carries_shopping_money (initial_amount : ℕ) (sweater_cost : ℕ) (tshirt_cost : ℕ) (shoes_cost : ℕ) 
  (h1 : initial_amount = 91)
  (h2 : sweater_cost = 24)
  (h3 : tshirt_cost = 6)
  (h4 : shoes_cost = 11) :
  initial_amount - (sweater_cost + tshirt_cost + shoes_cost) = 50 := by
  sorry

end NUMINAMATH_CALUDE_carries_shopping_money_l313_31395


namespace NUMINAMATH_CALUDE_one_fourth_of_six_point_three_l313_31335

theorem one_fourth_of_six_point_three (x : ℚ) : x = 6.3 / 4 → x = 63 / 40 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_six_point_three_l313_31335


namespace NUMINAMATH_CALUDE_f_inequality_l313_31386

noncomputable def f (x : ℝ) : ℝ := Real.log (abs (x - 1))

theorem f_inequality : f (1 / Real.exp 1) < f 0 ∧ f 0 < f (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l313_31386


namespace NUMINAMATH_CALUDE_plastic_bag_estimate_l313_31351

def plastic_bag_data : List Nat := [33, 25, 28, 26, 25, 31]
def total_students : Nat := 45

theorem plastic_bag_estimate :
  let average := (plastic_bag_data.sum / plastic_bag_data.length)
  average * total_students = 1260 := by
  sorry

end NUMINAMATH_CALUDE_plastic_bag_estimate_l313_31351


namespace NUMINAMATH_CALUDE_computer_price_increase_l313_31364

theorem computer_price_increase (x : ℝ) (h : x + 0.3 * x = 351) : x + 351 = 621 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l313_31364


namespace NUMINAMATH_CALUDE_water_left_l313_31359

theorem water_left (initial : ℚ) (used : ℚ) (left : ℚ) : 
  initial = 3 → used = 9/4 → left = initial - used → left = 3/4 := by sorry

end NUMINAMATH_CALUDE_water_left_l313_31359


namespace NUMINAMATH_CALUDE_window_purchase_savings_l313_31321

/-- Calculates the cost of windows given the number of windows and the discount rule -/
def calculateCost (windowCount : ℕ) (windowPrice : ℕ) : ℕ :=
  (windowCount - windowCount / 3) * windowPrice

/-- Represents the window purchase scenario -/
theorem window_purchase_savings
  (windowPrice : ℕ)
  (daveWindowCount : ℕ)
  (dougWindowCount : ℕ)
  (h1 : windowPrice = 100)
  (h2 : daveWindowCount = 10)
  (h3 : dougWindowCount = 12) :
  calculateCost (daveWindowCount + dougWindowCount) windowPrice =
  calculateCost daveWindowCount windowPrice + calculateCost dougWindowCount windowPrice :=
by sorry

#eval calculateCost 22 100 -- Joint purchase
#eval calculateCost 10 100 + calculateCost 12 100 -- Separate purchases

end NUMINAMATH_CALUDE_window_purchase_savings_l313_31321


namespace NUMINAMATH_CALUDE_locus_of_G_is_parabola_l313_31389

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a right triangle -/
structure RightTriangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Finds the intersection point of two lines -/
noncomputable def lineIntersection (l1 l2 : Line) : Point :=
  { x := (l1.b * l2.c - l2.b * l1.c) / (l1.a * l2.b - l2.a * l1.b),
    y := (l2.a * l1.c - l1.a * l2.c) / (l1.a * l2.b - l2.a * l1.b) }

/-- Theorem: Locus of point G forms a parabola -/
theorem locus_of_G_is_parabola (abc : RightTriangle) (d : Point) (ad ce ab : Line) :
  ∀ (e : Point), pointOnLine e ad →
  let f := lineIntersection ce ab
  let bc := Line.mk (abc.B.y - abc.C.y) (abc.C.x - abc.B.x) (abc.B.x * abc.C.y - abc.C.x * abc.B.y)
  let perpF := Line.mk bc.b (-bc.a) (-bc.b * f.x + bc.a * f.y)
  let be := Line.mk (e.y - abc.B.y) (abc.B.x - e.x) (e.x * abc.B.y - abc.B.x * e.y)
  let g := lineIntersection perpF be
  ∃ (a b : ℝ), g.y = (a / (b^2)) * (g.x - b)^2 := by
    sorry

end NUMINAMATH_CALUDE_locus_of_G_is_parabola_l313_31389


namespace NUMINAMATH_CALUDE_r_equals_1464_when_n_is_1_l313_31304

/-- Given the conditions for r and s, prove that r equals 1464 when n is 1 -/
theorem r_equals_1464_when_n_is_1 (n : ℕ) (s r : ℕ) 
  (h1 : s = 4^n + 2) 
  (h2 : r = 2 * 3^s + s) 
  (h3 : n = 1) : 
  r = 1464 := by
  sorry

end NUMINAMATH_CALUDE_r_equals_1464_when_n_is_1_l313_31304


namespace NUMINAMATH_CALUDE_prove_a_equals_two_l313_31373

/-- Given a > 1 and f(x) = a^x + 1, prove that a = 2 if f(2) - f(1) = 2 -/
theorem prove_a_equals_two (a : ℝ) (h1 : a > 1) : 
  (fun x => a^x + 1) 2 - (fun x => a^x + 1) 1 = 2 → a = 2 := by
sorry

end NUMINAMATH_CALUDE_prove_a_equals_two_l313_31373


namespace NUMINAMATH_CALUDE_equation_solution_l313_31352

theorem equation_solution (x : ℚ) : 
  (30 * x^2 + 17 = 47 * x - 6) →
  (x = 3/5 ∨ x = 23/36) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l313_31352


namespace NUMINAMATH_CALUDE_min_value_case1_min_value_case2_l313_31377

/-- The function f(x) defined as x^2 + |x-a| + 1 -/
def f (a x : ℝ) : ℝ := x^2 + |x-a| + 1

/-- The minimum value of f(x) when a ≤ -1/2 and x ≥ a -/
theorem min_value_case1 (a : ℝ) (h : a ≤ -1/2) :
  ∀ x ≥ a, f a x ≥ 3/4 - a :=
sorry

/-- The minimum value of f(x) when a > -1/2 and x ≥ a -/
theorem min_value_case2 (a : ℝ) (h : a > -1/2) :
  ∀ x ≥ a, f a x ≥ a^2 + 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_case1_min_value_case2_l313_31377


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l313_31361

/-- The decimal representation of 0.7888... -/
def repeating_decimal : ℚ := 0.7 + (8 / 9) / 10

theorem repeating_decimal_as_fraction :
  repeating_decimal = 71 / 90 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l313_31361


namespace NUMINAMATH_CALUDE_pentagon_diagonal_equality_l313_31313

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A pentagon defined by five points -/
structure Pentagon :=
  (A B C D E : Point)

/-- Checks if a pentagon is convex -/
def is_convex (p : Pentagon) : Prop := sorry

/-- Checks if a line segment bisects an angle -/
def bisects_angle (P Q R S : Point) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def intersection (P Q R S : Point) : Point := sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ := sorry

theorem pentagon_diagonal_equality (p : Pentagon) 
  (h_convex : is_convex p)
  (h_bd_bisect1 : bisects_angle p.C p.B p.E p.D)
  (h_bd_bisect2 : bisects_angle p.C p.D p.A p.B)
  (h_ce_bisect1 : bisects_angle p.A p.C p.D p.E)
  (h_ce_bisect2 : bisects_angle p.B p.E p.D p.C)
  (K : Point) (h_K : K = intersection p.B p.E p.A p.C)
  (L : Point) (h_L : L = intersection p.B p.E p.A p.D) :
  distance p.C K = distance p.D L := by sorry

end NUMINAMATH_CALUDE_pentagon_diagonal_equality_l313_31313


namespace NUMINAMATH_CALUDE_library_loans_l313_31348

theorem library_loans (init_a init_b current_a current_b : ℕ) 
  (return_rate_a return_rate_b : ℚ) : 
  init_a = 75 → 
  init_b = 100 → 
  current_a = 54 → 
  current_b = 82 → 
  return_rate_a = 65 / 100 → 
  return_rate_b = 1 / 2 → 
  ∃ (loaned_a loaned_b : ℕ), 
    loaned_a + loaned_b = 96 ∧ 
    (init_a - current_a : ℚ) = (1 - return_rate_a) * loaned_a ∧
    (init_b - current_b : ℚ) = (1 - return_rate_b) * loaned_b :=
by sorry

end NUMINAMATH_CALUDE_library_loans_l313_31348


namespace NUMINAMATH_CALUDE_jade_transactions_l313_31379

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 18 →
  jade = 84 :=
by sorry

end NUMINAMATH_CALUDE_jade_transactions_l313_31379


namespace NUMINAMATH_CALUDE_fixed_point_power_function_l313_31380

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2 - x) - 3/4

theorem fixed_point_power_function 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (α : ℝ) 
  (h3 : ∀ x : ℝ, (x : ℝ)^α = x^α) -- To ensure g is a power function
  (h4 : (2 : ℝ)^α = 1/4) -- g passes through (2, 1/4)
  : (1/2 : ℝ)^α = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_power_function_l313_31380


namespace NUMINAMATH_CALUDE_opposite_of_2023_l313_31333

theorem opposite_of_2023 : Int.neg 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l313_31333


namespace NUMINAMATH_CALUDE_derivative_of_y_l313_31378

noncomputable def y (x : ℝ) : ℝ :=
  (1 / Real.sqrt 8) * Real.log ((4 + Real.sqrt 8 * Real.tanh (x / 2)) / (4 - Real.sqrt 8 * Real.tanh (x / 2)))

theorem derivative_of_y (x : ℝ) :
  deriv y x = 1 / (2 * (Real.cosh (x / 2) ^ 2 + 1)) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_y_l313_31378


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l313_31303

/-- The number of bacteria after a given number of tripling events -/
def bacteria_count (initial_count : ℕ) (tripling_events : ℕ) : ℕ :=
  initial_count * (3 ^ tripling_events)

/-- The number of tripling events in a given number of seconds -/
def tripling_events (seconds : ℕ) : ℕ :=
  seconds / 20

theorem initial_bacteria_count :
  ∃ (initial_count : ℕ),
    bacteria_count initial_count (tripling_events 180) = 275562 ∧
    initial_count = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l313_31303


namespace NUMINAMATH_CALUDE_problem_solution_l313_31326

theorem problem_solution (a b : ℚ) :
  (∀ x y : ℚ, y = a + b / x) →
  (2 = a + b / (-2)) →
  (3 = a + b / (-6)) →
  a + b = 13/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l313_31326


namespace NUMINAMATH_CALUDE_inequality_solution_set_l313_31317

theorem inequality_solution_set : 
  {x : ℝ | -2 ≤ x ∧ x ≤ 1} = {x : ℝ | 2 - x - x^2 ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l313_31317


namespace NUMINAMATH_CALUDE_people_per_column_l313_31324

theorem people_per_column (total_people : ℕ) 
  (h1 : total_people / 60 = 8) 
  (h2 : total_people % 16 = 0) : 
  total_people / 16 = 30 := by
sorry

end NUMINAMATH_CALUDE_people_per_column_l313_31324


namespace NUMINAMATH_CALUDE_prob_two_black_one_red_standard_deck_l313_31388

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    red_cards := 26,
    black_cards := 26 }

/-- The probability of drawing two black cards followed by a red card -/
def prob_two_black_one_red (d : Deck) : ℚ :=
  (d.black_cards * (d.black_cards - 1) * d.red_cards) / 
  (d.total_cards * (d.total_cards - 1) * (d.total_cards - 2))

/-- Theorem stating the probability for a standard deck -/
theorem prob_two_black_one_red_standard_deck :
  prob_two_black_one_red standard_deck = 13 / 102 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_black_one_red_standard_deck_l313_31388


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l313_31345

/-- Given two parallel lines x - ky - k = 0 and y = k(x-1), prove that k = -1 -/
theorem parallel_lines_k_value (k : ℝ) : 
  (∀ x y : ℝ, x - k * y - k = 0 ↔ y = k * (x - 1)) → 
  (∀ x y : ℝ, x - k * y - k = 0 → ∃ c : ℝ, y = (1/k) * x + c) →
  k ≠ 0 →
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l313_31345


namespace NUMINAMATH_CALUDE_k_range_l313_31384

theorem k_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) → 
  -1 < k ∧ k ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_k_range_l313_31384


namespace NUMINAMATH_CALUDE_existence_of_specific_values_l313_31353

theorem existence_of_specific_values : ∃ (a b : ℝ), a * b = a^2 - a * b + b^2 ∧ a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_specific_values_l313_31353


namespace NUMINAMATH_CALUDE_evaluate_expression_l313_31306

theorem evaluate_expression : 3^2 * 4 * 6^3 * Nat.factorial 7 = 39191040 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l313_31306


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l313_31341

def arrange_books (geom_copies : ℕ) (alg_copies : ℕ) : ℕ :=
  let total_slots := geom_copies + alg_copies - 1
  let remaining_geom := geom_copies - 2
  (total_slots.choose remaining_geom) * 2

theorem book_arrangement_theorem :
  arrange_books 4 5 = 112 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l313_31341


namespace NUMINAMATH_CALUDE_g_of_5_l313_31372

def g (x : ℚ) : ℚ := (3 * x - 2) / (x + 2)

theorem g_of_5 : g 5 = 13 / 7 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l313_31372


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l313_31305

/-- The function f(x) = x³ - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_line_at_x_1 : 
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (2*x - y + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l313_31305


namespace NUMINAMATH_CALUDE_smallest_a_for_coeff_70_l313_31357

/-- The coefficient of x^4 in the expansion of (1-3x+ax^2)^8 -/
def coeff_x4 (a : ℝ) : ℝ := 28 * a^2 + 1512 * a + 4725

/-- The problem statement -/
theorem smallest_a_for_coeff_70 :
  ∃ a : ℝ, (∀ b : ℝ, coeff_x4 b = 70 → a ≤ b) ∧ coeff_x4 a = 70 ∧ a = -50 := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_for_coeff_70_l313_31357


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l313_31308

/-- Polar to Cartesian conversion theorem for ρ = 4cosθ -/
theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 4 * Real.cos θ) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) →
  (x - 2)^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l313_31308


namespace NUMINAMATH_CALUDE_olympic_inequalities_l313_31316

/-- Given positive real numbers a, b, c, d such that a + b + c + d = 3,
    prove the following inequalities:
    1. (1/a^2 + 1/b^2 + 1/c^2 + 1/d^2) ≤ 1/(a^2*b^2*c^2*d^2)
    2. (1/a^3 + 1/b^3 + 1/c^3 + 1/d^3) ≤ 1/(a^3*b^3*c^3*d^3) -/
theorem olympic_inequalities (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d = 3) :
  (1/a^2 + 1/b^2 + 1/c^2 + 1/d^2 ≤ 1/(a^2*b^2*c^2*d^2)) ∧
  (1/a^3 + 1/b^3 + 1/c^3 + 1/d^3 ≤ 1/(a^3*b^3*c^3*d^3)) := by
  sorry

end NUMINAMATH_CALUDE_olympic_inequalities_l313_31316


namespace NUMINAMATH_CALUDE_team_selection_count_l313_31331

/-- The number of ways to select a team of 8 students from 10 boys and 12 girls, with at least 4 girls -/
def select_team (total_boys : ℕ) (total_girls : ℕ) (team_size : ℕ) (min_girls : ℕ) : ℕ :=
  (Nat.choose total_girls min_girls * Nat.choose total_boys (team_size - min_girls)) +
  (Nat.choose total_girls (min_girls + 1) * Nat.choose total_boys (team_size - min_girls - 1)) +
  (Nat.choose total_girls (min_girls + 2) * Nat.choose total_boys (team_size - min_girls - 2)) +
  (Nat.choose total_girls (min_girls + 3) * Nat.choose total_boys (team_size - min_girls - 3)) +
  (Nat.choose total_girls (min_girls + 4))

theorem team_selection_count :
  select_team 10 12 8 4 = 245985 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_count_l313_31331


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l313_31318

/-- Given a mixture of milk and water, proves that the initial ratio is 4:1 --/
theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (final_ratio : ℝ) 
  (h1 : total_volume = 45) 
  (h2 : added_water = 11) 
  (h3 : final_ratio = 1.8) : 
  ∃ (milk water : ℝ), 
    milk + water = total_volume ∧ 
    milk / (water + added_water) = final_ratio ∧ 
    milk / water = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l313_31318


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l313_31376

theorem inequality_system_solution_set :
  let S : Set ℝ := {x | 2 * x - 4 ≤ 0 ∧ -x + 1 < 0}
  S = {x | 1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l313_31376


namespace NUMINAMATH_CALUDE_largest_divisor_of_9670_l313_31398

theorem largest_divisor_of_9670 : ∃ (d : ℕ), d > 0 ∧ d ∣ (9671 - 1) ∧ ∀ (x : ℕ), x > 0 → x ∣ (9671 - 1) → x ≤ d := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_9670_l313_31398


namespace NUMINAMATH_CALUDE_pastry_chef_eggs_l313_31367

theorem pastry_chef_eggs :
  ∃ n : ℕ,
    n > 0 ∧
    n % 43 = 0 ∧
    n % 2 = 1 ∧
    n % 3 = 1 ∧
    n % 4 = 1 ∧
    n % 5 = 1 ∧
    n % 6 = 1 ∧
    n / 43 < 9 ∧
    (∀ m : ℕ, m > 0 ∧ m < n →
      ¬(m % 43 = 0 ∧
        m % 2 = 1 ∧
        m % 3 = 1 ∧
        m % 4 = 1 ∧
        m % 5 = 1 ∧
        m % 6 = 1 ∧
        m / 43 < 9)) ∧
    n = 301 := by
  sorry

end NUMINAMATH_CALUDE_pastry_chef_eggs_l313_31367


namespace NUMINAMATH_CALUDE_new_average_age_l313_31381

def initial_people : ℕ := 6
def initial_average_age : ℚ := 25
def leaving_age : ℕ := 20
def entering_age : ℕ := 30

theorem new_average_age :
  let initial_total_age : ℚ := initial_people * initial_average_age
  let new_total_age : ℚ := initial_total_age - leaving_age + entering_age
  new_total_age / initial_people = 26.67 := by
sorry

end NUMINAMATH_CALUDE_new_average_age_l313_31381


namespace NUMINAMATH_CALUDE_kite_area_from_shifted_triangles_l313_31396

/-- The area of a kite-shaped figure formed by the intersection of two equilateral triangles -/
theorem kite_area_from_shifted_triangles (square_side : ℝ) (shift : ℝ) : 
  square_side = 4 →
  shift = 1 →
  let triangle_side := square_side
  let triangle_height := (Real.sqrt 3 / 2) * triangle_side
  let kite_base := square_side - shift
  let kite_area := kite_base * triangle_height
  kite_area = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_kite_area_from_shifted_triangles_l313_31396


namespace NUMINAMATH_CALUDE_equal_cost_at_200_unique_equal_cost_l313_31385

/-- Represents the price per book in yuan -/
def base_price : ℕ := 40

/-- Represents the discount factor for supplier A -/
def discount_a : ℚ := 9/10

/-- Represents the discount factor for supplier B on books over 100 -/
def discount_b : ℚ := 8/10

/-- Represents the threshold for supplier B's discount -/
def threshold : ℕ := 100

/-- Calculates the cost for supplier A given the number of books -/
def cost_a (n : ℕ) : ℚ := n * base_price * discount_a

/-- Calculates the cost for supplier B given the number of books -/
def cost_b (n : ℕ) : ℚ :=
  if n ≤ threshold then n * base_price
  else threshold * base_price + (n - threshold) * base_price * discount_b

/-- Theorem stating that the costs are equal when 200 books are ordered -/
theorem equal_cost_at_200 : cost_a 200 = cost_b 200 := by sorry

/-- Theorem stating that 200 is the unique number of books where costs are equal -/
theorem unique_equal_cost (n : ℕ) : cost_a n = cost_b n ↔ n = 200 := by sorry

end NUMINAMATH_CALUDE_equal_cost_at_200_unique_equal_cost_l313_31385


namespace NUMINAMATH_CALUDE_equation_solution_l313_31347

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  2 - 9 / x + 9 / x^2 = 0 → 2 / x = 2 / 3 ∨ 2 / x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l313_31347


namespace NUMINAMATH_CALUDE_roy_pens_count_l313_31365

def blue_pens : ℕ := 5

def black_pens : ℕ := 3 * blue_pens

def red_pens : ℕ := 2 * black_pens - 4

def total_pens : ℕ := blue_pens + black_pens + red_pens

theorem roy_pens_count : total_pens = 46 := by
  sorry

end NUMINAMATH_CALUDE_roy_pens_count_l313_31365


namespace NUMINAMATH_CALUDE_bus_stop_walking_time_l313_31387

theorem bus_stop_walking_time 
  (usual_speed : ℝ) 
  (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0) 
  (h3 : usual_speed * usual_time = (4/5 * usual_speed) * (usual_time + 7)) :
  usual_time = 28 := by
sorry

end NUMINAMATH_CALUDE_bus_stop_walking_time_l313_31387


namespace NUMINAMATH_CALUDE_point_c_coordinates_l313_31366

/-- Given point A, vector AB, and vector BC in a 2D Cartesian coordinate system,
    prove that the coordinates of point C are as calculated. -/
theorem point_c_coordinates (A B C : ℝ × ℝ) (AB BC : ℝ × ℝ) :
  A = (0, 1) →
  AB = (-4, -3) →
  BC = (-7, -4) →
  B = (A.1 + AB.1, A.2 + AB.2) →
  C = (B.1 + BC.1, B.2 + BC.2) →
  C = (-11, -6) := by
  sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l313_31366


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l313_31397

theorem quadratic_roots_property (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + a*x₁ + 8 = 0 ∧ 
    x₂^2 + a*x₂ + 8 = 0 ∧ 
    x₁ - 64/(17*x₂^3) = x₂ - 64/(17*x₁^3)) 
  → a = 12 ∨ a = -12 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l313_31397


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_ending_3_l313_31343

theorem smallest_four_digit_divisible_by_53_ending_3 :
  ∀ n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n % 10 = 3 →
  1113 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_ending_3_l313_31343


namespace NUMINAMATH_CALUDE_couscous_per_dish_l313_31358

theorem couscous_per_dish 
  (shipment1 : ℕ) 
  (shipment2 : ℕ) 
  (shipment3 : ℕ) 
  (num_dishes : ℕ) 
  (h1 : shipment1 = 7)
  (h2 : shipment2 = 13)
  (h3 : shipment3 = 45)
  (h4 : num_dishes = 13) :
  (shipment1 + shipment2 + shipment3) / num_dishes = 5 := by
  sorry

#check couscous_per_dish

end NUMINAMATH_CALUDE_couscous_per_dish_l313_31358


namespace NUMINAMATH_CALUDE_inscribed_trapezoids_equal_diagonals_l313_31374

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define an isosceles trapezoid
structure IsoscelesTrapezoid where
  vertices : Fin 4 → ℝ × ℝ

-- Define the property of being inscribed in a circle
def inscribed (t : IsoscelesTrapezoid) (c : Circle) : Prop := sorry

-- Define the property of sides being parallel
def parallel_sides (t1 t2 : IsoscelesTrapezoid) : Prop := sorry

-- Define the length of a diagonal
def diagonal_length (t : IsoscelesTrapezoid) : ℝ := sorry

-- Main theorem
theorem inscribed_trapezoids_equal_diagonals 
  (c : Circle) (t1 t2 : IsoscelesTrapezoid) 
  (h1 : inscribed t1 c) (h2 : inscribed t2 c) 
  (h3 : parallel_sides t1 t2) : 
  diagonal_length t1 = diagonal_length t2 := by sorry

end NUMINAMATH_CALUDE_inscribed_trapezoids_equal_diagonals_l313_31374


namespace NUMINAMATH_CALUDE_circle_center_l313_31344

/-- A circle passes through (0,1) and is tangent to y = x^3 at (1,1). Its center is (1/2, 7/6). -/
theorem circle_center (c : ℝ × ℝ) : 
  (∃ r : ℝ, (c.1 - 0)^2 + (c.2 - 1)^2 = r^2 ∧ (c.1 - 1)^2 + (c.2 - 1)^2 = r^2) →  -- circle passes through (0,1) and (1,1)
  (∃ t : ℝ, t ≠ 1 → (t^3 - 1) / (t - 1) = 3 * (t - 1)) →                        -- tangent to y = x^3 at (1,1)
  c = (1/2, 7/6) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l313_31344


namespace NUMINAMATH_CALUDE_imaginary_product_implies_a_value_l313_31399

theorem imaginary_product_implies_a_value (a : ℝ) : 
  (∃ b : ℝ, (a - Complex.I) * (3 - 2 * Complex.I) = b * Complex.I) → a = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_product_implies_a_value_l313_31399


namespace NUMINAMATH_CALUDE_work_completion_time_l313_31315

theorem work_completion_time (a_time b_time : ℝ) (ha : a_time = 10) (hb : b_time = 10) :
  1 / (1 / a_time + 1 / b_time) = 5 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l313_31315


namespace NUMINAMATH_CALUDE_digit_79_is_2_l313_31330

/-- The sequence of digits obtained by concatenating consecutive integers from 60 down to 1 -/
def digit_sequence : List Nat := sorry

/-- The 79th digit in the sequence -/
def digit_79 : Nat := sorry

/-- Theorem stating that the 79th digit in the sequence is 2 -/
theorem digit_79_is_2 : digit_79 = 2 := by sorry

end NUMINAMATH_CALUDE_digit_79_is_2_l313_31330


namespace NUMINAMATH_CALUDE_functional_equation_solution_l313_31309

theorem functional_equation_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x - f y) = f (f y) + x * f x + x^2) ↔ 
  (∀ x : ℝ, f x = 1 - x^2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l313_31309


namespace NUMINAMATH_CALUDE_f_is_K_function_l313_31371

-- Define a K function
def is_K_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0)

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- Theorem stating that f is a K function
theorem f_is_K_function : is_K_function f := by sorry

end NUMINAMATH_CALUDE_f_is_K_function_l313_31371


namespace NUMINAMATH_CALUDE_additive_inverse_of_zero_l313_31310

theorem additive_inverse_of_zero : (0 : ℤ) + (0 : ℤ) = (0 : ℤ) := by sorry

end NUMINAMATH_CALUDE_additive_inverse_of_zero_l313_31310


namespace NUMINAMATH_CALUDE_no_xy_term_l313_31340

-- Define the expression as a function of x, y, and a
def expression (x y a : ℝ) : ℝ :=
  2 * (x^2 - x*y + y^2) - (3*x^2 - a*x*y + y^2)

-- Theorem statement
theorem no_xy_term (a : ℝ) :
  (∀ x y : ℝ, ∃ k : ℝ, expression x y a = -x^2 + k + y^2) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_xy_term_l313_31340


namespace NUMINAMATH_CALUDE_intersection_M_N_l313_31301

-- Define the sets M and N
def M : Set ℝ := {x | (x + 2) * (x - 2) ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l313_31301


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l313_31319

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^6 - 1) * (X^2 - 1) = (X^3 - 1) * q + (X^2 - 1) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l313_31319


namespace NUMINAMATH_CALUDE_valid_sequences_count_l313_31300

/-- Represents a coin arrangement in a circle -/
def CoinArrangement := List Bool

/-- Represents a move that flips two adjacent coins -/
def Move := Nat

/-- The number of coins in the circle -/
def numCoins : Nat := 8

/-- The number of moves in a sequence -/
def numMoves : Nat := 6

/-- Checks if a coin arrangement is alternating heads and tails -/
def isAlternating (arrangement : CoinArrangement) : Bool :=
  sorry

/-- Applies a move to a coin arrangement -/
def applyMove (arrangement : CoinArrangement) (move : Move) : CoinArrangement :=
  sorry

/-- Applies a sequence of moves to a coin arrangement -/
def applyMoveSequence (arrangement : CoinArrangement) (moves : List Move) : CoinArrangement :=
  sorry

/-- Counts the number of valid 6-move sequences -/
def countValidSequences : Nat :=
  sorry

theorem valid_sequences_count :
  countValidSequences = 7680 :=
sorry

end NUMINAMATH_CALUDE_valid_sequences_count_l313_31300


namespace NUMINAMATH_CALUDE_joan_initial_books_l313_31322

/-- The number of books Joan initially gathered -/
def initial_books : ℕ := sorry

/-- The number of additional books Joan found -/
def additional_books : ℕ := 26

/-- The total number of books Joan has now -/
def total_books : ℕ := 59

/-- Theorem stating that the initial number of books is 33 -/
theorem joan_initial_books : 
  initial_books = total_books - additional_books :=
by sorry

end NUMINAMATH_CALUDE_joan_initial_books_l313_31322


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l313_31355

theorem polar_to_cartesian :
  let ρ : ℝ := 4
  let θ : ℝ := π / 3
  let x : ℝ := ρ * Real.cos θ
  let y : ℝ := ρ * Real.sin θ
  (x = 2 ∧ y = 2 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l313_31355


namespace NUMINAMATH_CALUDE_right_triangle_conditions_l313_31320

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles A, B, C in radians

-- Define the conditions
def condition1 (t : Triangle) : Prop := t.A + t.C = t.B
def condition2 (t : Triangle) : Prop := ∃ (k : ℝ), t.A = k ∧ t.B = 2*k ∧ t.C = 3*k
def condition3 (t : Triangle) : Prop := ∃ (AB BC AC : ℝ), 3*AB = 4*BC ∧ 4*BC = 5*AC
def condition4 (t : Triangle) : Prop := t.A = t.B ∧ t.B = t.C

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop := t.A = Real.pi/2 ∨ t.B = Real.pi/2 ∨ t.C = Real.pi/2

-- Theorem statement
theorem right_triangle_conditions (t : Triangle) :
  (condition1 t → is_right_triangle t) ∧
  (condition2 t → is_right_triangle t) ∧
  ¬(condition3 t → is_right_triangle t) ∧
  ¬(condition4 t → is_right_triangle t) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_conditions_l313_31320


namespace NUMINAMATH_CALUDE_range_of_m_for_sufficient_not_necessary_condition_l313_31362

/-- The range of m for which ¬p is a sufficient but not necessary condition for ¬q -/
theorem range_of_m_for_sufficient_not_necessary_condition 
  (p : ℝ → Prop) (q : ℝ → ℝ → Prop) (m : ℝ) : 
  (∀ x, p x ↔ x^2 - 8*x - 20 ≤ 0) →
  (∀ x, q x m ↔ x^2 - 2*x + 1 - m^2 ≤ 0) →
  m > 0 →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  0 < m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_sufficient_not_necessary_condition_l313_31362


namespace NUMINAMATH_CALUDE_function_identity_implies_zero_function_l313_31332

def IsPositive (n : ℤ) : Prop := n > 0

theorem function_identity_implies_zero_function 
  (f : ℤ → ℝ) 
  (h : ∀ (n m : ℤ), IsPositive n → IsPositive m → n ≥ m → 
       f (n + m) + f (n - m) = f (3 * n)) :
  ∀ (n : ℤ), IsPositive n → f n = 0 := by
sorry

end NUMINAMATH_CALUDE_function_identity_implies_zero_function_l313_31332


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l313_31329

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, (abs x = x → x^2 ≥ -x)) ∧
  (∃ x, x^2 ≥ -x ∧ abs x ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l313_31329


namespace NUMINAMATH_CALUDE_complement_of_union_equals_specific_set_l313_31337

-- Define the universal set U
def U : Set Int := {x | -3 < x ∧ x ≤ 4}

-- Define sets A and B
def A : Set Int := {-2, -1, 3}
def B : Set Int := {1, 2, 3}

-- State the theorem
theorem complement_of_union_equals_specific_set :
  (U \ (A ∪ B)) = {0, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_specific_set_l313_31337


namespace NUMINAMATH_CALUDE_calculation_proof_l313_31327

theorem calculation_proof : (-3) / (-1 - 3/4) * (3/4) / (3/7) = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l313_31327

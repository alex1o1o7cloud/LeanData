import Mathlib

namespace inequality_equivalence_l2263_226338

theorem inequality_equivalence (x : ℝ) :
  (3 ≤ |(x - 3)^2 - 4| ∧ |(x - 3)^2 - 4| ≤ 7) ↔ (3 - Real.sqrt 11 ≤ x ∧ x ≤ 3 + Real.sqrt 11) :=
by sorry

end inequality_equivalence_l2263_226338


namespace brittany_age_theorem_l2263_226370

/-- Brittany's age after returning from vacation -/
def brittany_age_after_vacation (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ) : ℕ :=
  rebecca_age + age_difference + vacation_duration

theorem brittany_age_theorem (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ) 
  (h1 : rebecca_age = 25)
  (h2 : age_difference = 3)
  (h3 : vacation_duration = 4) :
  brittany_age_after_vacation rebecca_age age_difference vacation_duration = 32 := by
  sorry

end brittany_age_theorem_l2263_226370


namespace equation_represents_parabola_l2263_226393

/-- The equation represents a parabola if it can be transformed into the form x² + bx + c = Ay + B --/
def is_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c A B : ℝ, a ≠ 0 ∧ 
  ∀ x y : ℝ, f x y ↔ a * x^2 + b * x + c = A * y + B

/-- The given equation |y-3| = √((x+4)² + y²) --/
def given_equation (x y : ℝ) : Prop :=
  |y - 3| = Real.sqrt ((x + 4)^2 + y^2)

theorem equation_represents_parabola : is_parabola given_equation := by
  sorry

end equation_represents_parabola_l2263_226393


namespace roots_of_quadratic_equation_l2263_226364

theorem roots_of_quadratic_equation (θ : Real) (x₁ x₂ : ℂ) :
  θ ∈ Set.Icc 0 π ∧
  x₁^2 - 3 * Real.sin θ * x₁ + Real.sin θ^2 + 1 = 0 ∧
  x₂^2 - 3 * Real.sin θ * x₂ + Real.sin θ^2 + 1 = 0 ∧
  Complex.abs x₁ + Complex.abs x₂ = 2 →
  θ = 0 := by sorry

end roots_of_quadratic_equation_l2263_226364


namespace expression_simplification_l2263_226360

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 4) : 
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 := by
  sorry

end expression_simplification_l2263_226360


namespace right_triangular_prism_volume_l2263_226371

theorem right_triangular_prism_volume 
  (a b h : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 2) 
  (hh : h = 3) 
  (right_triangle : a * a + b * b = (a + b) * (a + b) / 2) :
  (1 / 2) * a * b * h = 3 := by sorry

end right_triangular_prism_volume_l2263_226371


namespace det_evaluation_l2263_226348

theorem det_evaluation (x z : ℝ) : 
  Matrix.det !![1, x, z; 1, x + z, 2*z; 1, x, x + 2*z] = z * (3*x + z) := by
  sorry

end det_evaluation_l2263_226348


namespace equation_equivalence_l2263_226395

theorem equation_equivalence (a b c : ℝ) :
  (1 / (a + b) + 1 / (b + c) = 2 / (c + a)) ↔ (2 * b^2 = a^2 + c^2) := by sorry

end equation_equivalence_l2263_226395


namespace proof_by_contradiction_assumption_l2263_226322

theorem proof_by_contradiction_assumption (a b : ℕ) (h : 5 ∣ (a * b)) :
  (¬ (5 ∣ a) ∧ ¬ (5 ∣ b)) ↔ 
  ¬ (5 ∣ a ∨ 5 ∣ b) :=
by sorry

end proof_by_contradiction_assumption_l2263_226322


namespace product_of_solutions_l2263_226391

theorem product_of_solutions (x : ℝ) : 
  (25 = 3 * x^2 + 10 * x) → 
  (∃ α β : ℝ, (3 * α^2 + 10 * α = 25) ∧ (3 * β^2 + 10 * β = 25) ∧ (α * β = -25/3)) :=
by sorry

end product_of_solutions_l2263_226391


namespace childrens_admission_fee_l2263_226306

/-- Proves that the children's admission fee is $1.50 given the problem conditions -/
theorem childrens_admission_fee (total_people : ℕ) (total_fees : ℚ) (num_children : ℕ) (adult_fee : ℚ) :
  total_people = 315 →
  total_fees = 810 →
  num_children = 180 →
  adult_fee = 4 →
  ∃ (child_fee : ℚ),
    child_fee * num_children + adult_fee * (total_people - num_children) = total_fees ∧
    child_fee = 3/2 := by
  sorry

end childrens_admission_fee_l2263_226306


namespace continued_fraction_sum_l2263_226367

theorem continued_fraction_sum (a b c d : ℕ+) :
  (147 : ℚ) / 340 = 1 / (a + 1 / (b + 1 / (c + 1 / d))) →
  (a : ℕ) + b + c + d = 19 := by
  sorry

end continued_fraction_sum_l2263_226367


namespace difference_of_powers_l2263_226359

theorem difference_of_powers (a b c d : ℕ+) 
  (h1 : a ^ 5 = b ^ 4) 
  (h2 : c ^ 3 = d ^ 2) 
  (h3 : c - a = 19) : 
  d - b = 757 := by
sorry

end difference_of_powers_l2263_226359


namespace cubic_function_properties_l2263_226358

/-- Given a cubic function f with three distinct roots, prove properties about its values at 0, 1, and 3 -/
theorem cubic_function_properties (a b c : ℝ) (abc : ℝ) :
  a < b → b < c →
  let f : ℝ → ℝ := fun x ↦ x^3 - 6*x^2 + 9*x - abc
  f a = 0 → f b = 0 → f c = 0 →
  (f 0) * (f 1) < 0 ∧ (f 0) * (f 3) > 0 := by
  sorry

end cubic_function_properties_l2263_226358


namespace triangle_ratio_range_l2263_226374

theorem triangle_ratio_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Side lengths are positive
  a / Real.sin A = b / Real.sin B ∧  -- Law of sines
  b / Real.sin B = c / Real.sin C ∧  -- Law of sines
  -Real.cos B / Real.cos C = (2 * a + b) / c  -- Given condition
  →
  1 < (a + b) / c ∧ (a + b) / c ≤ 2 * Real.sqrt 3 / 3 :=
by sorry

end triangle_ratio_range_l2263_226374


namespace crayons_left_l2263_226366

theorem crayons_left (initial_crayons : ℕ) (crayons_taken : ℕ) : 
  initial_crayons = 7 → crayons_taken = 3 → initial_crayons - crayons_taken = 4 := by
  sorry

end crayons_left_l2263_226366


namespace integer_roots_of_polynomial_l2263_226308

theorem integer_roots_of_polynomial (a b c : ℚ) : 
  ∃ (p q : ℤ), p ≠ q ∧ 
    (∀ x : ℂ, x^4 + a*x^2 + b*x + c = 0 ↔ 
      (x = 2 - Real.sqrt 3 ∨ x = p ∨ x = q ∨ x = 2 + Real.sqrt 3)) ∧
    p = -1 ∧ q = -3 := by
  sorry

end integer_roots_of_polynomial_l2263_226308


namespace washing_machine_loads_l2263_226379

/-- Calculate the minimum number of loads required to wash a given number of items with a fixed machine capacity -/
def minimum_loads (total_items : ℕ) (machine_capacity : ℕ) : ℕ :=
  (total_items + machine_capacity - 1) / machine_capacity

/-- The washing machine capacity -/
def machine_capacity : ℕ := 12

/-- The total number of items to wash -/
def total_items : ℕ := 19 + 8 + 15 + 10

theorem washing_machine_loads :
  minimum_loads total_items machine_capacity = 5 := by
  sorry

end washing_machine_loads_l2263_226379


namespace total_layers_is_112_l2263_226344

/-- Represents an artist working on a multi-layered painting project -/
structure Artist where
  hours_per_week : ℕ
  hours_per_layer : ℕ

/-- Calculates the number of layers an artist can complete in a given number of weeks -/
def layers_completed (artist : Artist) (weeks : ℕ) : ℕ :=
  (artist.hours_per_week * weeks) / artist.hours_per_layer

/-- The duration of the project in weeks -/
def project_duration : ℕ := 4

/-- The team of artists working on the project -/
def artist_team : List Artist := [
  { hours_per_week := 30, hours_per_layer := 3 },
  { hours_per_week := 40, hours_per_layer := 5 },
  { hours_per_week := 20, hours_per_layer := 2 }
]

/-- Theorem: The total number of layers completed by all artists in the project is 112 -/
theorem total_layers_is_112 : 
  (artist_team.map (λ a => layers_completed a project_duration)).sum = 112 := by
  sorry

end total_layers_is_112_l2263_226344


namespace partial_fraction_decomposition_l2263_226398

theorem partial_fraction_decomposition (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 2 → x ≠ 3 →
    (50 * x - 42) / (x^2 - 5*x + 6) = N₁ / (x - 2) + N₂ / (x - 3)) →
  N₁ * N₂ = -6264 := by
sorry

end partial_fraction_decomposition_l2263_226398


namespace one_thirds_in_nine_fifths_l2263_226307

theorem one_thirds_in_nine_fifths : (9 : ℚ) / 5 / (1 / 3) = 27 / 5 := by sorry

end one_thirds_in_nine_fifths_l2263_226307


namespace imaginary_part_of_complex_fraction_l2263_226341

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (2 + Complex.I) / Complex.I
  Complex.im z = -2 := by sorry

end imaginary_part_of_complex_fraction_l2263_226341


namespace area_ratio_value_l2263_226311

/-- Represents a sequence of circles touching a right angle -/
structure CircleSequence where
  -- The ratio of radii between consecutive circles
  radius_ratio : ℝ
  -- Assumption that the ratio is equal to (√2 - 1)²
  h_ratio : radius_ratio = (Real.sqrt 2 - 1)^2

/-- The ratio of the area of the first circle to the sum of areas of all subsequent circles -/
def area_ratio (seq : CircleSequence) : ℝ := sorry

/-- Theorem stating the area ratio for the given circle sequence -/
theorem area_ratio_value (seq : CircleSequence) :
  area_ratio seq = 16 + 12 * Real.sqrt 2 := by sorry

end area_ratio_value_l2263_226311


namespace quadrilateral_tile_exists_l2263_226361

/-- A quadrilateral tile with angles measured in degrees -/
structure QuadTile where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ

/-- The property that six tiles meet at a vertex -/
def sixTilesMeet (t : QuadTile) : Prop :=
  ∃ (i : Fin 4), t.angle1 * (i.val : ℝ) + t.angle2 * ((4 - i).val : ℝ) = 360

/-- The sum of angles in a quadrilateral is 360° -/
def validQuadrilateral (t : QuadTile) : Prop :=
  t.angle1 + t.angle2 + t.angle3 + t.angle4 = 360

/-- The main theorem: there exists a quadrilateral tile with the specified angles -/
theorem quadrilateral_tile_exists : ∃ (t : QuadTile), 
  t.angle1 = 45 ∧ t.angle2 = 60 ∧ t.angle3 = 105 ∧ t.angle4 = 150 ∧
  sixTilesMeet t ∧ validQuadrilateral t :=
by
  sorry

end quadrilateral_tile_exists_l2263_226361


namespace candy_distribution_l2263_226330

theorem candy_distribution (x : ℕ) : 
  ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
sorry

end candy_distribution_l2263_226330


namespace mrs_hilt_markers_l2263_226312

theorem mrs_hilt_markers (num_packages : ℕ) (markers_per_package : ℕ) 
  (h1 : num_packages = 7) 
  (h2 : markers_per_package = 5) : 
  num_packages * markers_per_package = 35 := by
  sorry

end mrs_hilt_markers_l2263_226312


namespace first_year_balance_l2263_226387

/-- Proves that the total balance at the end of the first year is $5500,
    given the initial deposit of $5000 and the interest accrued in the first year of $500. -/
theorem first_year_balance (initial_deposit : ℝ) (interest_first_year : ℝ) 
  (h1 : initial_deposit = 5000)
  (h2 : interest_first_year = 500) :
  initial_deposit + interest_first_year = 5500 := by
  sorry

end first_year_balance_l2263_226387


namespace sqrt_comparison_l2263_226332

theorem sqrt_comparison : Real.sqrt 11 - 3 < Real.sqrt 7 - Real.sqrt 5 := by
  sorry

end sqrt_comparison_l2263_226332


namespace balloon_difference_l2263_226394

theorem balloon_difference (allan_initial : ℕ) (allan_bought : ℕ) (jake_balloons : ℕ)
  (h1 : allan_initial = 2)
  (h2 : allan_bought = 3)
  (h3 : jake_balloons = 6) :
  jake_balloons - (allan_initial + allan_bought) = 1 := by
  sorry

end balloon_difference_l2263_226394


namespace fifth_teapot_volume_l2263_226300

theorem fifth_teapot_volume
  (a : ℕ → ℚ)  -- arithmetic sequence of rational numbers
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- arithmetic sequence condition
  (h_length : ∀ n, n ≥ 9 → a n = a 9)  -- sequence has 9 terms
  (h_sum_first_three : a 1 + a 2 + a 3 = 1/2)  -- sum of first three terms
  (h_sum_last_three : a 7 + a 8 + a 9 = 5/2)  -- sum of last three terms
  : a 5 = 1/2 := by sorry

end fifth_teapot_volume_l2263_226300


namespace W_lower_bound_l2263_226303

/-- W(k,2) is the smallest number such that if n ≥ W(k,2), 
    for each coloring of the set {1,2,...,n} with two colors, 
    there exists a monochromatic arithmetic progression of length k -/
def W (k : ℕ) : ℕ := sorry

/-- The main theorem stating that W(k,2) = Ω(2^(k/2)) -/
theorem W_lower_bound : ∃ (c : ℝ) (k₀ : ℕ), c > 0 ∧ ∀ k ≥ k₀, (W k : ℝ) ≥ c * 2^(k/2 : ℝ) := by
  sorry

end W_lower_bound_l2263_226303


namespace equal_fractions_l2263_226399

theorem equal_fractions (x y z : ℝ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  let f1 := (x + y) / (x^2 + x*y + y^2)
  let f2 := (y + z) / (y^2 + y*z + z^2)
  let f3 := (z + x) / (z^2 + z*x + x^2)
  (f1 = f2 ∨ f2 = f3 ∨ f3 = f1) → (f1 = f2 ∧ f2 = f3) :=
by sorry

end equal_fractions_l2263_226399


namespace tan_sum_identity_l2263_226310

theorem tan_sum_identity (A B : Real) (hA : A = 10 * π / 180) (hB : B = 20 * π / 180) :
  (1 + Real.tan A) * (1 + Real.tan B) = 1 + Real.sqrt 3 * (Real.tan A + Real.tan B) := by
  sorry

end tan_sum_identity_l2263_226310


namespace max_value_inequality_l2263_226324

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (1 / x + 4 / y) ≤ 9 / 4 :=
by sorry

end max_value_inequality_l2263_226324


namespace conference_hall_tables_l2263_226385

/-- Given a conference hall with tables and chairs, prove the number of tables. -/
theorem conference_hall_tables (total_legs : ℕ) (chairs_per_table : ℕ) (chair_legs : ℕ) (table_legs : ℕ)
  (h1 : chairs_per_table = 8)
  (h2 : chair_legs = 4)
  (h3 : table_legs = 4)
  (h4 : total_legs = 648) :
  ∃ (num_tables : ℕ), num_tables = 18 ∧ 
    total_legs = num_tables * table_legs + num_tables * chairs_per_table * chair_legs :=
by sorry

end conference_hall_tables_l2263_226385


namespace hyperbola_circle_intersection_chord_length_l2263_226309

/-- Given a hyperbola and a circle, prove the length of the chord formed by their intersection -/
theorem hyperbola_circle_intersection_chord_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola : ℝ → ℝ → Prop) 
  (asymptote : ℝ → ℝ → Prop)
  (circle : ℝ → ℝ → Prop) :
  (∀ x y, hyperbola x y ↔ x^2 / a^2 - y^2 / b^2 = 1) →
  (asymptote 1 2) →
  (∀ x y, circle x y ↔ (x + 1)^2 + (y - 2)^2 = 4) →
  ∃ chord_length, chord_length = 4 * Real.sqrt 5 / 5 :=
sorry

end hyperbola_circle_intersection_chord_length_l2263_226309


namespace unique_two_digit_integer_l2263_226315

theorem unique_two_digit_integer (u : ℕ) : 
  (10 ≤ u ∧ u < 100) →
  (15 * u) % 100 = 45 →
  u % 17 = 7 →
  u = 43 :=
by sorry

end unique_two_digit_integer_l2263_226315


namespace shadow_arrangements_l2263_226369

def word_length : Nat := 6
def selection_size : Nat := 4
def remaining_letters : Nat := word_length - 1  -- excluding 'a'
def letters_to_choose : Nat := selection_size - 1  -- excluding 'a'

theorem shadow_arrangements : 
  (Nat.choose remaining_letters letters_to_choose) * 
  (Nat.factorial selection_size) = 240 := by
sorry

end shadow_arrangements_l2263_226369


namespace flower_bed_width_l2263_226304

theorem flower_bed_width (area : ℝ) (length : ℝ) (width : ℝ) :
  area = 35 →
  length = 7 →
  area = length * width →
  width = 5 :=
by
  sorry

end flower_bed_width_l2263_226304


namespace haleys_concert_tickets_l2263_226375

theorem haleys_concert_tickets (ticket_price : ℕ) (extra_tickets : ℕ) (total_spent : ℕ) : 
  ticket_price = 4 → 
  extra_tickets = 5 → 
  total_spent = 32 → 
  ∃ (tickets_for_friends : ℕ), 
    ticket_price * (tickets_for_friends + extra_tickets) = total_spent ∧ 
    tickets_for_friends = 3 :=
by sorry

end haleys_concert_tickets_l2263_226375


namespace zeta_sum_sixth_power_l2263_226396

theorem zeta_sum_sixth_power (ζ₁ ζ₂ ζ₃ : ℂ)
  (sum_condition : ζ₁ + ζ₂ + ζ₃ = 2)
  (sum_squares : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (sum_fourth_powers : ζ₁^4 + ζ₂^4 + ζ₃^4 = 29) :
  ζ₁^6 + ζ₂^6 + ζ₃^6 = 101.40625 := by
sorry

end zeta_sum_sixth_power_l2263_226396


namespace max_boxes_arrangement_l2263_226351

/-- A Box represents a rectangle in the plane with sides parallel to coordinate axes. -/
structure Box where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h_positive : x₁ < x₂ ∧ y₁ < y₂

/-- Two boxes intersect if they have a common point. -/
def intersect (b₁ b₂ : Box) : Prop :=
  ¬(b₁.x₂ ≤ b₂.x₁ ∨ b₂.x₂ ≤ b₁.x₁ ∨ b₁.y₂ ≤ b₂.y₁ ∨ b₂.y₂ ≤ b₁.y₁)

/-- A valid arrangement of n boxes satisfies the intersection condition. -/
def valid_arrangement (n : ℕ) (boxes : Fin n → Box) : Prop :=
  ∀ i j : Fin n, intersect (boxes i) (boxes j) ↔ (i.val + 1) % n ≠ j.val ∧ (i.val + n - 1) % n ≠ j.val

/-- The main theorem: The maximum number of boxes in a valid arrangement is 6. -/
theorem max_boxes_arrangement :
  (∃ (boxes : Fin 6 → Box), valid_arrangement 6 boxes) ∧
  (∀ n : ℕ, n > 6 → ¬∃ (boxes : Fin n → Box), valid_arrangement n boxes) :=
sorry

end max_boxes_arrangement_l2263_226351


namespace seventeen_meter_rod_pieces_l2263_226313

/-- The number of pieces of a given length that can be cut from a rod --/
def num_pieces (rod_length : ℕ) (piece_length : ℕ) : ℕ :=
  rod_length / piece_length

/-- Theorem: The number of 85 cm pieces that can be cut from a 17-meter rod is 20 --/
theorem seventeen_meter_rod_pieces : num_pieces (17 * 100) 85 = 20 := by
  sorry

end seventeen_meter_rod_pieces_l2263_226313


namespace triangle_side_length_l2263_226342

/-- Represents a triangle with sides a, b, c and median ma from vertex A to midpoint of side BC. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ma : ℝ

/-- The theorem states that for a triangle with sides 6 and 9, and a median of 5,
    the third side has length √134. -/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 6)
  (h2 : t.b = 9) 
  (h3 : t.ma = 5) : 
  t.c = Real.sqrt 134 := by
  sorry

end triangle_side_length_l2263_226342


namespace base13_representation_of_234_l2263_226357

/-- Represents a digit in base 13 -/
inductive Base13Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a natural number to its base 13 representation -/
def toBase13 (n : ℕ) : List Base13Digit := sorry

/-- Converts a list of Base13Digits to its decimal (base 10) value -/
def fromBase13 (digits : List Base13Digit) : ℕ := sorry

theorem base13_representation_of_234 :
  toBase13 234 = [Base13Digit.D1, Base13Digit.D5] := by sorry

end base13_representation_of_234_l2263_226357


namespace circle_square_area_l2263_226389

/-- A circle described by the equation 2x^2 = -2y^2 + 8x - 8y + 28 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1^2 = -2 * p.2^2 + 8 * p.1 - 8 * p.2 + 28}

/-- The square that circumscribes the circle with sides parallel to the axes -/
def CircumscribingSquare (c : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), (x, y) ∈ c ∧ 
    (p.1 = x ∨ p.1 = -x) ∧ (p.2 = y ∨ p.2 = -y)}

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

theorem circle_square_area : area (CircumscribingSquare Circle) = 88 := by sorry

end circle_square_area_l2263_226389


namespace largest_number_proof_l2263_226325

theorem largest_number_proof (a b c d : ℕ) 
  (sum1 : a + b + c = 180)
  (sum2 : a + b + d = 197)
  (sum3 : a + c + d = 208)
  (sum4 : b + c + d = 222) :
  max a (max b (max c d)) = 89 := by
sorry

end largest_number_proof_l2263_226325


namespace unique_solution_power_sum_square_l2263_226388

theorem unique_solution_power_sum_square :
  ∃! (x y z : ℕ+), 2^(x.val) + 3^(y.val) = z.val^2 ∧ x = 4 ∧ y = 2 ∧ z = 5 := by
  sorry

end unique_solution_power_sum_square_l2263_226388


namespace f_le_g_l2263_226327

def f (n : ℕ+) : ℚ :=
  (Finset.range n).sum (fun i => 1 / ((i + 1) : ℚ) ^ 2) + 1

def g (n : ℕ+) : ℚ :=
  1/2 * (3 - 1 / (n : ℚ) ^ 2)

theorem f_le_g : ∀ n : ℕ+, f n ≤ g n := by
  sorry

end f_le_g_l2263_226327


namespace parking_lot_cars_l2263_226323

theorem parking_lot_cars (car_wheels : ℕ) (motorcycle_wheels : ℕ) (num_motorcycles : ℕ) (total_wheels : ℕ) :
  car_wheels = 5 →
  motorcycle_wheels = 2 →
  num_motorcycles = 11 →
  total_wheels = 117 →
  ∃ num_cars : ℕ, num_cars * car_wheels + num_motorcycles * motorcycle_wheels = total_wheels ∧ num_cars = 19 :=
by sorry

end parking_lot_cars_l2263_226323


namespace rocket_components_most_suitable_for_comprehensive_survey_l2263_226397

/-- Represents the characteristics of a scenario that can be surveyed -/
structure SurveyScenario where
  population : Type
  countable : Bool
  criticalImportance : Bool
  requiresCompleteExamination : Bool

/-- Defines what makes a scenario suitable for a comprehensive survey -/
def isSuitableForComprehensiveSurvey (scenario : SurveyScenario) : Prop :=
  scenario.countable ∧ scenario.criticalImportance ∧ scenario.requiresCompleteExamination

/-- Represents the Long March II-F Y17 rocket components scenario -/
def rocketComponentsScenario : SurveyScenario :=
  { population := Unit,  -- The type doesn't matter for this example
    countable := true,
    criticalImportance := true,
    requiresCompleteExamination := true }

/-- Represents all other given scenarios -/
def otherScenarios : List SurveyScenario :=
  [ { population := Unit, countable := false, criticalImportance := false, requiresCompleteExamination := false },
    { population := Unit, countable := false, criticalImportance := true, requiresCompleteExamination := false },
    { population := Unit, countable := false, criticalImportance := false, requiresCompleteExamination := false } ]

theorem rocket_components_most_suitable_for_comprehensive_survey :
  isSuitableForComprehensiveSurvey rocketComponentsScenario ∧
  (∀ scenario ∈ otherScenarios, ¬(isSuitableForComprehensiveSurvey scenario)) :=
sorry

end rocket_components_most_suitable_for_comprehensive_survey_l2263_226397


namespace square_sum_from_means_l2263_226339

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 150) : 
  x^2 + y^2 = 1300 := by
sorry

end square_sum_from_means_l2263_226339


namespace paperboy_delivery_sequences_l2263_226350

/-- Recurrence relation for the number of valid delivery sequences -/
def D : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 7
  | n + 4 => D (n + 3) + D (n + 2) + D (n + 1)

/-- Number of valid delivery sequences ending with a delivery -/
def E (n : ℕ) : ℕ := D (n - 2)

/-- The number of houses on King's Avenue -/
def num_houses : ℕ := 15

theorem paperboy_delivery_sequences :
  E num_houses = 3136 := by sorry

end paperboy_delivery_sequences_l2263_226350


namespace expression_evaluation_l2263_226378

theorem expression_evaluation : -25 + 12 * (8 / (2 + 2)) = -1 := by
  sorry

end expression_evaluation_l2263_226378


namespace best_candidate_is_C_l2263_226336

structure Participant where
  name : String
  average_score : Float
  variance : Float

def participants : List Participant := [
  { name := "A", average_score := 8.5, variance := 1.7 },
  { name := "B", average_score := 8.8, variance := 2.1 },
  { name := "C", average_score := 9.1, variance := 1.7 },
  { name := "D", average_score := 9.1, variance := 2.5 }
]

def is_best_candidate (p : Participant) : Prop :=
  ∀ q ∈ participants,
    (p.average_score > q.average_score ∨
    (p.average_score = q.average_score ∧ p.variance ≤ q.variance))

theorem best_candidate_is_C :
  ∃ p ∈ participants, p.name = "C" ∧ is_best_candidate p :=
by sorry

end best_candidate_is_C_l2263_226336


namespace sum_with_radical_conjugate_l2263_226319

theorem sum_with_radical_conjugate :
  let x : ℝ := 12 - Real.sqrt 2023
  let y : ℝ := 12 + Real.sqrt 2023
  x + y = 24 := by sorry

end sum_with_radical_conjugate_l2263_226319


namespace pakistan_traditional_model_l2263_226381

-- Define the population growth models
inductive PopulationModel
  | Primitive
  | Traditional
  | Modern

-- Define a function that assigns a population model to a country
def countryModel : String → PopulationModel
  | "Nigeria" => PopulationModel.Traditional
  | "China" => PopulationModel.Modern
  | "India" => PopulationModel.Traditional
  | "Pakistan" => PopulationModel.Traditional
  | _ => PopulationModel.Traditional  -- Default case

-- Theorem stating that Pakistan follows the Traditional model
theorem pakistan_traditional_model :
  countryModel "Pakistan" = PopulationModel.Traditional := by
  sorry

end pakistan_traditional_model_l2263_226381


namespace inequality_proof_l2263_226334

theorem inequality_proof (x y z : ℝ) (h : x^4 + y^4 + z^4 + x*y*z = 4) :
  x ≤ 2 ∧ Real.sqrt (2 - x) ≥ (y + z) / 2 := by
  sorry

end inequality_proof_l2263_226334


namespace even_function_implies_a_equals_four_l2263_226352

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x+a)(x-4) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 4)

/-- If f(x) = (x+a)(x-4) is an even function, then a = 4 -/
theorem even_function_implies_a_equals_four (a : ℝ) :
  IsEven (f a) → a = 4 := by
  sorry

end even_function_implies_a_equals_four_l2263_226352


namespace hamburger_combinations_count_l2263_226372

/-- The number of condiments available for hamburgers -/
def num_condiments : ℕ := 8

/-- The number of patty options available for hamburgers -/
def num_patty_options : ℕ := 4

/-- Calculates the number of different hamburger combinations -/
def num_hamburger_combinations : ℕ := 2^num_condiments * num_patty_options

theorem hamburger_combinations_count :
  num_hamburger_combinations = 1024 := by
  sorry

end hamburger_combinations_count_l2263_226372


namespace samson_age_relation_l2263_226365

/-- Samson's current age in years -/
def samsonAge : ℝ := 6.25

/-- Samson's mother's current age in years -/
def motherAge : ℝ := 30.65

/-- The age Samson will be when his mother is exactly 4 times his age -/
def targetAge : ℝ := 8.1333

theorem samson_age_relation :
  ∃ (T : ℝ), 
    (samsonAge + T = targetAge) ∧ 
    (motherAge + T = 4 * (samsonAge + T)) := by
  sorry

end samson_age_relation_l2263_226365


namespace rectangle_perimeter_equals_area_l2263_226380

theorem rectangle_perimeter_equals_area (x y : ℕ) : 
  x ≠ y →
  x > 0 →
  y > 0 →
  2 * (x + y) = x * y →
  ((x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3)) :=
sorry

end rectangle_perimeter_equals_area_l2263_226380


namespace incorrect_relation_l2263_226317

theorem incorrect_relation (a b : ℝ) (h : a > b) : ∃ c : ℝ, ¬(a * c^2 > b * c^2) := by
  sorry

end incorrect_relation_l2263_226317


namespace locus_is_ellipse_l2263_226390

/-- The locus of points (x, y) in the complex plane satisfying the given equation is an ellipse -/
theorem locus_is_ellipse (x y : ℝ) : 
  let z : ℂ := x + y * Complex.I
  (Complex.abs (z - (2 - Complex.I)) + Complex.abs (z - (-3 + Complex.I)) = 6) →
  ∃ (a b c d e f : ℝ), 
    a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0 ∧ 
    b^2 - 4*a*c < 0 :=
by sorry

end locus_is_ellipse_l2263_226390


namespace polynomial_division_remainder_l2263_226386

theorem polynomial_division_remainder :
  ∃ (q : Polynomial ℝ), x^4 + x^3 - 4*x + 1 = (x^3 - 1) * q + (-3*x + 2) :=
by sorry

end polynomial_division_remainder_l2263_226386


namespace problem_solution_l2263_226354

theorem problem_solution :
  (∀ x : ℝ, x^2 - x ≥ x - 1) ∧
  (∃ x : ℝ, x > 1 ∧ x + 4 / (x - 1) = 6) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → b / a < (b + 1) / (a + 1)) ∧
  (∀ x : ℝ, (x^2 + 10) / Real.sqrt (x^2 + 9) > 2) :=
by sorry

end problem_solution_l2263_226354


namespace tangent_line_equation_l2263_226320

-- Define the curve
def f (x : ℝ) : ℝ := -x^2 + 4

-- Define the point of interest
def x₀ : ℝ := -1

-- Define the slope of the tangent line
def k : ℝ := -2 * x₀

-- Define the y-coordinate of the point on the curve
def y₀ : ℝ := f x₀

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, y = k * (x - x₀) + y₀ ↔ y = 2*x + 5 :=
by sorry

end tangent_line_equation_l2263_226320


namespace fish_after_ten_years_l2263_226355

def initial_fish : ℕ := 6

def fish_added (year : ℕ) : ℕ :=
  if year ≤ 10 then year + 1 else 0

def fish_died (year : ℕ) : ℕ :=
  if year ≤ 10 then
    if year ≤ 4 then 5 - year
    else year - 3
  else 0

def fish_count (year : ℕ) : ℕ :=
  if year = 0 then initial_fish
  else (fish_count (year - 1) + fish_added year - fish_died year)

theorem fish_after_ten_years :
  fish_count 10 = 34 := by sorry

end fish_after_ten_years_l2263_226355


namespace anniversary_sale_cost_l2263_226353

/-- The cost of the purchase during the anniversary sale -/
def total_cost (original_ice_cream_price sale_discount juice_price_per_5 ice_cream_tubs juice_cans : ℚ) : ℚ :=
  (ice_cream_tubs * (original_ice_cream_price - sale_discount)) + 
  (juice_cans / 5 * juice_price_per_5)

/-- Theorem stating that the total cost of the purchase is $24 -/
theorem anniversary_sale_cost : 
  total_cost 12 2 2 2 10 = 24 := by sorry

end anniversary_sale_cost_l2263_226353


namespace vector_sum_of_squares_l2263_226302

/-- Given vectors a and b, with n as their midpoint, prove that ‖a‖² + ‖b‖² = 48 -/
theorem vector_sum_of_squares (a b : ℝ × ℝ) (n : ℝ × ℝ) : 
  n = (4, -1) → n = (a + b) / 2 → a • b = 10 → ‖a‖^2 + ‖b‖^2 = 48 := by
  sorry

end vector_sum_of_squares_l2263_226302


namespace work_completion_time_l2263_226335

theorem work_completion_time 
  (efficiency_ratio : ℝ) 
  (combined_time : ℝ) 
  (a_efficiency : ℝ) 
  (b_efficiency : ℝ) :
  efficiency_ratio = 2 →
  combined_time = 6 →
  a_efficiency = efficiency_ratio * b_efficiency →
  (a_efficiency + b_efficiency) * combined_time = b_efficiency * 18 :=
by sorry

end work_completion_time_l2263_226335


namespace hyperbola_properties_l2263_226382

/-- Represents a hyperbola with the equation (x^2 / a^2) - (y^2 / b^2) = 1 -/
structure Hyperbola (a b : ℝ) where
  equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

/-- Represents an asymptote of a hyperbola -/
structure Asymptote (m : ℝ) where
  equation : ∀ x y : ℝ, y = m * x

/-- Represents a focus point of a hyperbola -/
structure Focus (x y : ℝ) where
  coordinates : ℝ × ℝ := (x, y)

theorem hyperbola_properties (h : Hyperbola 12 9) :
  (∃ a₁ : Asymptote (3/4), True) ∧
  (∃ a₂ : Asymptote (-3/4), True) ∧
  (∃ f₁ : Focus 15 0, True) ∧
  (∃ f₂ : Focus (-15) 0, True) := by
  sorry


end hyperbola_properties_l2263_226382


namespace expansion_terms_abcd_efghi_l2263_226345

/-- The number of terms in the expansion of a product of two sums -/
def expansion_terms (n m : ℕ) : ℕ := n * m

/-- The first group (a+b+c+d) has 4 terms -/
def first_group_terms : ℕ := 4

/-- The second group (e+f+g+h+i) has 5 terms -/
def second_group_terms : ℕ := 5

/-- Theorem: The number of terms in the expansion of (a+b+c+d)(e+f+g+h+i) is 20 -/
theorem expansion_terms_abcd_efghi :
  expansion_terms first_group_terms second_group_terms = 20 := by
  sorry

end expansion_terms_abcd_efghi_l2263_226345


namespace missing_chess_pieces_l2263_226337

/-- The number of pieces in a standard chess set -/
def standard_chess_set_pieces : ℕ := 32

/-- The number of pieces present -/
def present_pieces : ℕ := 28

/-- The number of missing pieces -/
def missing_pieces : ℕ := standard_chess_set_pieces - present_pieces

theorem missing_chess_pieces : missing_pieces = 4 := by
  sorry

end missing_chess_pieces_l2263_226337


namespace intersection_of_M_and_N_l2263_226349

-- Define the sets M and N
def M : Set ℝ := {x | 3 * x - x^2 > 0}
def N : Set ℝ := {x | x^2 - 4 * x + 3 > 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Ioo 0 1 := by sorry

end intersection_of_M_and_N_l2263_226349


namespace parabola_and_line_properties_l2263_226376

/-- Represents a parabola in the form y² = -2px --/
structure Parabola where
  p : ℝ

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about a specific parabola and line --/
theorem parabola_and_line_properties
  (C : Parabola)
  (A : Point)
  (h1 : A.y^2 = -2 * C.p * A.x) -- A lies on the parabola
  (h2 : A.x = -1 ∧ A.y = -2) -- A is (-1, -2)
  (h3 : ∃ (B : Point), B ≠ A ∧ 
    (B.y - A.y) / (B.x - A.x) = -Real.sqrt 3 ∧ -- Line AB has slope -√3
    B.y^2 = -2 * C.p * B.x) -- B also lies on the parabola
  : 
  (C.p = -2) ∧ -- Equation of parabola is y² = -4x
  (∀ (x y : ℝ), y^2 = -4*x ↔ y^2 = -2 * C.p * x) ∧ -- Equivalent form of parabola equation
  (1 = -C.p/2) ∧ -- Axis of symmetry is x = 1
  (∃ (B : Point), B ≠ A ∧
    (B.y - A.y)^2 + (B.x - A.x)^2 = (16/3)^2) -- Length of AB is 16/3
  := by sorry

end parabola_and_line_properties_l2263_226376


namespace f_2004_value_l2263_226329

/-- A function with the property that f(a) + f(b) = n^3 when a + b = 2^(n+1) -/
def special_function (f : ℕ → ℕ) : Prop :=
  ∀ (a b n : ℕ), a > 0 → b > 0 → n > 0 → a + b = 2^(n+1) → f a + f b = n^3

theorem f_2004_value (f : ℕ → ℕ) (h : special_function f) : f 2004 = 1320 := by
  sorry

end f_2004_value_l2263_226329


namespace complement_intersection_equality_l2263_226301

def S : Set Nat := {1,2,3,4,5}
def M : Set Nat := {1,4}
def N : Set Nat := {2,4}

theorem complement_intersection_equality : 
  (S \ M) ∩ (S \ N) = {3,5} := by sorry

end complement_intersection_equality_l2263_226301


namespace blue_chip_fraction_l2263_226368

theorem blue_chip_fraction (total : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : total = 60)
  (h2 : red = 34)
  (h3 : green = 16) :
  (total - red - green : ℚ) / total = 1 / 6 :=
by sorry

end blue_chip_fraction_l2263_226368


namespace difference_of_roots_quadratic_l2263_226321

theorem difference_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → |r₁ - r₂| = 3 :=
by
  sorry

#check difference_of_roots_quadratic 1 (-9) 18

end difference_of_roots_quadratic_l2263_226321


namespace lateral_surface_area_equilateral_prism_l2263_226363

/-- The lateral surface area of a prism with an equilateral triangular base -/
theorem lateral_surface_area_equilateral_prism (a : ℝ) (h : a > 0) :
  let base_side := a
  let base_center_to_vertex := a * Real.sqrt 3 / 3
  let edge_angle := 60 * π / 180
  let edge_length := 2 * base_center_to_vertex / Real.cos edge_angle
  let lateral_perimeter := a + a * Real.sqrt 13 / 2
  lateral_perimeter * edge_length = a^2 * Real.sqrt 3 * (Real.sqrt 13 + 2) / 3 :=
by sorry

end lateral_surface_area_equilateral_prism_l2263_226363


namespace pipe_equivalence_l2263_226314

/-- The number of smaller pipes needed to match the water-carrying capacity of a larger pipe -/
theorem pipe_equivalence (r_large r_small : ℝ) (h_large : r_large = 4) (h_small : r_small = 1) :
  (π * r_large ^ 2) / (π * r_small ^ 2) = 16 := by
  sorry

#check pipe_equivalence

end pipe_equivalence_l2263_226314


namespace campaign_fundraising_l2263_226383

-- Define the problem parameters
def max_donation : ℕ := 1200
def max_donors : ℕ := 500
def half_donors_multiplier : ℕ := 3
def donation_percentage : ℚ := 40 / 100

-- Define the total money raised
def total_money_raised : ℚ := 3750000

-- Theorem statement
theorem campaign_fundraising :
  let max_donation_total := max_donation * max_donors
  let half_donation_total := (max_donation / 2) * (max_donors * half_donors_multiplier)
  let total_donations := max_donation_total + half_donation_total
  total_donations = donation_percentage * total_money_raised := by
  sorry


end campaign_fundraising_l2263_226383


namespace quadratic_inequality_solution_l2263_226362

theorem quadratic_inequality_solution (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 4)*x - k + 8 > 0) ↔ k ∈ Set.Ioo (-8/3) 6 := by
  sorry

end quadratic_inequality_solution_l2263_226362


namespace chromium_percentage_in_new_alloy_l2263_226333

/-- Represents an alloy with its chromium percentage and weight -/
structure Alloy where
  chromium_percentage : Float
  weight : Float

/-- Calculates the total chromium weight in an alloy -/
def chromium_weight (a : Alloy) : Float :=
  a.chromium_percentage / 100 * a.weight

/-- Calculates the percentage of chromium in a new alloy formed by combining multiple alloys -/
def new_alloy_chromium_percentage (alloys : List Alloy) : Float :=
  let total_chromium : Float := (alloys.map chromium_weight).sum
  let total_weight : Float := (alloys.map (·.weight)).sum
  total_chromium / total_weight * 100

theorem chromium_percentage_in_new_alloy : 
  let a1 : Alloy := { chromium_percentage := 12, weight := 15 }
  let a2 : Alloy := { chromium_percentage := 10, weight := 35 }
  let a3 : Alloy := { chromium_percentage := 8, weight := 25 }
  let a4 : Alloy := { chromium_percentage := 15, weight := 10 }
  let alloys : List Alloy := [a1, a2, a3, a4]
  new_alloy_chromium_percentage alloys = 10.35 := by
  sorry

end chromium_percentage_in_new_alloy_l2263_226333


namespace smaller_number_problem_l2263_226316

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) :
  min x y = 15 := by
  sorry

end smaller_number_problem_l2263_226316


namespace jogger_train_distance_l2263_226318

/-- Proves that a jogger is 200 meters ahead of a train's engine given specific conditions --/
theorem jogger_train_distance (jogger_speed train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (5 / 18) →
  train_speed = 45 * (5 / 18) →
  train_length = 200 →
  passing_time = 40 →
  (train_speed - jogger_speed) * passing_time = train_length + 200 :=
by
  sorry

#check jogger_train_distance

end jogger_train_distance_l2263_226318


namespace common_sum_in_square_matrix_l2263_226326

theorem common_sum_in_square_matrix : 
  let n : ℕ := 36
  let a : ℤ := -15
  let l : ℤ := 20
  let total_sum : ℤ := n * (a + l) / 2
  let matrix_size : ℕ := 6
  total_sum / matrix_size = 15 := by sorry

end common_sum_in_square_matrix_l2263_226326


namespace quadratic_inequality_and_logic_l2263_226343

theorem quadratic_inequality_and_logic :
  (∀ x : ℝ, x^2 + x + 1 ≥ 0) ∧
  ¬(∃ x : ℝ, x^2 + x + 1 < 0) ∧
  ((∀ x : ℝ, x^2 + x + 1 ≥ 0) → 
   ((∃ x : ℝ, x^2 + x + 1 < 0) ∨ (∀ x : ℝ, x^2 + x + 1 ≥ 0))) :=
by sorry

end quadratic_inequality_and_logic_l2263_226343


namespace labeling_periodic_l2263_226346

/-- Represents the labeling of vertices at a given time -/
def Labeling := Fin 1993 → Int

/-- The rule for updating labels -/
def update_label (l : Labeling) (n : Fin 1993) : Int :=
  if l (n - 1) = l (n + 1) then 1 else -1

/-- The next labeling based on the current one -/
def next_labeling (l : Labeling) : Labeling :=
  fun n => update_label l n

/-- The labeling after t steps -/
def labeling_at_time (initial : Labeling) : ℕ → Labeling
  | 0 => initial
  | t + 1 => next_labeling (labeling_at_time initial t)

theorem labeling_periodic (initial : Labeling) :
  ∃ n : ℕ, n > 1 ∧ labeling_at_time initial n = labeling_at_time initial 1 := by
  sorry

end labeling_periodic_l2263_226346


namespace sequence_problem_l2263_226340

def D (A : ℕ → ℝ) : ℕ → ℝ := λ n => A (n + 1) - A n

theorem sequence_problem (A : ℕ → ℝ) 
  (h1 : ∀ n, D (D A) n = 1) 
  (h2 : A 19 = 0) 
  (h3 : A 92 = 0) : 
  A 1 = 819 := by
sorry

end sequence_problem_l2263_226340


namespace average_of_remaining_numbers_l2263_226328

theorem average_of_remaining_numbers
  (total : ℝ)
  (avg_all : ℝ)
  (avg_group1 : ℝ)
  (avg_group2 : ℝ)
  (h1 : total = 6 * avg_all)
  (h2 : avg_all = 3.95)
  (h3 : avg_group1 = 3.6)
  (h4 : avg_group2 = 3.85) :
  (total - 2 * avg_group1 - 2 * avg_group2) / 2 = 4.4 := by
sorry

end average_of_remaining_numbers_l2263_226328


namespace type_q_machine_time_l2263_226305

theorem type_q_machine_time (q : ℝ) (h1 : q > 0) 
  (h2 : 2 / q + 3 / 7 = 1 / 1.2) : q = 84 / 17 := by
  sorry

end type_q_machine_time_l2263_226305


namespace complex_equation_sum_l2263_226392

theorem complex_equation_sum (x y : ℝ) : 
  (Complex.mk (x - 1) (y + 1)) * (Complex.mk 2 1) = 0 → x + y = 0 := by
sorry

end complex_equation_sum_l2263_226392


namespace root_in_interval_l2263_226331

-- Define the function g(x) = lg x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 10 + x - 2

-- State the theorem
theorem root_in_interval :
  ∃ x₀ : ℝ, g x₀ = 0 ∧ 1 < x₀ ∧ x₀ < 2 :=
sorry

end root_in_interval_l2263_226331


namespace like_terms_exponent_relation_l2263_226384

theorem like_terms_exponent_relation (x y : ℤ) : 
  (∃ (m n : ℝ), -0.5 * m^x * n^3 = 5 * m^4 * n^y) → (y - x)^2023 = -1 := by
  sorry

end like_terms_exponent_relation_l2263_226384


namespace distance_between_parallel_lines_l2263_226373

/-- Given two lines l₁ and l₂, prove that their distance is √10/5 -/
theorem distance_between_parallel_lines (m : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | 2*x + 3*m*y - m + 2 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | m*x + 6*y - 4 = 0}
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l₁ → (x₂, y₂) ∈ l₂ → 
    (2 * (y₂ - y₁) = 3*m * (x₂ - x₁))) →  -- parallel condition
  (∃ (d : ℝ), d = Real.sqrt 10 / 5 ∧
    ∀ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ l₁ → p₂ ∈ l₂ →
      d ≤ Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)) :=
by sorry

end distance_between_parallel_lines_l2263_226373


namespace sum_of_squares_of_roots_l2263_226347

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 1 = 0 → x₂^2 - 2*x₂ - 1 = 0 → x₁^2 + x₂^2 = 6 := by
  sorry

end sum_of_squares_of_roots_l2263_226347


namespace three_digit_sum_problem_l2263_226377

/-- Represents a three-digit number in the form abc -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem three_digit_sum_problem (a c : Nat) :
  let num1 := ThreeDigitNumber.mk 3 a 7 (by sorry)
  let num2 := ThreeDigitNumber.mk 2 1 4 (by sorry)
  let sum := ThreeDigitNumber.mk 5 c 1 (by sorry)
  (num1.toNat + num2.toNat = sum.toNat) →
  (sum.toNat % 3 = 0) →
  a + c = 4 := by
  sorry

#check three_digit_sum_problem

end three_digit_sum_problem_l2263_226377


namespace triangle_in_circle_and_polygon_l2263_226356

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the regular polygon
structure RegularPolygon where
  n : ℕ
  vertices : Fin n → ℝ × ℝ

def is_inscribed (t : Triangle) (c : Circle) : Prop :=
  sorry

def angle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  sorry

def are_adjacent_vertices (p1 p2 : ℝ × ℝ) (poly : RegularPolygon) : Prop :=
  sorry

theorem triangle_in_circle_and_polygon (t : Triangle) (c : Circle) (poly : RegularPolygon) :
  is_inscribed t c →
  angle t.B t.A t.C = angle t.C t.A t.B →
  angle t.B t.A t.C = 3 * angle t.A t.B t.C →
  are_adjacent_vertices t.B t.C poly →
  is_inscribed (Triangle.mk t.A t.B t.C) c →
  poly.n = 7 :=
sorry

end triangle_in_circle_and_polygon_l2263_226356

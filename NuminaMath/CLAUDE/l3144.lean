import Mathlib

namespace object_height_l3144_314447

/-- The height function of an object thrown upward -/
def h (k : ℝ) (t : ℝ) : ℝ := -k * (t - 3)^2 + 150

/-- The value of k for which the object is at 94 feet after 5 seconds -/
theorem object_height (k : ℝ) : h k 5 = 94 → k = 14 := by
  sorry

end object_height_l3144_314447


namespace system_solution_l3144_314474

theorem system_solution : ∃! (x y z : ℝ), 
  (x - y ≥ z ∧ x^2 + 4*y^2 + 5 = 4*z) ∧ 
  x = 2 ∧ y = -1/2 ∧ z = 5/2 := by
  sorry

end system_solution_l3144_314474


namespace exponential_above_line_l3144_314499

theorem exponential_above_line (k : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.exp x > k * x + 1) → k ≤ 1 := by
  sorry

end exponential_above_line_l3144_314499


namespace max_draws_for_all_pairs_l3144_314403

/-- Represents the number of items of a specific color -/
structure ColorCount where
  total : Nat
  deriving Repr

/-- Calculates the maximum number of draws needed to guarantee a pair for a single color -/
def maxDrawsForColor (count : ColorCount) : Nat :=
  count.total + 1

/-- The box containing hats and gloves -/
structure Box where
  red : ColorCount
  green : ColorCount
  orange : ColorCount
  blue : ColorCount
  yellow : ColorCount

/-- Calculates the total maximum draws needed for all colors -/
def totalMaxDraws (box : Box) : Nat :=
  maxDrawsForColor box.red +
  maxDrawsForColor box.green +
  maxDrawsForColor box.orange +
  maxDrawsForColor box.blue +
  maxDrawsForColor box.yellow

/-- The given box with the specified item counts -/
def givenBox : Box :=
  { red := { total := 41 },
    green := { total := 23 },
    orange := { total := 11 },
    blue := { total := 15 },
    yellow := { total := 10 } }

theorem max_draws_for_all_pairs (box : Box := givenBox) :
  totalMaxDraws box = 105 := by
  sorry

end max_draws_for_all_pairs_l3144_314403


namespace sandwich_combinations_l3144_314446

/-- The number of different kinds of lunch meats -/
def num_meats : ℕ := 12

/-- The number of different kinds of cheeses -/
def num_cheeses : ℕ := 8

/-- The number of ways to choose meats for a sandwich -/
def meat_choices : ℕ := Nat.choose num_meats 1 + Nat.choose num_meats 2

/-- The number of ways to choose cheeses for a sandwich -/
def cheese_choices : ℕ := Nat.choose num_cheeses 2

/-- The total number of different sandwiches that can be made -/
def total_sandwiches : ℕ := meat_choices * cheese_choices

theorem sandwich_combinations : total_sandwiches = 2184 := by
  sorry

end sandwich_combinations_l3144_314446


namespace champion_sequences_l3144_314477

/-- The number of letters in CHAMPION -/
def num_letters : ℕ := 8

/-- The number of letters in each sequence -/
def sequence_length : ℕ := 5

/-- The number of letters available for the last position (excluding N) -/
def last_position_options : ℕ := 6

/-- The number of positions to fill after fixing the first and last -/
def middle_positions : ℕ := sequence_length - 2

/-- The number of letters available for the middle positions -/
def middle_options : ℕ := num_letters - 2

theorem champion_sequences :
  (middle_options.factorial / (middle_options - middle_positions).factorial) * last_position_options = 720 := by
  sorry

end champion_sequences_l3144_314477


namespace max_value_quadratic_l3144_314409

theorem max_value_quadratic :
  ∃ (c : ℝ), c = 3395 / 49 ∧ ∀ (r : ℝ), -7 * r^2 + 50 * r - 20 ≤ c := by
  sorry

end max_value_quadratic_l3144_314409


namespace compound_molecular_weight_l3144_314466

/-- Atomic weight of Copper in g/mol -/
def copper_weight : ℝ := 63.546

/-- Atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.011

/-- Atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 15.999

/-- Number of Copper atoms in the compound -/
def copper_count : ℕ := 1

/-- Number of Carbon atoms in the compound -/
def carbon_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def oxygen_count : ℕ := 3

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  copper_count * copper_weight + 
  carbon_count * carbon_weight + 
  oxygen_count * oxygen_weight

/-- Theorem stating that the molecular weight of the compound is 123.554 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight = 123.554 := by sorry

end compound_molecular_weight_l3144_314466


namespace sum_equals_5070_l3144_314459

theorem sum_equals_5070 (P : ℕ) : 
  1010 + 1012 + 1014 + 1016 + 1018 = 5100 - P → P = 30 :=
by sorry

end sum_equals_5070_l3144_314459


namespace sqrt_integer_part_problem_l3144_314431

theorem sqrt_integer_part_problem :
  ∃ n : ℕ, 
    (∀ k : ℕ, k < 35 → ⌊Real.sqrt (n^2 + k)⌋ = n) ∧ 
    (∀ m : ℕ, m > n → ∃ j : ℕ, j < 35 ∧ ⌊Real.sqrt (m^2 + j)⌋ ≠ m) :=
sorry

end sqrt_integer_part_problem_l3144_314431


namespace blocks_color_theorem_l3144_314427

theorem blocks_color_theorem (total_blocks : ℕ) (blocks_per_color : ℕ) (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) :
  total_blocks / blocks_per_color = 7 := by
  sorry

end blocks_color_theorem_l3144_314427


namespace max_value_xyz_l3144_314492

theorem max_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : 
  x^3 * y^3 * z^2 ≤ 4782969/390625 := by
  sorry

end max_value_xyz_l3144_314492


namespace certain_fraction_is_two_fifths_l3144_314486

theorem certain_fraction_is_two_fifths :
  ∀ (x y : ℚ),
    (x ≠ 0 ∧ y ≠ 0) →
    ((1 : ℚ) / 7) / (x / y) = ((3 : ℚ) / 7) / ((6 : ℚ) / 5) →
    x / y = (2 : ℚ) / 5 := by
  sorry

end certain_fraction_is_two_fifths_l3144_314486


namespace sum_ages_after_ten_years_l3144_314450

/-- Given Ann's age and Tom's age relative to Ann's, calculate the sum of their ages after a certain number of years. -/
def sum_ages_after_years (ann_age : ℕ) (tom_age_multiplier : ℕ) (years_later : ℕ) : ℕ :=
  (ann_age + years_later) + (ann_age * tom_age_multiplier + years_later)

/-- Prove that given Ann is 6 years old and Tom is twice her age, the sum of their ages 10 years later will be 38 years. -/
theorem sum_ages_after_ten_years :
  sum_ages_after_years 6 2 10 = 38 := by
  sorry

end sum_ages_after_ten_years_l3144_314450


namespace rotated_line_equation_l3144_314453

/-- Given a line l₁ with equation x - y - 3 = 0 rotated counterclockwise by 15° around
    the point (3,0) to obtain line l₂, the equation of l₂ is √3x - y - 3√3 = 0 --/
theorem rotated_line_equation (x y : ℝ) :
  let l₁ : ℝ → ℝ → Prop := fun x y ↦ x - y - 3 = 0
  let rotation_angle : ℝ := 15 * π / 180
  let rotation_center : ℝ × ℝ := (3, 0)
  let l₂ : ℝ → ℝ → Prop := fun x y ↦
    ∃ (x₀ y₀ : ℝ), l₁ x₀ y₀ ∧
    x - 3 = (x₀ - 3) * Real.cos rotation_angle - (y₀ - 0) * Real.sin rotation_angle ∧
    y - 0 = (x₀ - 3) * Real.sin rotation_angle + (y₀ - 0) * Real.cos rotation_angle
  l₂ x y ↔ Real.sqrt 3 * x - y - 3 * Real.sqrt 3 = 0 := by
  sorry

end rotated_line_equation_l3144_314453


namespace darwin_remaining_money_l3144_314494

/-- Calculates the remaining money after Darwin's expenditures --/
def remaining_money (initial : ℝ) : ℝ :=
  let after_gas := initial * (1 - 0.35)
  let after_food := after_gas * (1 - 0.2)
  let after_clothing := after_food * (1 - 0.25)
  after_clothing * (1 - 0.15)

/-- Theorem stating that Darwin's remaining money is $4,972.50 --/
theorem darwin_remaining_money :
  remaining_money 15000 = 4972.50 := by
  sorry

end darwin_remaining_money_l3144_314494


namespace average_problem_l3144_314493

theorem average_problem (n₁ n₂ : ℕ) (avg_all avg₂ : ℚ) (h₁ : n₁ = 30) (h₂ : n₂ = 20) 
  (h₃ : avg₂ = 30) (h₄ : avg_all = 24) :
  let sum_all := (n₁ + n₂ : ℚ) * avg_all
  let sum₂ := n₂ * avg₂
  let sum₁ := sum_all - sum₂
  sum₁ / n₁ = 20 := by sorry

end average_problem_l3144_314493


namespace roots_expression_simplification_l3144_314451

theorem roots_expression_simplification (p q : ℝ) (α β γ δ : ℝ) 
  (h1 : α^2 + p*α + 2 = 0) 
  (h2 : β^2 + p*β + 2 = 0) 
  (h3 : γ^2 + q*γ + 2 = 0) 
  (h4 : δ^2 + q*δ + 2 = 0) : 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = 2*(p^2 - q^2) := by
  sorry

end roots_expression_simplification_l3144_314451


namespace identical_angular_acceleration_l3144_314491

/-- Two wheels with identical masses and different radii have identical angular accelerations -/
theorem identical_angular_acceleration (m : ℝ) (R₁ R₂ F₁ F₂ : ℝ) 
  (h_m : m = 1)
  (h_R₁ : R₁ = 0.5)
  (h_R₂ : R₂ = 1)
  (h_F₁ : F₁ = 1)
  (h_positive : m > 0 ∧ R₁ > 0 ∧ R₂ > 0 ∧ F₁ > 0 ∧ F₂ > 0) :
  (F₁ * R₁ / (m * R₁^2) = F₂ * R₂ / (m * R₂^2)) → F₂ = 2 := by
  sorry

#check identical_angular_acceleration

end identical_angular_acceleration_l3144_314491


namespace inscribed_quadrilateral_area_l3144_314412

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - x*y + x - 4*y = 12

/-- Point A is on the y-axis and satisfies the ellipse equation -/
def point_A : ℝ × ℝ := sorry

/-- Point C is on the y-axis and satisfies the ellipse equation -/
def point_C : ℝ × ℝ := sorry

/-- Point B is on the x-axis and satisfies the ellipse equation -/
def point_B : ℝ × ℝ := sorry

/-- Point D is on the x-axis and satisfies the ellipse equation -/
def point_D : ℝ × ℝ := sorry

/-- The area of the inscribed quadrilateral ABCD -/
def area_ABCD : ℝ := sorry

theorem inscribed_quadrilateral_area :
  ellipse_equation point_A.1 point_A.2 ∧
  ellipse_equation point_B.1 point_B.2 ∧
  ellipse_equation point_C.1 point_C.2 ∧
  ellipse_equation point_D.1 point_D.2 ∧
  point_A.1 = 0 ∧ point_C.1 = 0 ∧
  point_B.2 = 0 ∧ point_D.2 = 0 →
  area_ABCD = 28 := by sorry

end inscribed_quadrilateral_area_l3144_314412


namespace pi_is_irrational_l3144_314408

-- Define the property of being an infinite non-repeating decimal
def is_infinite_non_repeating_decimal (x : ℝ) : Prop := sorry

-- Define the property of being an irrational number
def is_irrational (x : ℝ) : Prop := sorry

-- Axiom: All irrational numbers are infinite non-repeating decimals
axiom irrational_are_infinite_non_repeating : 
  ∀ x : ℝ, is_irrational x → is_infinite_non_repeating_decimal x

-- Given: π is an infinite non-repeating decimal
axiom pi_is_infinite_non_repeating : is_infinite_non_repeating_decimal Real.pi

-- Theorem to prove
theorem pi_is_irrational : is_irrational Real.pi := sorry

end pi_is_irrational_l3144_314408


namespace trigonometric_identity_l3144_314426

theorem trigonometric_identity : 
  100 * (Real.sin (253 * π / 180) * Real.sin (313 * π / 180) + 
         Real.sin (163 * π / 180) * Real.sin (223 * π / 180)) = 50 := by
  sorry

end trigonometric_identity_l3144_314426


namespace ways_to_soccer_field_l3144_314480

theorem ways_to_soccer_field (walk_ways drive_ways : ℕ) : 
  walk_ways = 3 → drive_ways = 4 → walk_ways + drive_ways = 7 := by
  sorry

end ways_to_soccer_field_l3144_314480


namespace inverse_of_B_cubed_l3144_314424

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = !![3, 4; -2, -3] →
  (B^3)⁻¹ = !![3, 4; -2, -3] :=
by
  sorry

end inverse_of_B_cubed_l3144_314424


namespace proper_fraction_triple_when_cubed_l3144_314402

theorem proper_fraction_triple_when_cubed (a b : ℕ) (h1 : 0 < a) (h2 : a < b) :
  (a^3 : ℚ) / (b + 3) = 3 * (a : ℚ) / b ↔ a = 2 ∧ b = 9 := by
  sorry

end proper_fraction_triple_when_cubed_l3144_314402


namespace inequalities_proof_l3144_314458

theorem inequalities_proof (a b : ℝ) : 
  (a^2 + b^2 ≥ (a + b)^2 / 2) ∧ (a^2 + b^2 ≥ 2*(a - b - 1)) := by
  sorry

end inequalities_proof_l3144_314458


namespace f_of_f_of_3_l3144_314464

def f (x : ℝ) : ℝ := 3 * x^2 + x - 4

theorem f_of_f_of_3 : f (f 3) = 2050 := by
  sorry

end f_of_f_of_3_l3144_314464


namespace parabola_shift_theorem_l3144_314463

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := p.b - 2 * p.a * h,
    c := p.c + p.a * h^2 - p.b * h }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a,
    b := p.b,
    c := p.c + v }

theorem parabola_shift_theorem (x : ℝ) :
  let initial_parabola := Parabola.mk (-2) 0 0
  let shifted_left := shift_horizontal initial_parabola 1
  let final_parabola := shift_vertical shifted_left 3
  final_parabola.a * x^2 + final_parabola.b * x + final_parabola.c = -2 * (x + 1)^2 + 3 := by
  sorry

end parabola_shift_theorem_l3144_314463


namespace lcm_factor_proof_l3144_314437

def is_hcf (a b h : ℕ) : Prop := Nat.gcd a b = h

def is_lcm (a b l : ℕ) : Prop := Nat.lcm a b = l

theorem lcm_factor_proof (A B : ℕ) 
  (h1 : is_hcf A B 23)
  (h2 : A = 322)
  (h3 : ∃ x : ℕ, is_lcm A B (23 * 13 * x)) :
  ∃ x : ℕ, is_lcm A B (23 * 13 * x) ∧ x = 14 := by
  sorry

end lcm_factor_proof_l3144_314437


namespace only_consecutive_primes_fifth_power_difference_prime_l3144_314442

theorem only_consecutive_primes_fifth_power_difference_prime :
  ∀ p q : ℕ,
    Prime p → Prime q → p > q →
    Prime (p^5 - q^5) →
    p = 3 ∧ q = 2 :=
by sorry

end only_consecutive_primes_fifth_power_difference_prime_l3144_314442


namespace kevin_distance_after_seven_leaps_l3144_314443

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Kevin's total distance hopped after n leaps -/
def kevinDistance (n : ℕ) : ℚ :=
  geometricSum (1/4) (3/4) n

/-- Theorem: Kevin's total distance after 7 leaps is 14197/16384 -/
theorem kevin_distance_after_seven_leaps :
  kevinDistance 7 = 14197 / 16384 := by
  sorry

end kevin_distance_after_seven_leaps_l3144_314443


namespace distance_to_origin_l3144_314432

theorem distance_to_origin (a : ℝ) : |a| = 3 → (a - 2 = 1 ∨ a - 2 = -5) := by
  sorry

end distance_to_origin_l3144_314432


namespace triangle_perpendicular_bisector_distance_l3144_314414

/-- Given a triangle ABC with sides a, b, c where b > c, if a line HK perpendicular to BC
    divides the triangle into two equal areas, then the distance CK (from C to the foot of
    the perpendicular) is equal to (1/2) * sqrt(a^2 + b^2 - c^2). -/
theorem triangle_perpendicular_bisector_distance
  (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_order : b > c) (h_equal_areas : ∃ (k : ℝ), k > 0 ∧ k < b ∧ k * (a * k / (2 * b)) = a * b / 4) :
  ∃ (k : ℝ), k = (1/2) * Real.sqrt (a^2 + b^2 - c^2) ∧
              k > 0 ∧ k < b ∧ k * (a * k / (2 * b)) = a * b / 4 := by
  sorry

end triangle_perpendicular_bisector_distance_l3144_314414


namespace cistern_wet_surface_area_l3144_314496

/-- Represents the dimensions of a cistern with an elevated platform --/
structure CisternDimensions where
  length : Real
  width : Real
  waterDepth : Real
  platformLength : Real
  platformWidth : Real
  platformHeight : Real

/-- Calculates the total wet surface area of the cistern --/
def totalWetSurfaceArea (d : CisternDimensions) : Real :=
  let wallArea := 2 * (d.length * d.waterDepth) + 2 * (d.width * d.waterDepth)
  let bottomArea := d.length * d.width
  let submergedHeight := d.waterDepth - d.platformHeight
  let platformSideArea := 2 * (d.platformLength * submergedHeight) + 2 * (d.platformWidth * submergedHeight)
  wallArea + bottomArea + platformSideArea

/-- Theorem stating that the total wet surface area of the given cistern is 63.5 square meters --/
theorem cistern_wet_surface_area :
  let d : CisternDimensions := {
    length := 8,
    width := 4,
    waterDepth := 1.25,
    platformLength := 1,
    platformWidth := 0.5,
    platformHeight := 0.75
  }
  totalWetSurfaceArea d = 63.5 := by
  sorry


end cistern_wet_surface_area_l3144_314496


namespace roof_difference_l3144_314429

theorem roof_difference (width : ℝ) (length : ℝ) (area : ℝ) : 
  width > 0 →
  length = 4 * width →
  area = 588 →
  length * width = area →
  length - width = 21 * Real.sqrt 3 := by
sorry

end roof_difference_l3144_314429


namespace visibility_time_proof_l3144_314465

/-- Alice's walking speed in feet per second -/
def alice_speed : ℝ := 2

/-- Bob's walking speed in feet per second -/
def bob_speed : ℝ := 4

/-- Distance between Alice and Bob's parallel paths in feet -/
def path_distance : ℝ := 300

/-- Diameter of the circular monument in feet -/
def monument_diameter : ℝ := 150

/-- Initial distance between Alice and Bob when the monument first blocks their line of sight -/
def initial_distance : ℝ := 300

/-- Time until Alice and Bob can see each other again -/
def visibility_time : ℝ := 48

theorem visibility_time_proof :
  alice_speed = 2 ∧
  bob_speed = 4 ∧
  path_distance = 300 ∧
  monument_diameter = 150 ∧
  initial_distance = 300 →
  visibility_time = 48 := by
  sorry

#check visibility_time_proof

end visibility_time_proof_l3144_314465


namespace max_value_complex_expression_l3144_314417

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = Real.sqrt 2) :
  Complex.abs ((z - 1)^2 * (z + 1)) ≤ 4 * Real.sqrt 2 ∧
  ∃ w : ℂ, Complex.abs w = Real.sqrt 2 ∧ Complex.abs ((w - 1)^2 * (w + 1)) = 4 * Real.sqrt 2 :=
by sorry

end max_value_complex_expression_l3144_314417


namespace complex_pure_imaginary_m_l3144_314461

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_m (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 4) (m + 2)
  is_pure_imaginary z → m = 2 :=
by sorry

end complex_pure_imaginary_m_l3144_314461


namespace book_arrangement_count_book_arrangement_proof_l3144_314484

theorem book_arrangement_count : Nat :=
  let num_dictionaries : Nat := 3
  let num_novels : Nat := 2
  let dict_arrangements : Nat := Nat.factorial num_dictionaries
  let novel_arrangements : Nat := Nat.factorial num_novels
  let group_arrangements : Nat := Nat.factorial 2
  dict_arrangements * novel_arrangements * group_arrangements

theorem book_arrangement_proof :
  book_arrangement_count = 24 := by
  sorry

end book_arrangement_count_book_arrangement_proof_l3144_314484


namespace probability_two_even_balls_l3144_314420

theorem probability_two_even_balls (n : ℕ) (h1 : n = 16) :
  let total_balls := n
  let even_balls := n / 2
  let prob_first_even := even_balls / total_balls
  let prob_second_even := (even_balls - 1) / (total_balls - 1)
  prob_first_even * prob_second_even = 7 / 30 :=
by
  sorry

end probability_two_even_balls_l3144_314420


namespace bridge_length_calculation_l3144_314475

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) :
  train_length = 240 →
  crossing_time = 20 →
  train_speed_kmh = 70.2 →
  ∃ (bridge_length : ℝ), bridge_length = 150 := by
    sorry


end bridge_length_calculation_l3144_314475


namespace even_square_iff_even_l3144_314440

theorem even_square_iff_even (p : ℕ) : Even p ↔ Even (p^2) := by
  sorry

end even_square_iff_even_l3144_314440


namespace gcd_problem_l3144_314448

theorem gcd_problem : Int.gcd (123^2 + 235^2 - 347^2) (122^2 + 234^2 - 348^2) = 1 := by
  sorry

end gcd_problem_l3144_314448


namespace duck_count_relation_l3144_314439

theorem duck_count_relation :
  ∀ (muscovy cayuga khaki : ℕ),
    muscovy = 39 →
    muscovy = cayuga + 4 →
    muscovy + cayuga + khaki = 90 →
    muscovy = 2 * cayuga - 31 :=
by
  sorry

end duck_count_relation_l3144_314439


namespace rectangle_perimeter_l3144_314405

theorem rectangle_perimeter (a b : ℝ) (h : a * b > 2 * a + 2 * b) : 2 * a + 2 * b > 16 := by
  sorry

end rectangle_perimeter_l3144_314405


namespace makeup_exam_probability_l3144_314478

/-- Given a class with a total number of students and a number of students who need to take a makeup exam,
    calculate the probability of a student participating in the makeup exam. -/
theorem makeup_exam_probability (total_students : ℕ) (makeup_students : ℕ) 
    (h1 : total_students = 42) (h2 : makeup_students = 3) :
    (makeup_students : ℚ) / total_students = 1 / 14 := by
  sorry

#check makeup_exam_probability

end makeup_exam_probability_l3144_314478


namespace debate_team_girls_l3144_314472

theorem debate_team_girls (boys : ℕ) (groups : ℕ) (members_per_group : ℕ) : 
  boys = 26 → groups = 8 → members_per_group = 9 → 
  (groups * members_per_group) - boys = 46 := by sorry

end debate_team_girls_l3144_314472


namespace sequence_sum_l3144_314467

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem sequence_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = (1 : ℝ) * (1 / 3) ^ n) →
  (a 1 = 1) →
  ∀ n : ℕ, n ≥ 1 → a n = (3 / 2) * (1 - (1 / 3) ^ n) :=
by sorry

end sequence_sum_l3144_314467


namespace hyperbola_a_plus_h_value_l3144_314455

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  pos_a : a > 0
  pos_b : b > 0

/-- The asymptotes of the hyperbola -/
def asymptotes (slope : ℝ) (y_intercept1 y_intercept2 : ℝ) :=
  (fun x => slope * x + y_intercept1, fun x => -slope * x + y_intercept2)

theorem hyperbola_a_plus_h_value
  (slope : ℝ)
  (y_intercept1 y_intercept2 : ℝ)
  (point_x point_y : ℝ)
  (h : Hyperbola)
  (asym : asymptotes slope y_intercept1 y_intercept2 = 
    (fun x => 3 * x + 4, fun x => -3 * x + 2))
  (point_on_hyperbola : (point_x, point_y) = (1, 8))
  (hyperbola_eq : ∀ x y, 
    (y - h.k)^2 / h.a^2 - (x - h.h)^2 / h.b^2 = 1 ↔ 
    (fun x y => (y - h.k)^2 / h.a^2 - (x - h.h)^2 / h.b^2 = 1) x y) :
  h.a + h.h = 8/3 := by
  sorry

end hyperbola_a_plus_h_value_l3144_314455


namespace gcd_of_78_and_182_l3144_314430

theorem gcd_of_78_and_182 : Nat.gcd 78 182 = 26 := by
  sorry

end gcd_of_78_and_182_l3144_314430


namespace gcd_of_squares_sum_l3144_314422

theorem gcd_of_squares_sum : Nat.gcd (168^2 + 301^2 + 502^2) (169^2 + 300^2 + 501^2) = 1 := by
  sorry

end gcd_of_squares_sum_l3144_314422


namespace central_square_illumination_l3144_314407

theorem central_square_illumination (n : ℕ) (h_odd : Odd n) :
  ∃ (min_lamps : ℕ),
    min_lamps = (n + 1)^2 / 2 ∧
    (∀ (lamps : ℕ),
      (∀ (i j : ℕ), i ≤ n ∧ j ≤ n →
        ∃ (k₁ k₂ : ℕ), k₁ ≠ k₂ ∧ k₁ ≤ lamps ∧ k₂ ≤ lamps ∧
          ((i = 0 ∨ i = n ∨ j = 0 ∨ j = n) →
            (k₁ ≤ 4 ∧ k₂ ≤ 4))) →
      lamps ≥ min_lamps) :=
by sorry

end central_square_illumination_l3144_314407


namespace sequence_properties_l3144_314471

def sequence_a (n : ℕ) : ℝ := sorry

def sum_S (n : ℕ) : ℝ := sorry

axiom a_def (n : ℕ) : n ≠ 0 → sequence_a n = 2 * sum_S n - 1

def sequence_b (n : ℕ) : ℝ := (2 * n + 1) * sequence_a n

def sum_T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, n ≠ 0 → sequence_a n = (-1)^(n-1)) ∧
  (∀ n : ℕ, sum_T n = 1 - (n + 1) * (-1)^n) :=
by sorry

end sequence_properties_l3144_314471


namespace larger_number_proof_l3144_314462

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1335) (h3 : L = 6 * S + 15) :
  L = 1599 := by
sorry

end larger_number_proof_l3144_314462


namespace most_likely_parent_genotypes_l3144_314433

/-- Represents the alleles for rabbit fur type -/
inductive Allele
| H  -- Dominant hairy
| h  -- Recessive hairy
| S  -- Dominant smooth
| s  -- Recessive smooth

/-- Represents the genotype of a rabbit -/
structure Genotype :=
(allele1 : Allele)
(allele2 : Allele)

/-- Represents the phenotype (observable trait) of a rabbit -/
inductive Phenotype
| Hairy
| Smooth

/-- Function to determine the phenotype from a genotype -/
def phenotypeFromGenotype (g : Genotype) : Phenotype :=
  match g.allele1, g.allele2 with
  | Allele.H, _ => Phenotype.Hairy
  | _, Allele.H => Phenotype.Hairy
  | Allele.S, _ => Phenotype.Smooth
  | _, Allele.S => Phenotype.Smooth
  | Allele.h, Allele.h => Phenotype.Hairy
  | Allele.s, Allele.s => Phenotype.Smooth
  | _, _ => Phenotype.Smooth

/-- The probability of the hairy allele in the population -/
def hairyAlleleProbability : ℝ := 0.1

/-- Theorem stating the most likely genotype combination for parents -/
theorem most_likely_parent_genotypes
  (hairyParent smoothParent : Genotype)
  (allOffspringHairy : ∀ (offspring : Genotype),
    phenotypeFromGenotype offspring = Phenotype.Hairy) :
  (hairyParent = ⟨Allele.H, Allele.H⟩ ∧
   smoothParent = ⟨Allele.S, Allele.h⟩) ∨
  (hairyParent = ⟨Allele.H, Allele.H⟩ ∧
   smoothParent = ⟨Allele.h, Allele.S⟩) :=
sorry


end most_likely_parent_genotypes_l3144_314433


namespace solution_satisfies_system_l3144_314489

theorem solution_satisfies_system : ∃ (x y : ℝ), 
  (y = 2 - x ∧ 3 * x = 1 + 2 * y) ∧ (x = 1 ∧ y = 1) := by
  sorry

end solution_satisfies_system_l3144_314489


namespace greatest_x_under_conditions_l3144_314483

theorem greatest_x_under_conditions (x : ℕ) : 
  x % 4 = 0 → 
  x > 0 → 
  x^3 < 8000 → 
  x ≤ 16 ∧ 
  ∀ y : ℕ, y % 4 = 0 → y > 0 → y^3 < 8000 → y ≤ x :=
by sorry

end greatest_x_under_conditions_l3144_314483


namespace circular_board_area_l3144_314418

/-- The area of a circular wooden board that rolls forward for 10 revolutions
    and advances exactly 62.8 meters is π square meters. -/
theorem circular_board_area (revolutions : ℕ) (distance : ℝ) (area : ℝ) :
  revolutions = 10 →
  distance = 62.8 →
  area = π →
  area = (distance / (2 * revolutions : ℝ))^2 * π := by
  sorry

end circular_board_area_l3144_314418


namespace log_simplification_l3144_314488

theorem log_simplification (a b c d x y : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hx : 0 < x) (hy : 0 < y) : 
  Real.log (a / b) + Real.log (b / c) + Real.log (c / d) - Real.log ((a * y) / (d * x)) = Real.log (x / y) := by
  sorry

end log_simplification_l3144_314488


namespace sum_of_reciprocals_l3144_314481

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (sum_eq_10 : x + y = 10) (sum_eq_5prod : x + y = 5 * x * y) : 
  1 / x + 1 / y = 5 := by
  sorry

end sum_of_reciprocals_l3144_314481


namespace vector_parallelism_l3144_314415

theorem vector_parallelism (m : ℚ) : 
  let a : Fin 2 → ℚ := ![(-1), 2]
  let b : Fin 2 → ℚ := ![m, 1]
  (∃ (k : ℚ), k ≠ 0 ∧ (a + 2 • b) = k • (2 • a - b)) → m = -1/2 := by
  sorry

end vector_parallelism_l3144_314415


namespace negation_equivalence_l3144_314452

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ≥ 3 ∧ x^2 - 2*x + 3 < 0) ↔ (∀ x : ℝ, x ≥ 3 → x^2 - 2*x + 3 ≥ 0) := by
  sorry

end negation_equivalence_l3144_314452


namespace constant_term_expansion_l3144_314469

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the kth term in the expansion -/
def coeff (n k : ℕ) : ℕ := binomial n k * 2^k

theorem constant_term_expansion (n : ℕ) 
  (h : coeff n 4 / coeff n 2 = 56 / 3) : 
  coeff n 2 = 180 := by sorry

end constant_term_expansion_l3144_314469


namespace age_comparison_l3144_314428

theorem age_comparison (present_age : ℕ) (years_ago : ℕ) : 
  (present_age = 50) →
  (present_age = (125 * (present_age - years_ago)) / 100) →
  (present_age = (250 * present_age) / (250 + 50)) →
  (years_ago = 10) := by
sorry


end age_comparison_l3144_314428


namespace poster_ratio_l3144_314435

theorem poster_ratio (total medium large small : ℕ) : 
  total = 50 ∧ 
  medium = total / 2 ∧ 
  large = 5 ∧ 
  small = total - medium - large → 
  small * 5 = total * 2 := by
sorry

end poster_ratio_l3144_314435


namespace subset_polynomial_equivalence_l3144_314410

theorem subset_polynomial_equivalence (n : ℕ) (h : n > 4) :
  (∀ (A B : Set (Fin n)), ∃ (f : Polynomial ℤ),
    (∀ a ∈ A, ∃ b ∈ B, f.eval a ≡ b [ZMOD n]) ∨
    (∀ b ∈ B, ∃ a ∈ A, f.eval b ≡ a [ZMOD n])) ↔
  Nat.Prime n := by
  sorry

end subset_polynomial_equivalence_l3144_314410


namespace max_rented_trucks_l3144_314457

theorem max_rented_trucks (total_trucks : ℕ) (return_rate : ℚ) (min_saturday_trucks : ℕ) :
  total_trucks = 20 →
  return_rate = 1/2 →
  min_saturday_trucks = 10 →
  ∃ (max_rented : ℕ), max_rented ≤ total_trucks ∧
    max_rented * return_rate = total_trucks - min_saturday_trucks ∧
    ∀ (rented : ℕ), rented ≤ total_trucks ∧ 
      rented * return_rate = total_trucks - min_saturday_trucks →
      rented ≤ max_rented :=
by sorry

end max_rented_trucks_l3144_314457


namespace rectangle_rotation_path_length_l3144_314421

/-- The length of the path traveled by point A of a rectangle ABCD when rotated as described -/
theorem rectangle_rotation_path_length (AB CD BC AD : ℝ) (h1 : AB = 4) (h2 : CD = 4) (h3 : BC = 8) (h4 : AD = 8) :
  let diagonal := Real.sqrt (AB ^ 2 + AD ^ 2)
  let first_rotation_arc := (π / 2) * diagonal
  let second_rotation_arc := (π / 2) * diagonal
  first_rotation_arc + second_rotation_arc = 4 * Real.sqrt 5 * π :=
by sorry

end rectangle_rotation_path_length_l3144_314421


namespace adeline_work_hours_l3144_314411

/-- Adeline's work schedule and earnings problem -/
theorem adeline_work_hours
  (hourly_rate : ℕ)
  (days_per_week : ℕ)
  (total_earnings : ℕ)
  (total_weeks : ℕ)
  (h1 : hourly_rate = 12)
  (h2 : days_per_week = 5)
  (h3 : total_earnings = 3780)
  (h4 : total_weeks = 7) :
  total_earnings / (total_weeks * days_per_week * hourly_rate) = 9 :=
by sorry

end adeline_work_hours_l3144_314411


namespace ball_radius_from_hole_dimensions_l3144_314460

/-- 
Given a spherical ball that leaves a circular hole when removed from a frozen surface,
this theorem proves that if the hole has a diameter of 30 cm and a depth of 10 cm,
then the radius of the ball is 16.25 cm.
-/
theorem ball_radius_from_hole_dimensions (hole_diameter : ℝ) (hole_depth : ℝ) 
    (h_diameter : hole_diameter = 30) 
    (h_depth : hole_depth = 10) : 
    ∃ (ball_radius : ℝ), ball_radius = 16.25 := by
  sorry

end ball_radius_from_hole_dimensions_l3144_314460


namespace average_of_abc_l3144_314497

theorem average_of_abc (A B C : ℚ) : 
  A = 2 → 
  2002 * C - 1001 * A = 8008 → 
  2002 * B + 3003 * A = 7007 → 
  (A + B + C) / 3 = 7 / 3 := by
sorry

end average_of_abc_l3144_314497


namespace perpendicular_lines_a_values_l3144_314419

theorem perpendicular_lines_a_values (a : ℝ) : 
  (∃ (x y : ℝ), ax + 2*y + 6 = 0 ∧ x + a*(a+1)*y + (a^2 - 1) = 0) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    ax₁ + 2*y₁ + 6 = 0 ∧ 
    x₂ + a*(a+1)*y₂ + (a^2 - 1) = 0 →
    (x₂ - x₁) * (ax₁ + 2*y₁) + (y₂ - y₁) * (2*x₁ - 2*a*y₁) = 0) →
  a = 0 ∨ a = -3/2 := by
sorry

end perpendicular_lines_a_values_l3144_314419


namespace smallest_multiple_17_6_more_than_multiple_73_l3144_314468

theorem smallest_multiple_17_6_more_than_multiple_73 : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), n = 17 * k) ∧ 
  (∃ (m : ℕ), n = 73 * m + 6) ∧
  (∀ (x : ℕ), x > 0 → (∃ (k : ℕ), x = 17 * k) → (∃ (m : ℕ), x = 73 * m + 6) → x ≥ n) ∧
  n = 663 := by
sorry

end smallest_multiple_17_6_more_than_multiple_73_l3144_314468


namespace opposite_grey_is_violet_l3144_314425

-- Define the colors
inductive Color
| Yellow
| Grey
| Orange
| Violet
| Blue
| Black

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  faces : Fin 6 → Face

-- Define a view of the cube
structure View where
  top : Color
  front : Color
  right : Color

-- Define the given views
def view1 : View := { top := Color.Yellow, front := Color.Blue, right := Color.Black }
def view2 : View := { top := Color.Orange, front := Color.Yellow, right := Color.Black }
def view3 : View := { top := Color.Orange, front := Color.Violet, right := Color.Black }

-- Theorem statement
theorem opposite_grey_is_violet (c : Cube) 
  (h1 : ∃ (f1 f2 f3 : Fin 6), c.faces f1 = { color := view1.top } ∧ 
                               c.faces f2 = { color := view1.front } ∧ 
                               c.faces f3 = { color := view1.right })
  (h2 : ∃ (f1 f2 f3 : Fin 6), c.faces f1 = { color := view2.top } ∧ 
                               c.faces f2 = { color := view2.front } ∧ 
                               c.faces f3 = { color := view2.right })
  (h3 : ∃ (f1 f2 f3 : Fin 6), c.faces f1 = { color := view3.top } ∧ 
                               c.faces f2 = { color := view3.front } ∧ 
                               c.faces f3 = { color := view3.right })
  (h4 : ∃! (f : Fin 6), c.faces f = { color := Color.Grey }) :
  ∃ (f1 f2 : Fin 6), c.faces f1 = { color := Color.Grey } ∧ 
                     c.faces f2 = { color := Color.Violet } ∧ 
                     f1 ≠ f2 ∧ 
                     ∀ (f3 : Fin 6), f3 ≠ f1 ∧ f3 ≠ f2 → 
                       (c.faces f3).color ≠ Color.Grey ∧ (c.faces f3).color ≠ Color.Violet :=
by
  sorry


end opposite_grey_is_violet_l3144_314425


namespace least_valid_number_l3144_314487

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d : ℕ) (p : ℕ), 
    d ≥ 1 ∧ d ≤ 9 ∧
    n = 10^p * d + (n % 10^p) ∧
    10^p * d + (n % 10^p) = 17 * (n % 10^p)

theorem least_valid_number : 
  is_valid_number 10625 ∧ 
  ∀ (m : ℕ), m < 10625 → ¬(is_valid_number m) :=
sorry

end least_valid_number_l3144_314487


namespace circle_radius_from_arc_and_angle_l3144_314470

/-- 
Given a circle where an arc of length 5π cm corresponds to a central angle of 150°, 
the radius of the circle is 6 cm.
-/
theorem circle_radius_from_arc_and_angle : 
  ∀ (r : ℝ), 
  (150 / 180 : ℝ) * Real.pi * r = 5 * Real.pi → 
  r = 6 := by
  sorry

end circle_radius_from_arc_and_angle_l3144_314470


namespace sushil_marks_proof_l3144_314490

def total_marks (english science maths : ℕ) : ℕ := english + science + maths

theorem sushil_marks_proof (english science maths : ℕ) :
  english = 3 * science →
  english = maths / 4 →
  science = 17 →
  total_marks english science maths = 272 :=
by
  sorry

end sushil_marks_proof_l3144_314490


namespace katie_marbles_l3144_314449

theorem katie_marbles (pink : ℕ) (orange : ℕ) (purple : ℕ) 
  (h1 : pink = 13)
  (h2 : orange = pink - 9)
  (h3 : purple = 4 * orange) :
  pink + orange + purple = 33 := by
sorry

end katie_marbles_l3144_314449


namespace four_circles_theorem_l3144_314416

/-- Represents a square piece of paper -/
structure Paper :=
  (side : ℝ)
  (is_square : side > 0)

/-- Represents the state of the paper after folding and corner removal -/
structure FoldedPaper :=
  (original : Paper)
  (num_folds : ℕ)
  (corner_removed : Bool)

/-- Calculates the number of layers after folding -/
def num_layers (fp : FoldedPaper) : ℕ :=
  2^(fp.num_folds)

/-- Represents the hole pattern after unfolding -/
structure HolePattern :=
  (num_circles : ℕ)

/-- Function to determine the hole pattern after unfolding -/
def unfold_pattern (fp : FoldedPaper) : HolePattern :=
  { num_circles := if fp.corner_removed then (num_layers fp) / 4 else 0 }

theorem four_circles_theorem (p : Paper) :
  let fp := FoldedPaper.mk p 4 true
  (unfold_pattern fp).num_circles = 4 := by sorry

end four_circles_theorem_l3144_314416


namespace sophie_germain_identity_l3144_314404

theorem sophie_germain_identity (a b : ℝ) : 
  a^4 + 4*b^4 = (a^2 + 2*a*b + 2*b^2) * (a^2 - 2*a*b + 2*b^2) := by
  sorry

end sophie_germain_identity_l3144_314404


namespace rogers_initial_money_l3144_314498

theorem rogers_initial_money (game_cost toy_cost num_toys : ℕ) 
  (h1 : game_cost = 48)
  (h2 : toy_cost = 3)
  (h3 : num_toys = 5)
  (h4 : ∃ (remaining : ℕ), remaining = num_toys * toy_cost) :
  game_cost + num_toys * toy_cost = 63 := by
sorry

end rogers_initial_money_l3144_314498


namespace special_ellipse_ratio_l3144_314400

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  -- Semi-major axis
  a : ℝ
  -- Semi-minor axis
  b : ℝ
  -- Distance from center to focus
  c : ℝ
  -- Ensure a > b > 0 and c > 0
  h1 : a > b
  h2 : b > 0
  h3 : c > 0
  -- Ellipse equation: a² = b² + c²
  h4 : a^2 = b^2 + c^2
  -- Special condition: |F1B2|² = |OF1| * |B1B2|
  h5 : (a + c)^2 = c * (2 * b)

/-- The ratio of semi-major axis to center-focus distance is 3:2 -/
theorem special_ellipse_ratio (e : SpecialEllipse) : a / c = 3 / 2 := by
  sorry

end special_ellipse_ratio_l3144_314400


namespace division_problem_l3144_314413

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 23)
  (h2 : divisor = 4)
  (h3 : remainder = 3)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 5 := by
sorry

end division_problem_l3144_314413


namespace three_pairs_product_l3144_314445

theorem three_pairs_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 1005) (h₂ : y₁^3 - 3*x₁^2*y₁ = 1004)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 1005) (h₄ : y₂^3 - 3*x₂^2*y₂ = 1004)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 1005) (h₆ : y₃^3 - 3*x₃^2*y₃ = 1004) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/502 := by
  sorry

end three_pairs_product_l3144_314445


namespace extreme_points_theorem_l3144_314423

open Real

/-- The function f(x) = x ln x - ax^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x - a * x^2

/-- Predicate indicating that f has two extreme points -/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 ↔ (x = x₁ ∨ x = x₂))

/-- The main theorem -/
theorem extreme_points_theorem :
  (∀ a : ℝ, has_two_extreme_points a → 0 < a ∧ a < 1/2) ∧
  (∃ a : ℝ, has_two_extreme_points a ∧
    ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
      (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
      x₁ + x₂ = x₂ / x₁) :=
sorry

end extreme_points_theorem_l3144_314423


namespace stratified_sample_male_teachers_l3144_314406

theorem stratified_sample_male_teachers 
  (male_teachers : ℕ) 
  (female_teachers : ℕ) 
  (sample_size : ℕ) 
  (h1 : male_teachers = 56) 
  (h2 : female_teachers = 42) 
  (h3 : sample_size = 14) : 
  ℕ :=
by
  -- The proof goes here
  sorry

#check stratified_sample_male_teachers

end stratified_sample_male_teachers_l3144_314406


namespace coin_flip_expected_value_l3144_314434

/-- The expected value of flipping a set of coins -/
def expected_value (coin_values : List ℚ) : ℚ :=
  (coin_values.map (· / 2)).sum

/-- Theorem: The expected value of flipping a penny, nickel, dime, quarter, and half-dollar is 45.5 cents -/
theorem coin_flip_expected_value :
  expected_value [1, 5, 10, 25, 50] = 91/2 := by
  sorry

end coin_flip_expected_value_l3144_314434


namespace line_segment_polar_equation_l3144_314441

theorem line_segment_polar_equation :
  ∀ (x y ρ θ : ℝ),
  (y = 1 - x ∧ 0 ≤ x ∧ x ≤ 1) ↔
  (ρ = 1 / (Real.cos θ + Real.sin θ) ∧ 0 ≤ θ ∧ θ ≤ Real.pi / 2) :=
by sorry

end line_segment_polar_equation_l3144_314441


namespace steiner_symmetrization_preserves_convexity_l3144_314401

-- Define a convex polygon
def ConvexPolygon (M : Set (ℝ × ℝ)) : Prop := sorry

-- Define Steiner symmetrization
def SteinerSymmetrization (M : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem steiner_symmetrization_preserves_convexity
  (M : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)) :
  ConvexPolygon M →
  ConvexPolygon (SteinerSymmetrization M l) := by
  sorry

end steiner_symmetrization_preserves_convexity_l3144_314401


namespace base_subtraction_l3144_314473

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The problem statement -/
theorem base_subtraction :
  let base9_321 := [3, 2, 1]
  let base6_254 := [2, 5, 4]
  (toBase10 base9_321 9) - (toBase10 base6_254 6) = 156 := by
  sorry

end base_subtraction_l3144_314473


namespace factorization_equality_l3144_314444

theorem factorization_equality (a : ℝ) : 2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2 := by
  sorry

end factorization_equality_l3144_314444


namespace min_women_proof_l3144_314456

/-- The probability of at least 4 men standing together given x women -/
def probability (x : ℕ) : ℚ :=
  (2 * Nat.choose (x + 1) 2 + (x + 1)) / (Nat.choose (x + 1) 3 + 3 * Nat.choose (x + 1) 2 + (x + 1))

/-- The minimum number of women required -/
def min_women : ℕ := 594

theorem min_women_proof :
  ∀ x : ℕ, x ≥ min_women ↔ probability x ≤ 1/100 := by
  sorry

#check min_women_proof

end min_women_proof_l3144_314456


namespace rectangle_shorter_side_length_l3144_314436

theorem rectangle_shorter_side_length 
  (perimeter : ℝ) 
  (longer_side : ℝ) 
  (h1 : perimeter = 100) 
  (h2 : longer_side = 28) : 
  (perimeter - 2 * longer_side) / 2 = 22 :=
by sorry

end rectangle_shorter_side_length_l3144_314436


namespace xiaofang_final_score_l3144_314438

/-- Calculates the final score in a speech contest given the scores and weights for each category -/
def calculate_final_score (speech_content_score : ℝ) (language_expression_score : ℝ) (overall_effect_score : ℝ) 
  (speech_content_weight : ℝ) (language_expression_weight : ℝ) (overall_effect_weight : ℝ) : ℝ :=
  speech_content_score * speech_content_weight + 
  language_expression_score * language_expression_weight + 
  overall_effect_score * overall_effect_weight

/-- Theorem stating that Xiaofang's final score is 90 points -/
theorem xiaofang_final_score : 
  calculate_final_score 85 95 90 0.4 0.4 0.2 = 90 := by
  sorry

end xiaofang_final_score_l3144_314438


namespace odd_numbers_between_300_and_700_l3144_314485

def count_odd_numbers (lower upper : ℕ) : ℕ :=
  (upper - lower - 1 + (lower % 2)) / 2

theorem odd_numbers_between_300_and_700 :
  count_odd_numbers 300 700 = 200 := by
  sorry

end odd_numbers_between_300_and_700_l3144_314485


namespace finite_decimal_is_rational_l3144_314476

theorem finite_decimal_is_rational (x : ℝ) (h : ∃ (n : ℕ) (m : ℤ), x = m / (10 ^ n)) : 
  ∃ (p q : ℤ), x = p / q ∧ q ≠ 0 :=
sorry

end finite_decimal_is_rational_l3144_314476


namespace puppies_sold_is_24_l3144_314454

/-- Represents the pet store scenario --/
structure PetStore where
  initial_puppies : ℕ
  puppies_per_cage : ℕ
  cages_used : ℕ

/-- Calculates the number of puppies sold --/
def puppies_sold (store : PetStore) : ℕ :=
  store.initial_puppies - (store.puppies_per_cage * store.cages_used)

/-- Theorem stating that 24 puppies were sold --/
theorem puppies_sold_is_24 :
  ∃ (store : PetStore),
    store.initial_puppies = 56 ∧
    store.puppies_per_cage = 4 ∧
    store.cages_used = 8 ∧
    puppies_sold store = 24 := by
  sorry

end puppies_sold_is_24_l3144_314454


namespace total_money_divided_l3144_314495

def money_division (maya annie saiji : ℕ) : Prop :=
  maya = annie / 2 ∧ annie = saiji / 2 ∧ saiji = 400

theorem total_money_divided : 
  ∀ maya annie saiji : ℕ, 
  money_division maya annie saiji → 
  maya + annie + saiji = 700 := by
sorry

end total_money_divided_l3144_314495


namespace quadratic_roots_condition_l3144_314479

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x + c = 0 ↔ x = (-3 + Real.sqrt 7) / 2 ∨ x = (-3 - Real.sqrt 7) / 2) → 
  c = 1/2 := by
sorry

end quadratic_roots_condition_l3144_314479


namespace sara_lunch_bill_l3144_314482

/-- The cost of Sara's lunch given the prices of a hotdog and a salad -/
def lunch_bill (hotdog_price salad_price : ℚ) : ℚ :=
  hotdog_price + salad_price

/-- Theorem stating that Sara's lunch bill is the sum of the hotdog and salad prices -/
theorem sara_lunch_bill :
  lunch_bill 5.36 5.10 = 10.46 :=
by sorry

end sara_lunch_bill_l3144_314482

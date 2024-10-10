import Mathlib

namespace point_arrangement_theorem_l1227_122729

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- A set of n points in a plane satisfying the given condition -/
structure PointSet where
  n : ℕ
  points : Fin n → Point
  angle_condition : ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    (angle (points i) (points j) (points k) > 120) ∨
    (angle (points j) (points k) (points i) > 120) ∨
    (angle (points k) (points i) (points j) > 120)

/-- The main theorem -/
theorem point_arrangement_theorem (ps : PointSet) :
  ∃ (σ : Fin ps.n ↪ Fin ps.n),
    ∀ (i j k : Fin ps.n), i < j → j < k →
      angle (ps.points (σ i)) (ps.points (σ j)) (ps.points (σ k)) > 120 := by sorry

end point_arrangement_theorem_l1227_122729


namespace triple_hash_72_l1227_122769

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.5 * N - 1

-- Theorem statement
theorem triple_hash_72 : hash (hash (hash 72)) = 7.25 := by
  sorry

end triple_hash_72_l1227_122769


namespace min_value_expression_l1227_122741

theorem min_value_expression (x y : ℝ) :
  Real.sqrt (4 + y^2) + Real.sqrt (x^2 + y^2 - 4*x - 4*y + 8) + Real.sqrt (x^2 - 8*x + 17) ≥ 5 := by
  sorry

end min_value_expression_l1227_122741


namespace brownies_per_person_l1227_122766

/-- Given a pan of brownies cut into columns and rows, calculate how many brownies each person can eat. -/
theorem brownies_per_person 
  (columns : ℕ) 
  (rows : ℕ) 
  (people : ℕ) 
  (h1 : columns = 6) 
  (h2 : rows = 3) 
  (h3 : people = 6) 
  : (columns * rows) / people = 3 := by
  sorry

end brownies_per_person_l1227_122766


namespace indefinite_integral_proof_l1227_122705

theorem indefinite_integral_proof (x : ℝ) (h : x > 0) : 
  (deriv (λ x => -3 * (1 + x^(4/3))^(5/3) / (4 * x^(4/3))) x) = 
    (1 + x^(4/3))^(2/3) / (x^2 * x^(1/3)) := by
  sorry

end indefinite_integral_proof_l1227_122705


namespace fraction_equality_l1227_122750

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / ((a : ℚ) + 35) = 865 / 1000 → a = 225 := by
  sorry

end fraction_equality_l1227_122750


namespace race_theorem_l1227_122763

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ

/-- Represents a race between two runners -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner

/-- The result of the first race -/
def first_race_result (r : Race) (d : ℝ) : Prop :=
  r.distance / r.runner_b.speed = (r.distance - d) / r.runner_a.speed

/-- The theorem to be proved -/
theorem race_theorem (h d : ℝ) (r : Race) 
  (h_pos : h > 0)
  (d_pos : d > 0)
  (first_race : first_race_result r d)
  (h_eq : r.distance = h) :
  let second_race_time := (h + d/2) / r.runner_a.speed
  let second_race_b_distance := second_race_time * r.runner_b.speed
  h - second_race_b_distance = d * (d + h) / (2 * h) := by
  sorry

end race_theorem_l1227_122763


namespace curve_self_intersection_l1227_122711

/-- A curve defined by parametric equations x = t^2 + 1 and y = t^4 - 9t^2 + 6 -/
def curve (t : ℝ) : ℝ × ℝ :=
  (t^2 + 1, t^4 - 9*t^2 + 6)

/-- The theorem stating that the curve crosses itself at (10, 6) -/
theorem curve_self_intersection :
  ∃ (t1 t2 : ℝ), t1 ≠ t2 ∧ curve t1 = curve t2 ∧ curve t1 = (10, 6) := by
  sorry

end curve_self_intersection_l1227_122711


namespace three_digit_number_proof_l1227_122713

def is_geometric_progression (a b c : ℕ) : Prop :=
  b * b = a * c

def swap_hundreds_units (abc : ℕ) : ℕ :=
  let a := abc / 100
  let b := (abc / 10) % 10
  let c := abc % 10
  c * 100 + b * 10 + a

def last_two_digits (abc : ℕ) : ℕ :=
  abc % 100

def swap_last_two_digits (abc : ℕ) : ℕ :=
  let b := (abc / 10) % 10
  let c := abc % 10
  c * 10 + b

theorem three_digit_number_proof (abc : ℕ) :
  abc ≥ 100 ∧ abc < 1000 ∧
  is_geometric_progression (abc / 100) ((abc / 10) % 10) (abc % 10) ∧
  swap_hundreds_units abc = abc - 594 ∧
  swap_last_two_digits (last_two_digits abc) = last_two_digits abc - 18 →
  abc = 842 := by
  sorry

end three_digit_number_proof_l1227_122713


namespace perfect_square_proof_l1227_122720

theorem perfect_square_proof (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ((2 * l - n - k) * (2 * l - n + k)) / 2 = (l - n)^2 := by
  sorry

end perfect_square_proof_l1227_122720


namespace greatest_integer_satisfying_inequality_l1227_122778

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℕ+, x ≤ 3 ↔ (x : ℝ)^4 / (x : ℝ)^2 < 15 :=
by sorry

end greatest_integer_satisfying_inequality_l1227_122778


namespace cubic_equation_solution_l1227_122797

theorem cubic_equation_solution (y : ℝ) (h : y ≠ 0) :
  (3 * y)^5 = (9 * y)^4 ↔ y = 27 := by
  sorry

end cubic_equation_solution_l1227_122797


namespace shaded_area_proof_l1227_122770

theorem shaded_area_proof (square_side : ℝ) (triangle_side : ℝ) : 
  square_side = 40 →
  triangle_side = 25 →
  square_side^2 - 2 * (1/2 * triangle_side^2) = 975 :=
by sorry

end shaded_area_proof_l1227_122770


namespace sector_arc_length_l1227_122706

/-- Given a sector with central angle 1 radian and radius 5 cm, the arc length is 5 cm. -/
theorem sector_arc_length (θ : Real) (r : Real) (l : Real) : 
  θ = 1 → r = 5 → l = r * θ → l = 5 := by
  sorry

end sector_arc_length_l1227_122706


namespace range_of_w_l1227_122717

theorem range_of_w (x y : ℝ) (h : 2*x^2 + 4*x*y + 2*y^2 + x^2*y^2 = 9) :
  let w := 2*Real.sqrt 2*(x + y) + x*y
  ∃ (a b : ℝ), a = -3*Real.sqrt 5 ∧ b = Real.sqrt 5 ∧ 
    (∀ w', w' = w → a ≤ w' ∧ w' ≤ b) ∧
    (∃ w₁ w₂, w₁ = w ∧ w₂ = w ∧ w₁ = a ∧ w₂ = b) :=
by sorry


end range_of_w_l1227_122717


namespace percentage_problem_l1227_122718

theorem percentage_problem (x y P : ℝ) 
  (h1 : 0.3 * (x - y) = (P / 100) * (x + y))
  (h2 : y = 0.2 * x) : 
  P = 20 := by
sorry

end percentage_problem_l1227_122718


namespace parabola_translation_l1227_122758

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = x^2 --/
def original_parabola : Parabola := { a := 1, b := 0, c := 0 }

/-- Translates a parabola vertically --/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + d }

/-- Translates a parabola horizontally --/
def translate_horizontal (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * d + p.b, c := p.a * d^2 - p.b * d + p.c }

/-- The resulting parabola after translations --/
def result_parabola : Parabola :=
  translate_horizontal (translate_vertical original_parabola 3) 5

theorem parabola_translation :
  result_parabola.a = 1 ∧
  result_parabola.b = -10 ∧
  result_parabola.c = 28 := by
  sorry

#check parabola_translation

end parabola_translation_l1227_122758


namespace investment_triple_period_l1227_122726

/-- The annual interest rate as a real number -/
def r : ℝ := 0.341

/-- The condition for the investment to more than triple -/
def triple_condition (t : ℝ) : Prop := (1 + r) ^ t > 3

/-- The smallest investment period in years -/
def smallest_period : ℕ := 4

theorem investment_triple_period :
  (∀ t : ℝ, t < smallest_period → ¬(triple_condition t)) ∧
  (triple_condition (smallest_period : ℝ)) :=
sorry

end investment_triple_period_l1227_122726


namespace irrigation_canal_construction_l1227_122752

/-- Irrigation Canal Construction Problem -/
theorem irrigation_canal_construction
  (total_length : ℝ)
  (team_b_extra : ℝ)
  (time_ratio : ℝ)
  (cost_a : ℝ)
  (cost_b : ℝ)
  (total_time : ℝ)
  (h_total_length : total_length = 1650)
  (h_team_b_extra : team_b_extra = 30)
  (h_time_ratio : time_ratio = 3/2)
  (h_cost_a : cost_a = 90000)
  (h_cost_b : cost_b = 120000)
  (h_total_time : total_time = 14) :
  ∃ (rate_a rate_b total_cost : ℝ),
    rate_a = 60 ∧
    rate_b = 90 ∧
    total_cost = 2340000 ∧
    rate_b = rate_a + team_b_extra ∧
    (total_length / rate_b) * time_ratio = (total_length / rate_a) ∧
    ∃ (solo_days : ℝ),
      solo_days * rate_a + (total_time - solo_days) * (rate_a + rate_b) = total_length ∧
      total_cost = solo_days * cost_a + total_time * cost_a + (total_time - solo_days) * cost_b :=
by sorry

end irrigation_canal_construction_l1227_122752


namespace largest_n_with_unique_k_l1227_122757

theorem largest_n_with_unique_k : 
  (∀ n : ℕ+, n > 1 → 
    ¬(∃! k : ℤ, (3 : ℚ)/7 < (n : ℚ)/((n : ℚ) + k) ∧ (n : ℚ)/((n : ℚ) + k) < 8/19)) ∧
  (∃! k : ℤ, (3 : ℚ)/7 < (1 : ℚ)/((1 : ℚ) + k) ∧ (1 : ℚ)/((1 : ℚ) + k) < 8/19) :=
by sorry

end largest_n_with_unique_k_l1227_122757


namespace sum_of_integers_from_1_to_3_l1227_122762

theorem sum_of_integers_from_1_to_3 : 
  (Finset.range 3).sum (fun i => i + 1) = 6 := by
  sorry

end sum_of_integers_from_1_to_3_l1227_122762


namespace cos_2alpha_value_l1227_122708

theorem cos_2alpha_value (α : Real) (h : Real.sin (α + 3 * Real.pi / 2) = Real.sqrt 3 / 3) :
  Real.cos (2 * α) = -1 / 3 := by
  sorry

end cos_2alpha_value_l1227_122708


namespace smallest_number_above_threshold_l1227_122786

theorem smallest_number_above_threshold : 
  let numbers : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]
  let threshold : ℚ := 1.1
  let filtered := numbers.filter (λ x => x ≥ threshold)
  filtered.minimum? = some 1.2 := by
sorry

end smallest_number_above_threshold_l1227_122786


namespace tetrahedron_volume_l1227_122703

/-- Given a tetrahedron with volume V, two faces with areas S₁ and S₂, 
    their common edge of length a, and the dihedral angle φ between these faces, 
    prove that V = (2/3) * (S₁ * S₂ * sin(φ)) / a. -/
theorem tetrahedron_volume (V S₁ S₂ a φ : ℝ) 
  (h₁ : V > 0) 
  (h₂ : S₁ > 0) 
  (h₃ : S₂ > 0) 
  (h₄ : a > 0) 
  (h₅ : 0 < φ ∧ φ < π) : 
  V = (2/3) * (S₁ * S₂ * Real.sin φ) / a := by
  sorry

end tetrahedron_volume_l1227_122703


namespace total_germs_count_l1227_122735

/-- The number of petri dishes in the biology lab -/
def num_dishes : ℕ := 10800

/-- The number of germs in a single petri dish -/
def germs_per_dish : ℕ := 500

/-- The total number of germs in the biology lab -/
def total_germs : ℕ := num_dishes * germs_per_dish

/-- Theorem stating that the total number of germs is 5,400,000 -/
theorem total_germs_count : total_germs = 5400000 := by
  sorry

end total_germs_count_l1227_122735


namespace susan_stationery_purchase_l1227_122759

theorem susan_stationery_purchase (pencil_cost : ℚ) (pen_cost : ℚ) (total_spent : ℚ) (pencils_bought : ℕ) :
  pencil_cost = 25 / 100 →
  pen_cost = 80 / 100 →
  total_spent = 20 →
  pencils_bought = 16 →
  ∃ (pens_bought : ℕ),
    (pencils_bought : ℚ) * pencil_cost + (pens_bought : ℚ) * pen_cost = total_spent ∧
    pencils_bought + pens_bought = 36 :=
by sorry

end susan_stationery_purchase_l1227_122759


namespace cost_price_calculation_l1227_122749

theorem cost_price_calculation (cost_price : ℝ) : 
  cost_price * 1.20 * 0.91 = cost_price + 16 → cost_price = 200 := by
  sorry

end cost_price_calculation_l1227_122749


namespace expected_points_100_games_prob_specific_envelope_l1227_122702

/- Define the game parameters -/
def num_envelopes : ℕ := 13
def win_points : ℕ := 6

/- Define the probability of winning a single question -/
def win_prob : ℚ := 1/2

/- Define the expected number of envelopes played in a single game -/
noncomputable def expected_envelopes_per_game : ℝ := 12

/- Theorem for the expected points over 100 games -/
theorem expected_points_100_games :
  ∃ (expected_points : ℕ), expected_points = 465 := by sorry

/- Theorem for the probability of choosing a specific envelope -/
theorem prob_specific_envelope :
  ∃ (prob : ℚ), prob = 12/13 := by sorry

end expected_points_100_games_prob_specific_envelope_l1227_122702


namespace cistern_fill_time_l1227_122732

/-- The time it takes to fill the cistern with both taps open -/
def both_taps_time : ℚ := 28 / 3

/-- The time it takes to empty the cistern with the second tap -/
def empty_time : ℚ := 7

/-- The time it takes to fill the cistern with the first tap -/
def fill_time : ℚ := 4

theorem cistern_fill_time :
  (1 / fill_time - 1 / empty_time) = 1 / both_taps_time :=
by sorry

end cistern_fill_time_l1227_122732


namespace marble_doubling_l1227_122730

theorem marble_doubling (k : ℕ) : (∀ n : ℕ, n < k → 5 * 2^n ≤ 200) ∧ 5 * 2^k > 200 ↔ k = 6 := by
  sorry

end marble_doubling_l1227_122730


namespace cubic_root_reciprocal_sum_l1227_122743

theorem cubic_root_reciprocal_sum (a b c d : ℝ) (p q r : ℝ) : 
  a ≠ 0 → d ≠ 0 →
  (a * p^3 + b * p^2 + c * p + d = 0) →
  (a * q^3 + b * q^2 + c * q + d = 0) →
  (a * r^3 + b * r^2 + c * r + d = 0) →
  (1 / p^2 + 1 / q^2 + 1 / r^2) = (c^2 - 2 * b * d) / d^2 :=
by sorry

end cubic_root_reciprocal_sum_l1227_122743


namespace first_month_sale_l1227_122787

def average_sale : ℕ := 7000
def num_months : ℕ := 6
def sale_month2 : ℕ := 6524
def sale_month3 : ℕ := 5689
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6000
def sale_month6 : ℕ := 12557

theorem first_month_sale (sale_month1 : ℕ) : 
  sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6 = average_sale * num_months →
  sale_month1 = average_sale * num_months - (sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) :=
by
  sorry

#eval average_sale * num_months - (sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6)

end first_month_sale_l1227_122787


namespace perfect_square_trinomial_l1227_122712

theorem perfect_square_trinomial (a b c : ℤ) :
  (∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^2) →
  ∃ d e : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (d * x + e)^2 :=
by sorry

end perfect_square_trinomial_l1227_122712


namespace initial_worksheets_l1227_122789

theorem initial_worksheets (graded : ℕ) (new_worksheets : ℕ) (total : ℕ) :
  graded = 7 → new_worksheets = 36 → total = 63 →
  ∃ initial : ℕ, initial - graded + new_worksheets = total ∧ initial = 34 :=
by sorry

end initial_worksheets_l1227_122789


namespace martha_crayon_count_l1227_122768

def final_crayon_count (initial : ℕ) (first_purchase : ℕ) (contest_win : ℕ) (second_purchase : ℕ) : ℕ :=
  (initial / 2) + first_purchase + contest_win + second_purchase

theorem martha_crayon_count :
  final_crayon_count 18 20 15 25 = 69 := by
  sorry

end martha_crayon_count_l1227_122768


namespace kathryn_gave_skittles_l1227_122776

def cheryl_start : ℕ := 8
def cheryl_end : ℕ := 97

theorem kathryn_gave_skittles : cheryl_end - cheryl_start = 89 := by
  sorry

end kathryn_gave_skittles_l1227_122776


namespace door_pole_equation_l1227_122747

/-- 
Given a rectangular door and a pole:
- The door's diagonal length is x
- The pole's length is x
- When placed horizontally, the pole extends 4 feet beyond the door's width
- When placed vertically, the pole extends 2 feet beyond the door's height

This theorem proves that the equation (x-2)^2 + (x-4)^2 = x^2 holds true for this configuration.
-/
theorem door_pole_equation (x : ℝ) : (x - 2)^2 + (x - 4)^2 = x^2 := by
  sorry

end door_pole_equation_l1227_122747


namespace square_plus_one_ge_two_abs_l1227_122799

theorem square_plus_one_ge_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end square_plus_one_ge_two_abs_l1227_122799


namespace pentadecagon_triangles_l1227_122704

/-- The number of vertices in a pentadecagon -/
def n : ℕ := 15

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The number of sides in a pentadecagon, which is equal to the number of triangles 
    that have a side coinciding with a side of the pentadecagon -/
def excluded_triangles : ℕ := n

theorem pentadecagon_triangles : 
  (Nat.choose n k) - excluded_triangles = 440 :=
sorry

end pentadecagon_triangles_l1227_122704


namespace arithmetic_progression_ratio_l1227_122790

theorem arithmetic_progression_ratio (a d : ℝ) : 
  (15 * a + 105 * d = 4 * (8 * a + 28 * d)) → (a / d = -7 / 17) := by
  sorry

end arithmetic_progression_ratio_l1227_122790


namespace factorization_equality_l1227_122773

theorem factorization_equality (a b : ℝ) : a^3 + 2*a^2*b + a*b^2 = a*(a+b)^2 := by
  sorry

end factorization_equality_l1227_122773


namespace parabola_translation_correct_l1227_122798

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 2x² -/
def original_parabola : Parabola := { a := 2, b := 0, c := 0 }

/-- Translates a parabola horizontally by h units and vertically by k units -/
def translate (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + k }

/-- The resulting parabola after translation -/
def translated_parabola : Parabola :=
  translate (translate original_parabola 3 0) 0 4

theorem parabola_translation_correct :
  translated_parabola.a = 2 ∧
  translated_parabola.b = -12 ∧
  translated_parabola.c = 22 :=
sorry

end parabola_translation_correct_l1227_122798


namespace playground_insects_l1227_122760

/-- Calculates the number of remaining insects in the playground --/
def remaining_insects (initial_bees initial_beetles initial_ants initial_termites
                       initial_praying_mantises initial_ladybugs initial_butterflies
                       initial_dragonflies : ℕ)
                      (bees_left beetles_taken ants_left termites_moved
                       ladybugs_left butterflies_left dragonflies_left : ℕ) : ℕ :=
  (initial_bees - bees_left) +
  (initial_beetles - beetles_taken) +
  (initial_ants - ants_left) +
  (initial_termites - termites_moved) +
  initial_praying_mantises +
  (initial_ladybugs - ladybugs_left) +
  (initial_butterflies - butterflies_left) +
  (initial_dragonflies - dragonflies_left)

/-- Theorem stating that the number of remaining insects is 54 --/
theorem playground_insects :
  remaining_insects 15 7 12 10 2 10 11 8 6 2 4 3 2 3 1 = 54 := by
  sorry

end playground_insects_l1227_122760


namespace different_size_circles_not_one_tangent_l1227_122751

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_positive : radius > 0

-- Define the number of common tangents between two circles
def num_common_tangents (c1 c2 : Circle) : ℕ := sorry

-- Theorem statement
theorem different_size_circles_not_one_tangent (c1 c2 : Circle) :
  c1.radius ≠ c2.radius →
  num_common_tangents c1 c2 ≠ 1 := by
  sorry

end different_size_circles_not_one_tangent_l1227_122751


namespace annual_forest_gathering_handshakes_count_l1227_122785

/-- The number of handshakes at the Annual Forest Gathering -/
def annual_forest_gathering_handshakes (num_goblins num_elves : ℕ) : ℕ :=
  (num_goblins.choose 2) + (num_goblins * num_elves)

/-- Theorem stating the number of handshakes at the Annual Forest Gathering -/
theorem annual_forest_gathering_handshakes_count :
  annual_forest_gathering_handshakes 25 18 = 750 := by
  sorry

end annual_forest_gathering_handshakes_count_l1227_122785


namespace sum_of_max_min_values_l1227_122793

theorem sum_of_max_min_values (f : ℝ → ℝ) (h : f = fun x ↦ 9 * (Real.cos x)^4 + 12 * (Real.sin x)^2 - 4) :
  (⨆ x, f x) + (⨅ x, f x) = 13 := by
  sorry

end sum_of_max_min_values_l1227_122793


namespace unique_three_digit_number_l1227_122745

theorem unique_three_digit_number :
  ∃! n : ℕ,
    100 ≤ n ∧ n < 1000 ∧
    (n % 100 % 10 = 3 * (n / 100)) ∧
    (n % 5 = 4) ∧
    (n % 11 = 3) ∧
    n = 359 := by
  sorry

end unique_three_digit_number_l1227_122745


namespace average_marks_l1227_122728

def english_marks : ℕ := 86
def mathematics_marks : ℕ := 85
def physics_marks : ℕ := 92
def chemistry_marks : ℕ := 87
def biology_marks : ℕ := 95

def total_marks : ℕ := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

theorem average_marks :
  (total_marks : ℚ) / num_subjects = 89 := by sorry

end average_marks_l1227_122728


namespace postcard_collection_average_l1227_122771

/-- 
Given an arithmetic sequence with:
- First term: 10
- Common difference: 12
- Number of terms: 7
Prove that the average of all terms is 46.
-/
theorem postcard_collection_average : 
  let first_term := 10
  let common_diff := 12
  let num_days := 7
  let last_term := first_term + (num_days - 1) * common_diff
  (first_term + last_term) / 2 = 46 := by
sorry

end postcard_collection_average_l1227_122771


namespace trigonometric_equality_l1227_122734

theorem trigonometric_equality (α β : ℝ) :
  (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 2 →
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 2 := by
  sorry

end trigonometric_equality_l1227_122734


namespace coronavirus_radius_scientific_notation_l1227_122710

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem coronavirus_radius_scientific_notation :
  toScientificNotation 0.000000045 =
    ScientificNotation.mk 4.5 (-8) (by norm_num) :=
by sorry

end coronavirus_radius_scientific_notation_l1227_122710


namespace cubic_root_form_l1227_122796

theorem cubic_root_form : ∃ (x : ℝ), 
  8 * x^3 - 3 * x^2 - 3 * x - 1 = 0 ∧ 
  x = (Real.rpow 81 (1/3 : ℝ) + Real.rpow 9 (1/3 : ℝ) + 1) / 8 := by
  sorry

end cubic_root_form_l1227_122796


namespace sheridan_cats_goal_l1227_122724

/-- The number of cats Mrs. Sheridan currently has -/
def current_cats : ℕ := 11

/-- The number of additional cats Mrs. Sheridan needs -/
def additional_cats : ℕ := 32

/-- The total number of cats Mrs. Sheridan wants to have -/
def total_cats : ℕ := current_cats + additional_cats

theorem sheridan_cats_goal : total_cats = 43 := by
  sorry

end sheridan_cats_goal_l1227_122724


namespace hundredth_ring_squares_nth_ring_squares_l1227_122719

/-- The number of unit squares in the nth ring around a center square -/
def ring_squares (n : ℕ) : ℕ := 8 * n

/-- Theorem: The 100th ring contains 800 unit squares -/
theorem hundredth_ring_squares : ring_squares 100 = 800 := by
  sorry

/-- Theorem: For any positive integer n, the number of unit squares in the nth ring is 8n -/
theorem nth_ring_squares (n : ℕ) : ring_squares n = 8 * n := by
  sorry

end hundredth_ring_squares_nth_ring_squares_l1227_122719


namespace square_difference_of_sum_and_diff_l1227_122744

theorem square_difference_of_sum_and_diff (a b : ℕ+) 
  (h_sum : a + b = 60) 
  (h_diff : a - b = 14) : 
  a^2 - b^2 = 840 := by
sorry

end square_difference_of_sum_and_diff_l1227_122744


namespace smallest_number_satisfying_conditions_l1227_122794

theorem smallest_number_satisfying_conditions : 
  ∃ N : ℕ, 
    N > 0 ∧ 
    N % 4 = 0 ∧ 
    (N + 9) % 2 = 1 ∧ 
    (∀ M : ℕ, M > 0 → M % 4 = 0 → (M + 9) % 2 = 1 → M ≥ N) ∧
    N = 4 := by
  sorry

end smallest_number_satisfying_conditions_l1227_122794


namespace function_properties_l1227_122725

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem function_properties (a b : ℝ) 
  (h : (a - 1)^2 - 4*b < 0) : 
  (∀ x, f a b x > x) ∧ 
  (∀ x, f a b (f a b x) > x) ∧ 
  (a + b > 0) := by
  sorry

end function_properties_l1227_122725


namespace expression_equality_l1227_122739

theorem expression_equality (a b c : ℝ) (h : a^2 + b = b^2 + c ∧ b^2 + c = c^2 + a) :
  a*(a^2 - b^2) + b*(b^2 - c^2) + c*(c^2 - a^2) = 0 := by sorry

end expression_equality_l1227_122739


namespace triangle_properties_l1227_122772

theorem triangle_properties (a b c A B C : Real) :
  -- Triangle conditions
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  -- Given conditions
  π/2 < A ∧ -- A is obtuse
  a * Real.sin B = b * Real.cos B ∧
  C = π/6 →
  -- Conclusions
  A = 2*π/3 ∧
  1 < Real.cos A + Real.cos B + Real.cos C ∧
  Real.cos A + Real.cos B + Real.cos C ≤ 5/4 := by
sorry


end triangle_properties_l1227_122772


namespace age_difference_l1227_122783

theorem age_difference (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : 3 * b = 2 * a) (h4 : a + b = 60) : a - b = 12 := by
  sorry

end age_difference_l1227_122783


namespace annie_laps_bonnie_l1227_122765

/-- The length of the circular track in meters -/
def track_length : ℝ := 500

/-- Annie's speed relative to Bonnie's -/
def annie_speed_ratio : ℝ := 1.5

/-- The number of laps Annie has run when she first laps Bonnie -/
def annie_laps : ℝ := 3

theorem annie_laps_bonnie :
  track_length > 0 →
  annie_speed_ratio = 1.5 →
  (annie_laps * track_length) / annie_speed_ratio = (annie_laps - 1) * track_length :=
by sorry

end annie_laps_bonnie_l1227_122765


namespace has_unique_prime_divisor_l1227_122792

theorem has_unique_prime_divisor (n m : ℕ) (h1 : n > m) (h2 : m > 0) :
  ∃ p : ℕ, Prime p ∧ (p ∣ (2^n - 1)) ∧ ¬(p ∣ (2^m - 1)) := by
  sorry

end has_unique_prime_divisor_l1227_122792


namespace portrait_price_ratio_l1227_122761

def price_8inch : ℝ := 5
def daily_8inch_sales : ℕ := 3
def daily_16inch_sales : ℕ := 5
def earnings_3days : ℝ := 195

def price_ratio : ℝ := 2

theorem portrait_price_ratio :
  let daily_earnings := daily_8inch_sales * price_8inch + daily_16inch_sales * (price_ratio * price_8inch)
  earnings_3days = 3 * daily_earnings :=
by sorry

end portrait_price_ratio_l1227_122761


namespace inscribable_polygons_l1227_122737

/-- The number of evenly spaced holes on the circumference of the circle -/
def num_holes : ℕ := 24

/-- A function that determines if a regular polygon with 'n' sides can be inscribed in the circle -/
def can_inscribe (n : ℕ) : Prop :=
  n ≥ 3 ∧ num_holes % n = 0

/-- The set of numbers of sides for regular polygons that can be inscribed in the circle -/
def valid_polygons : Set ℕ := {n | can_inscribe n}

/-- Theorem stating that the only valid numbers of sides for inscribable regular polygons are 3, 4, 6, 8, 12, and 24 -/
theorem inscribable_polygons :
  valid_polygons = {3, 4, 6, 8, 12, 24} :=
sorry

end inscribable_polygons_l1227_122737


namespace integral_f_minus_x_equals_five_sixths_l1227_122738

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x

-- State the theorem
theorem integral_f_minus_x_equals_five_sixths :
  (∀ x, deriv f x = 2 * x + 1) →
  ∫ x in (1)..(2), f (-x) = 5/6 := by sorry

end integral_f_minus_x_equals_five_sixths_l1227_122738


namespace perpendicular_vectors_m_value_l1227_122756

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a = (1, 2) and b = (-1, m), if they are perpendicular, then m = 1/2 -/
theorem perpendicular_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-1, m)
  perpendicular a b → m = 1/2 := by
sorry

end perpendicular_vectors_m_value_l1227_122756


namespace finite_game_has_winning_strategy_l1227_122777

/-- Represents a two-player game with finite choices and finite length -/
structure FiniteGame where
  /-- The maximum number of moves before the game ends -/
  max_moves : ℕ
  /-- The number of possible choices for each move -/
  num_choices : ℕ
  /-- Predicate to check if the game has ended -/
  is_game_over : (List ℕ) → Bool
  /-- Predicate to determine the winner (true for player A, false for player B) -/
  winner : (List ℕ) → Bool

/-- Definition of a winning strategy for a player -/
def has_winning_strategy (game : FiniteGame) (player : Bool) : Prop :=
  ∃ (strategy : List ℕ → ℕ),
    ∀ (game_state : List ℕ),
      (game_state.length < game.max_moves) →
      (game.is_game_over game_state = false) →
      (game_state.length % 2 = if player then 0 else 1) →
      (strategy game_state ≤ game.num_choices) ∧
      (∃ (final_state : List ℕ),
        final_state.length ≤ game.max_moves ∧
        game.is_game_over final_state = true ∧
        game.winner final_state = player)

/-- Theorem: In a finite two-player game, one player must have a winning strategy -/
theorem finite_game_has_winning_strategy (game : FiniteGame) :
  has_winning_strategy game true ∨ has_winning_strategy game false :=
sorry

end finite_game_has_winning_strategy_l1227_122777


namespace fraction_subtraction_equality_l1227_122791

theorem fraction_subtraction_equality : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end fraction_subtraction_equality_l1227_122791


namespace basketball_score_ratio_l1227_122779

/-- Given the scores of three basketball players, prove that the ratio of Tim's points to Ken's points is 1:2 -/
theorem basketball_score_ratio 
  (joe tim ken : ℕ)  -- Scores of Joe, Tim, and Ken
  (h1 : tim = joe + 20)  -- Tim scored 20 points more than Joe
  (h2 : joe + tim + ken = 100)  -- Total points scored is 100
  (h3 : tim = 30)  -- Tim scored 30 points
  : tim * 2 = ken :=
by sorry

end basketball_score_ratio_l1227_122779


namespace temperature_difference_l1227_122774

/-- Given the highest and lowest temperatures in a city on a certain day, 
    calculate the temperature difference. -/
theorem temperature_difference 
  (highest_temp lowest_temp : ℤ) 
  (h_highest : highest_temp = 11)
  (h_lowest : lowest_temp = -1) :
  highest_temp - lowest_temp = 12 := by
  sorry

end temperature_difference_l1227_122774


namespace exactly_two_talents_l1227_122727

theorem exactly_two_talents (total_students : ℕ) 
  (cannot_sing cannot_dance cannot_act no_talent : ℕ) :
  total_students = 120 →
  cannot_sing = 50 →
  cannot_dance = 75 →
  cannot_act = 45 →
  no_talent = 15 →
  (∃ (two_talents : ℕ), two_talents = 70 ∧ 
    two_talents = total_students - 
      (cannot_sing + cannot_dance + cannot_act - 2 * no_talent)) :=
by sorry

end exactly_two_talents_l1227_122727


namespace machine_precision_test_l1227_122755

-- Define the sample data
def sample_data : List (Float × Nat) := [(3.0, 2), (3.5, 6), (3.8, 9), (4.4, 7), (4.5, 1)]

-- Define the hypothesized variance
def sigma_0_squared : Float := 0.1

-- Define the significance level
def alpha : Float := 0.05

-- Define the degrees of freedom
def df : Nat := 24

-- Function to calculate sample variance
def calculate_sample_variance (data : List (Float × Nat)) : Float :=
  sorry

-- Function to calculate chi-square test statistic
def calculate_chi_square (sample_variance : Float) (n : Nat) (sigma_0_squared : Float) : Float :=
  sorry

-- Function to get critical value from chi-square distribution
def get_chi_square_critical (alpha : Float) (df : Nat) : Float :=
  sorry

theorem machine_precision_test (data : List (Float × Nat)) (alpha : Float) (df : Nat) (sigma_0_squared : Float) :
  let sample_variance := calculate_sample_variance data
  let chi_square_obs := calculate_chi_square sample_variance data.length sigma_0_squared
  let chi_square_crit := get_chi_square_critical alpha df
  chi_square_obs > chi_square_crit :=
by
  sorry

#check machine_precision_test sample_data alpha df sigma_0_squared

end machine_precision_test_l1227_122755


namespace derivative_at_zero_l1227_122748

theorem derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x*(deriv f (-1))) :
  deriv f 0 = 4 := by
sorry

end derivative_at_zero_l1227_122748


namespace sheila_work_hours_l1227_122767

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  monday_wednesday_friday_hours : ℕ
  tuesday_thursday_hours : ℕ
  weekly_earnings : ℕ
  hourly_rate : ℕ

/-- Theorem stating the number of hours Sheila works on Monday, Wednesday, and Friday --/
theorem sheila_work_hours (schedule : WorkSchedule) : 
  schedule.monday_wednesday_friday_hours = 24 :=
by
  have h1 : schedule.tuesday_thursday_hours = 6 * 2 := by sorry
  have h2 : schedule.weekly_earnings = 468 := by sorry
  have h3 : schedule.hourly_rate = 13 := by sorry
  sorry

end sheila_work_hours_l1227_122767


namespace length_breadth_difference_l1227_122709

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area_is_23_times_breadth : area = 23 * breadth
  breadth_is_13 : breadth = 13

/-- The area of a rectangle -/
def area (r : RectangularPlot) : ℝ := r.length * r.breadth

/-- Theorem: The difference between length and breadth is 10 meters -/
theorem length_breadth_difference (r : RectangularPlot) :
  r.length - r.breadth = 10 := by
  sorry

#check length_breadth_difference

end length_breadth_difference_l1227_122709


namespace triangle_area_13_14_15_l1227_122754

/-- The area of a triangle with sides 13, 14, and 15 is 84 -/
theorem triangle_area_13_14_15 : ∃ (area : ℝ), area = 84 ∧ 
  (∀ (s : ℝ), s = (13 + 14 + 15) / 2 → 
    area = Real.sqrt (s * (s - 13) * (s - 14) * (s - 15))) := by
  sorry

end triangle_area_13_14_15_l1227_122754


namespace book_series_first_year_l1227_122788

/-- Represents the publication years of a book series -/
def BookSeries (a : ℕ) : List ℕ :=
  List.range 7 |>.map (fun i => a + 7 * i)

/-- The theorem stating the properties of the book series -/
theorem book_series_first_year :
  ∀ a : ℕ,
  (BookSeries a).length = 7 ∧
  (∀ i j, i < j → (BookSeries a).get i < (BookSeries a).get j) ∧
  (BookSeries a).sum = 13524 →
  a = 1911 := by
sorry


end book_series_first_year_l1227_122788


namespace tetrahedron_volume_is_sqrt3_over_3_l1227_122714

-- Define the square ABCD
def square_side_length : ℝ := 2

-- Define point E as the midpoint of AB
def E_is_midpoint (A B E : ℝ × ℝ) : Prop :=
  E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the folding along EC and ED
def folded_square (A B C D E : ℝ × ℝ) : Prop :=
  E_is_midpoint A B E ∧
  (A.1 - E.1)^2 + (A.2 - E.2)^2 = (B.1 - E.1)^2 + (B.2 - E.2)^2

-- Define the tetrahedron CDEA
structure Tetrahedron :=
  (C D E A : ℝ × ℝ)

-- Define the volume of a tetrahedron
def tetrahedron_volume (t : Tetrahedron) : ℝ := sorry

-- Theorem statement
theorem tetrahedron_volume_is_sqrt3_over_3 
  (A B C D E : ℝ × ℝ) 
  (h1 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = square_side_length^2)
  (h2 : (D.1 - B.1)^2 + (D.2 - B.2)^2 = square_side_length^2)
  (h3 : folded_square A B C D E) :
  tetrahedron_volume {C := C, D := D, E := E, A := A} = Real.sqrt 3 / 3 :=
sorry

end tetrahedron_volume_is_sqrt3_over_3_l1227_122714


namespace sum_property_unique_l1227_122775

/-- The property that the sum of the first n natural numbers can be written as n followed by three digits in base 10 -/
def sum_property (n : ℕ) : Prop :=
  ∃ k : ℕ, k < 1000 ∧ (n * (n + 1)) / 2 = 1000 * n + k

/-- Theorem stating that 1999 is the only natural number satisfying the sum property -/
theorem sum_property_unique : ∀ n : ℕ, sum_property n ↔ n = 1999 :=
sorry

end sum_property_unique_l1227_122775


namespace frustum_max_volume_l1227_122722

/-- The maximum volume of a frustum within a sphere -/
theorem frustum_max_volume (r : ℝ) (r_top : ℝ) (r_bottom : ℝ) (h_r : r = 5) (h_top : r_top = 3) (h_bottom : r_bottom = 4) :
  ∃ v : ℝ, v = (259 / 3) * Real.pi ∧ 
  (∀ v' : ℝ, v' ≤ v ∧ 
    (∃ h : ℝ, v' = (1 / 3) * h * (r_top^2 * Real.pi + r_top * r_bottom * Real.pi + r_bottom^2 * Real.pi) ∧
              0 < h ∧ h ≤ 2 * (r^2 - r_top^2).sqrt)) :=
sorry

end frustum_max_volume_l1227_122722


namespace minimum_value_theorem_l1227_122782

-- Define the variables x, y, and z as positive real numbers
variable (x y z : ℝ)

-- Define the conditions
def positive_conditions : Prop := x > 0 ∧ y > 0 ∧ z > 0
def equation_condition : Prop := x - 2*y + 3*z = 0

-- State the theorem
theorem minimum_value_theorem 
  (h1 : positive_conditions x y z) 
  (h2 : equation_condition x y z) :
  ∃ (f : ℝ → ℝ → ℝ → ℝ), ∀ (a b c : ℝ), 
    positive_conditions a b c → 
    equation_condition a b c → 
    f x y z ≤ f a b c :=
sorry

end minimum_value_theorem_l1227_122782


namespace product_of_decimals_l1227_122740

theorem product_of_decimals : (0.5 : ℝ) * 0.8 = 0.40 := by sorry

end product_of_decimals_l1227_122740


namespace evaluate_expression_l1227_122723

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y - 2 * y^x = 277 := by
  sorry

end evaluate_expression_l1227_122723


namespace james_cattle_problem_l1227_122715

/-- Represents the problem of determining the number of cattle James bought --/
theorem james_cattle_problem (purchase_price feeding_cost_percentage cattle_weight selling_price_per_pound profit : ℝ) 
  (h1 : purchase_price = 40000)
  (h2 : feeding_cost_percentage = 0.2)
  (h3 : cattle_weight = 1000)
  (h4 : selling_price_per_pound = 2)
  (h5 : profit = 112000) :
  (purchase_price + purchase_price * feeding_cost_percentage) / 
  (cattle_weight * selling_price_per_pound) + 
  profit / (cattle_weight * selling_price_per_pound) = 100 := by
  sorry

end james_cattle_problem_l1227_122715


namespace sum_difference_equals_3146_main_theorem_l1227_122721

theorem sum_difference_equals_3146 : ℕ → Prop :=
  fun n =>
    let even_sum := n * (n + 1)
    let multiples_of_3_sum := (n / 3) * ((n / 3) + 1) * 3 / 2
    let odd_sum := ((n - 1) / 2 + 1) ^ 2
    (even_sum - multiples_of_3_sum - odd_sum = 3146) ∧ (2 * n = 400)

theorem main_theorem : ∃ n : ℕ, sum_difference_equals_3146 n := by
  sorry

end sum_difference_equals_3146_main_theorem_l1227_122721


namespace curve_C_properties_l1227_122742

-- Define the curve C
def C (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (4 - t) + p.2^2 / (t - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (t : ℝ) : Prop :=
  t < 1 ∨ t > 4

-- Define what it means for C to be an ellipse with foci on the X-axis
def is_ellipse_x_axis (t : ℝ) : Prop :=
  1 < t ∧ t < 5/2

theorem curve_C_properties (t : ℝ) :
  (is_hyperbola t ↔ ∃ (a b : ℝ), C t = {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1}) ∧
  (is_ellipse_x_axis t ↔ ∃ (a b : ℝ), a > b ∧ C t = {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1}) :=
by sorry

end curve_C_properties_l1227_122742


namespace total_pies_baked_l1227_122731

/-- The number of pies Eddie can bake in a day -/
def eddie_pies_per_day : ℕ := 3

/-- The number of pies Eddie's sister can bake in a day -/
def sister_pies_per_day : ℕ := 6

/-- The number of pies Eddie's mother can bake in a day -/
def mother_pies_per_day : ℕ := 8

/-- The number of days they will bake pies -/
def days_baking : ℕ := 7

/-- Theorem stating the total number of pies baked in 7 days -/
theorem total_pies_baked : 
  (eddie_pies_per_day * days_baking) + 
  (sister_pies_per_day * days_baking) + 
  (mother_pies_per_day * days_baking) = 119 := by
sorry

end total_pies_baked_l1227_122731


namespace output_for_twelve_l1227_122753

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 20 then step1 - 2 else step1 / 2

theorem output_for_twelve : function_machine 12 = 34 := by sorry

end output_for_twelve_l1227_122753


namespace small_cuboid_length_l1227_122780

/-- Represents the dimensions of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

/-- Theorem: Given a large cuboid of 16m x 10m x 12m and small cuboids of Lm x 4m x 3m,
    if 32 small cuboids can be formed from the large cuboid, then L = 5m -/
theorem small_cuboid_length
  (large : Cuboid)
  (small : Cuboid)
  (h1 : large.length = 16)
  (h2 : large.width = 10)
  (h3 : large.height = 12)
  (h4 : small.width = 4)
  (h5 : small.height = 3)
  (h6 : volume large = 32 * volume small) :
  small.length = 5 := by
  sorry

end small_cuboid_length_l1227_122780


namespace power_sum_problem_l1227_122716

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 26)
  (h4 : a * x^4 + b * y^4 = 58) :
  a * x^6 + b * y^6 = -220 := by
  sorry

end power_sum_problem_l1227_122716


namespace min_largest_median_l1227_122736

/-- Represents a 5 × 18 rectangle filled with numbers from 1 to 90 -/
def Rectangle := Fin 5 → Fin 18 → Fin 90

/-- The median of a column in the rectangle -/
def columnMedian (rect : Rectangle) (col : Fin 18) : Fin 90 :=
  sorry

/-- The largest median among all columns -/
def largestMedian (rect : Rectangle) : Fin 90 :=
  sorry

/-- Theorem stating the minimum possible value for the largest median -/
theorem min_largest_median :
  ∃ (rect : Rectangle), largestMedian rect = 54 ∧
  ∀ (rect' : Rectangle), largestMedian rect' ≥ 54 :=
sorry

end min_largest_median_l1227_122736


namespace min_disks_needed_l1227_122733

/-- Represents the capacity of a disk in MB -/
def diskCapacity : ℚ := 2.88

/-- Represents the sizes of files in MB -/
def fileSizes : List ℚ := [1.2, 0.9, 0.6, 0.3]

/-- Represents the quantities of files for each size -/
def fileQuantities : List ℕ := [5, 10, 8, 7]

/-- Calculates the total size of all files -/
def totalFileSize : ℚ := (List.zip fileSizes fileQuantities).foldl (λ acc (size, quantity) => acc + size * quantity) 0

/-- Theorem stating the minimum number of disks needed -/
theorem min_disks_needed : 
  ∃ (arrangement : List (List ℚ)), 
    (∀ disk ∈ arrangement, disk.sum ≤ diskCapacity) ∧ 
    (arrangement.map (List.length)).sum = (fileQuantities.sum) ∧
    arrangement.length = 14 :=
sorry

end min_disks_needed_l1227_122733


namespace local_max_range_l1227_122764

def f' (x a : ℝ) : ℝ := a * (x + 1) * (x - a)

theorem local_max_range (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, (deriv f) x = f' x a)
  (h2 : IsLocalMax f a) :
  -1 < a ∧ a < 0 := by
  sorry

end local_max_range_l1227_122764


namespace stamp_collection_value_l1227_122707

/-- Given a collection of stamps with equal individual value, 
    calculate the total value of the collection. -/
theorem stamp_collection_value 
  (total_stamps : ℕ) 
  (sample_stamps : ℕ) 
  (sample_value : ℝ) 
  (h1 : total_stamps = 30)
  (h2 : sample_stamps = 10)
  (h3 : sample_value = 45) :
  (total_stamps : ℝ) * (sample_value / sample_stamps) = 135 :=
by sorry

end stamp_collection_value_l1227_122707


namespace subtraction_difference_l1227_122700

theorem subtraction_difference (original : ℝ) (percentage : ℝ) (flat_amount : ℝ) : 
  original = 200 → percentage = 25 → flat_amount = 25 →
  (original - flat_amount) - (original - percentage / 100 * original) = 25 := by
  sorry

end subtraction_difference_l1227_122700


namespace inequality_and_equality_condition_l1227_122795

theorem inequality_and_equality_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 ≥ 6 * Real.sqrt 3 ∧ 
  (a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 = 6 * Real.sqrt 3 ↔ 
   a = b ∧ b = c ∧ c = Real.rpow 3 (1/4)) :=
sorry

end inequality_and_equality_condition_l1227_122795


namespace f_difference_l1227_122781

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f(n) defined as sigma(n) / n -/
def f (n : ℕ+) : ℚ := (sigma n : ℚ) / n

/-- Theorem stating that f(540) - f(180) = 7/90 -/
theorem f_difference : f 540 - f 180 = 7 / 90 := by sorry

end f_difference_l1227_122781


namespace motorboat_speed_adjustment_l1227_122746

/-- 
Given two motorboats with the same initial speed traveling in opposite directions
relative to a river current, prove that if one boat increases its speed by x and
the other decreases by x, resulting in equal time changes, then x equals twice
the current speed.
-/
theorem motorboat_speed_adjustment (v a x : ℝ) (h1 : v > a) (h2 : v > 0) (h3 : a > 0) :
  (1 / (v - a) - 1 / (v + x - a) = 1 / (v + a - x) - 1 / (v + a)) →
  x = 2 * a := by
sorry

end motorboat_speed_adjustment_l1227_122746


namespace ladder_angle_approx_l1227_122701

/-- Given a right triangle with hypotenuse 19 meters and adjacent side 9.493063650744542 meters,
    the angle between the hypotenuse and the adjacent side is approximately 60 degrees. -/
theorem ladder_angle_approx (hypotenuse : ℝ) (adjacent : ℝ) (angle : ℝ) 
    (h1 : hypotenuse = 19)
    (h2 : adjacent = 9.493063650744542)
    (h3 : angle = Real.arccos (adjacent / hypotenuse)) :
    ∃ ε > 0, |angle - 60 * π / 180| < ε :=
  sorry

end ladder_angle_approx_l1227_122701


namespace lending_interest_rate_lending_rate_is_six_percent_l1227_122784

/-- Calculates the lending interest rate given the borrowing details and yearly gain -/
theorem lending_interest_rate 
  (borrowed_amount : ℝ) 
  (borrowing_rate : ℝ) 
  (duration : ℝ) 
  (yearly_gain : ℝ) : ℝ :=
let borrowed_interest := borrowed_amount * borrowing_rate * duration / 100
let total_gain := yearly_gain * duration
let lending_rate := (total_gain + borrowed_interest) * 100 / (borrowed_amount * duration)
lending_rate

/-- The lending interest rate is 6% given the specified conditions -/
theorem lending_rate_is_six_percent : 
  lending_interest_rate 5000 4 2 100 = 6 := by
  sorry

end lending_interest_rate_lending_rate_is_six_percent_l1227_122784

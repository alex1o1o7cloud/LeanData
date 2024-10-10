import Mathlib

namespace sufficient_not_necessary_l1596_159685

theorem sufficient_not_necessary :
  (∀ x : ℝ, x - 1 > 0 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ x - 1 ≤ 0) := by
  sorry

end sufficient_not_necessary_l1596_159685


namespace worker_payment_l1596_159609

/-- Given a sum of money that can pay worker A for 18 days, worker B for 12 days, 
    and worker C for 24 days, prove that it can pay all three workers together for 5 days. -/
theorem worker_payment (S : ℚ) (A B C : ℚ) (hA : S = 18 * A) (hB : S = 12 * B) (hC : S = 24 * C) :
  ∃ D : ℕ, D = 5 ∧ S = D * (A + B + C) :=
sorry

end worker_payment_l1596_159609


namespace count_pairs_eq_27_l1596_159694

open Set

def S : Finset Char := {'a', 'b', 'c'}

/-- The number of ordered pairs (A, B) of subsets of S such that A ∪ B = S and A ≠ B -/
def count_pairs : ℕ :=
  (Finset.powerset S).card * (Finset.powerset S).card -
  (Finset.powerset S).card

theorem count_pairs_eq_27 : count_pairs = 27 := by sorry

end count_pairs_eq_27_l1596_159694


namespace arithmetic_sequence_common_difference_l1596_159653

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) -- a is the arithmetic sequence
  (h1 : a 5 = 3) -- given condition: a_5 = 3
  (h2 : a 6 = -2) -- given condition: a_6 = -2
  : ∃ d : ℤ, d = -5 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry

end arithmetic_sequence_common_difference_l1596_159653


namespace partial_fraction_decomposition_l1596_159655

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  5 * x^2 + 7 * x = 3.5 * (x - 4)^2 + 1.5 * (x - 2) * (x - 4) + 18 * (x - 2) := by
  sorry

end partial_fraction_decomposition_l1596_159655


namespace sum_of_squares_of_roots_l1596_159617

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (10 * x₁^2 + 15 * x₁ - 20 = 0) →
  (10 * x₂^2 + 15 * x₂ - 20 = 0) →
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 25/4 := by
sorry

end sum_of_squares_of_roots_l1596_159617


namespace three_competition_participation_l1596_159624

theorem three_competition_participation 
  (total : ℕ) 
  (chinese : ℕ) 
  (math : ℕ) 
  (english : ℕ) 
  (chinese_math : ℕ) 
  (math_english : ℕ) 
  (chinese_english : ℕ) 
  (none : ℕ) 
  (h1 : total = 100)
  (h2 : chinese = 39)
  (h3 : math = 49)
  (h4 : english = 41)
  (h5 : chinese_math = 14)
  (h6 : math_english = 13)
  (h7 : chinese_english = 9)
  (h8 : none = 1) :
  ∃ (all_three : ℕ), 
    all_three = 6 ∧ 
    total = chinese + math + english - chinese_math - math_english - chinese_english + all_three + none :=
by sorry

end three_competition_participation_l1596_159624


namespace average_seashells_per_person_l1596_159633

/-- The number of seashells found by Sally -/
def sally_shells : ℕ := 9

/-- The number of seashells found by Tom -/
def tom_shells : ℕ := 7

/-- The number of seashells found by Jessica -/
def jessica_shells : ℕ := 5

/-- The number of seashells found by Alex -/
def alex_shells : ℕ := 12

/-- The total number of people who found seashells -/
def total_people : ℕ := 4

/-- The average number of seashells found per person -/
def average_shells : ℚ := (sally_shells + tom_shells + jessica_shells + alex_shells : ℚ) / total_people

theorem average_seashells_per_person :
  average_shells = 33 / 4 :=
by sorry

end average_seashells_per_person_l1596_159633


namespace geometric_series_sum_l1596_159612

/-- Sum of a geometric series with n terms -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum :
  let a : ℚ := 1/3
  let r : ℚ := -1/2
  let n : ℕ := 6
  geometric_sum a r n = 7/32 := by
sorry

end geometric_series_sum_l1596_159612


namespace min_value_reciprocal_sum_min_value_reciprocal_sum_equality_l1596_159634

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

theorem min_value_reciprocal_sum_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b = 3 + 2 * Real.sqrt 2) ↔ (b / a = 2 * a / b) :=
by sorry

end min_value_reciprocal_sum_min_value_reciprocal_sum_equality_l1596_159634


namespace is_rectangle_l1596_159684

/-- Given points A, B, C, and D in a 2D plane, prove that ABCD is a rectangle -/
theorem is_rectangle (A B C D : ℝ × ℝ) : 
  A = (-2, 0) → B = (1, 6) → C = (5, 4) → D = (2, -2) →
  (B.1 - A.1, B.2 - A.2) = (C.1 - D.1, C.2 - D.2) ∧
  (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0 := by
  sorry

#check is_rectangle

end is_rectangle_l1596_159684


namespace parabola_equation_l1596_159611

/-- A parabola with vertex at the origin, opening upward -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ
  pointOnParabola : ℝ × ℝ

/-- The parabola satisfies the given conditions -/
def satisfiesConditions (p : Parabola) : Prop :=
  let (xF, yF) := p.focus
  let (xA, yA) := p.pointOnParabola
  let yM := p.directrix 0
  yF > 0 ∧ 
  Real.sqrt ((xA - 0)^2 + (yA - yM)^2) = Real.sqrt 17 ∧
  Real.sqrt ((xA - xF)^2 + (yA - yF)^2) = 3

/-- The equation of the parabola is x² = 12y -/
def hasEquation (p : Parabola) : Prop :=
  let (x, y) := p.pointOnParabola
  x^2 = 12 * y

theorem parabola_equation (p : Parabola) 
  (h : satisfiesConditions p) : hasEquation p := by
  sorry

end parabola_equation_l1596_159611


namespace recurrence_sequence_a9_l1596_159686

/-- An increasing sequence of positive integers satisfying the given recurrence relation. -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ n, 1 ≤ n → a (n + 2) = a (n + 1) + a n)

theorem recurrence_sequence_a9 (a : ℕ → ℕ) (h : RecurrenceSequence a) (h6 : a 6 = 56) :
  a 9 = 270 := by
  sorry

#check recurrence_sequence_a9

end recurrence_sequence_a9_l1596_159686


namespace paper_products_distribution_l1596_159627

theorem paper_products_distribution (total : ℕ) 
  (h1 : total = 20)
  (h2 : total / 2 + total / 4 + total / 5 < total) :
  total - (total / 2 + total / 4 + total / 5) = 1 := by
  sorry

end paper_products_distribution_l1596_159627


namespace parabola_directrix_equation_l1596_159646

/-- A parabola is defined by its equation in the form y² = -4px, where p is the focal length. -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = -4 * p * x

/-- The directrix of a parabola is a vertical line with equation x = p. -/
def directrix (parabola : Parabola) : ℝ → Prop :=
  fun x => x = parabola.p

theorem parabola_directrix_equation :
  ∀ (y : ℝ), ∃ (parabola : Parabola),
    (∀ (x : ℝ), parabola.equation x y ↔ y^2 = -4*x) →
    (∀ (x : ℝ), directrix parabola x ↔ x = 1) := by
  sorry

end parabola_directrix_equation_l1596_159646


namespace snooker_tournament_ticket_sales_l1596_159629

/-- Calculates the total cost of tickets sold at a snooker tournament --/
theorem snooker_tournament_ticket_sales 
  (total_tickets : ℕ) 
  (vip_price general_price : ℚ) 
  (ticket_difference : ℕ) 
  (h1 : total_tickets = 320)
  (h2 : vip_price = 45)
  (h3 : general_price = 20)
  (h4 : ticket_difference = 276) :
  let general_tickets := (total_tickets + ticket_difference) / 2
  let vip_tickets := total_tickets - general_tickets
  vip_price * vip_tickets + general_price * general_tickets = 6950 :=
by sorry

end snooker_tournament_ticket_sales_l1596_159629


namespace binomial_sum_even_n_l1596_159672

theorem binomial_sum_even_n (n : ℕ) (h : Even n) :
  (Finset.sum (Finset.range (n + 1)) (fun k =>
    if k % 2 = 0 then (1 : ℕ) * Nat.choose n k
    else 2 * Nat.choose n k)) = 3 * 2^(n - 1) :=
by sorry

end binomial_sum_even_n_l1596_159672


namespace inequality_proof_l1596_159681

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  (abs a > abs b) ∧ (b / a < a / b) := by
  sorry

end inequality_proof_l1596_159681


namespace hilt_fountain_trips_l1596_159615

/-- The number of trips to the water fountain -/
def number_of_trips (distance_to_fountain : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance / distance_to_fountain

/-- Theorem: Mrs. Hilt will go to the water fountain 4 times -/
theorem hilt_fountain_trips :
  let distance_to_fountain : ℕ := 30
  let total_distance : ℕ := 120
  number_of_trips distance_to_fountain total_distance = 4 := by
  sorry

end hilt_fountain_trips_l1596_159615


namespace min_value_theorem_equality_condition_l1596_159657

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  6 * x + 1 / (x^2) ≥ 7 * (6^(1/3)) := by
  sorry

theorem equality_condition : 
  6 * ((1/6)^(1/3)) + 1 / (((1/6)^(1/3))^2) = 7 * (6^(1/3)) := by
  sorry

end min_value_theorem_equality_condition_l1596_159657


namespace problem_solution_l1596_159628

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = 0)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 1) :
  b / (a + b) + c / (b + c) + a / (c + a) = 5/2 := by
sorry

end problem_solution_l1596_159628


namespace jungkook_has_bigger_number_l1596_159668

theorem jungkook_has_bigger_number : 
  let jungkook_number := 3 + 6
  let yoongi_number := 4
  jungkook_number > yoongi_number := by
sorry

end jungkook_has_bigger_number_l1596_159668


namespace mod_power_seventeen_seven_l1596_159698

theorem mod_power_seventeen_seven (m : ℕ) : 
  17^7 % 11 = m ∧ 0 ≤ m ∧ m < 11 → m = 8 := by
  sorry

end mod_power_seventeen_seven_l1596_159698


namespace perpendicular_vectors_l1596_159695

def vector_a (t : ℝ) : Fin 2 → ℝ := ![t, 1]
def vector_b : Fin 2 → ℝ := ![2, 4]

def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem perpendicular_vectors (t : ℝ) :
  perpendicular (vector_a t) vector_b → t = -2 := by
  sorry

end perpendicular_vectors_l1596_159695


namespace central_region_area_l1596_159690

/-- The area of the central region in a square with intersecting lines --/
theorem central_region_area (s : ℝ) (h : s = 10) : 
  let a := s / 3
  let b := 2 * s / 3
  let central_side := (s - (a + b)) / 2
  central_side ^ 2 = (s / 6) ^ 2 := by
  sorry

end central_region_area_l1596_159690


namespace line_transformation_l1596_159660

/-- Given a line l: ax + y - 7 = 0 transformed by matrix A to line l': 9x + y - 91 = 0,
    prove that a = 2 and b = 13 -/
theorem line_transformation (a b : ℝ) : 
  (∀ x y : ℝ, a * x + y - 7 = 0 → 
    9 * (3 * x) + (-x + b * y) - 91 = 0) → 
  a = 2 ∧ b = 13 := by
sorry

end line_transformation_l1596_159660


namespace ln_1_1_approx_fourth_root_17_approx_l1596_159679

-- Define the required accuracy
def accuracy : ℝ := 0.0001

-- Theorem for ln(1.1)
theorem ln_1_1_approx : |Real.log 1.1 - 0.0953| < accuracy := by sorry

-- Theorem for ⁴√17
theorem fourth_root_17_approx : |((17 : ℝ) ^ (1/4)) - 2.0305| < accuracy := by sorry

end ln_1_1_approx_fourth_root_17_approx_l1596_159679


namespace room_width_proof_l1596_159637

theorem room_width_proof (room_length : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_length = 20 →
  veranda_width = 2 →
  veranda_area = 144 →
  ∃ room_width : ℝ,
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = veranda_area ∧
    room_width = 12 := by
  sorry

end room_width_proof_l1596_159637


namespace physics_players_l1596_159602

def total_players : ℕ := 30
def math_players : ℕ := 15
def both_subjects : ℕ := 6

theorem physics_players :
  ∃ (physics_players : ℕ),
    physics_players = total_players - (math_players - both_subjects) ∧
    physics_players = 21 :=
by sorry

end physics_players_l1596_159602


namespace initial_average_production_l1596_159603

theorem initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℚ) 
  (h1 : n = 19)
  (h2 : today_production = 90)
  (h3 : new_average = 52) : 
  ∃ A : ℚ, A = 50 ∧ (A * n + today_production) / (n + 1) = new_average :=
by sorry

end initial_average_production_l1596_159603


namespace circle_area_difference_l1596_159600

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let d2 : ℝ := 30
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 675 * π := by sorry

end circle_area_difference_l1596_159600


namespace square_areas_and_perimeters_l1596_159696

theorem square_areas_and_perimeters (x : ℝ) : 
  (∃ (s₁ s₂ : ℝ), 
    s₁^2 = x^2 + 12*x + 36 ∧ 
    s₂^2 = 4*x^2 - 12*x + 9 ∧ 
    4*s₁ + 4*s₂ = 64) → 
  x = 13/3 := by
sorry

end square_areas_and_perimeters_l1596_159696


namespace plan_y_cost_effective_l1596_159650

/-- The cost in cents for Plan X given the number of minutes used -/
def planXCost (minutes : ℕ) : ℕ := 15 * minutes

/-- The cost in cents for Plan Y given the number of minutes used -/
def planYCost (minutes : ℕ) : ℕ := 3000 + 10 * minutes

/-- The minimum number of minutes for Plan Y to be cost-effective -/
def minMinutes : ℕ := 601

theorem plan_y_cost_effective : 
  ∀ m : ℕ, m ≥ minMinutes → planYCost m < planXCost m :=
by
  sorry

#check plan_y_cost_effective

end plan_y_cost_effective_l1596_159650


namespace bus_fare_difference_sam_alex_l1596_159643

/-- The cost difference between two people's bus fares for a given number of trips -/
def busFareDifference (alexFare samFare : ℚ) (numTrips : ℕ) : ℚ :=
  numTrips * (samFare - alexFare)

/-- Theorem stating the cost difference between Sam and Alex's bus fares for 20 trips -/
theorem bus_fare_difference_sam_alex :
  busFareDifference (25/10) 3 20 = 15 := by sorry

end bus_fare_difference_sam_alex_l1596_159643


namespace circle_radius_when_perimeter_equals_area_l1596_159607

/-- Given a square and its circumscribed circle, if the perimeter of the square in inches
    equals the area of the circle in square inches, then the radius of the circle is 8/π inches. -/
theorem circle_radius_when_perimeter_equals_area (s : ℝ) (r : ℝ) :
  s > 0 → r > 0 → s = 2 * r → 4 * s = π * r^2 → r = 8 / π :=
by sorry

end circle_radius_when_perimeter_equals_area_l1596_159607


namespace cosine_arcsine_tangent_arccos_equation_l1596_159616

theorem cosine_arcsine_tangent_arccos_equation :
  ∃! x : ℝ, x ∈ [(-1 : ℝ), 1] ∧
    Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x ∧
    x = 1 := by
  sorry

end cosine_arcsine_tangent_arccos_equation_l1596_159616


namespace arrangement_count_l1596_159644

theorem arrangement_count (volunteers : ℕ) (elderly : ℕ) : 
  volunteers = 4 ∧ elderly = 1 → (volunteers.factorial : ℕ) = 24 := by
  sorry

end arrangement_count_l1596_159644


namespace specific_arithmetic_sequence_sum_l1596_159652

/-- The sum of an arithmetic sequence with given parameters -/
def arithmetic_sequence_sum (n : ℕ) (a : ℤ) (d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Theorem: The sum of the specific arithmetic sequence is 18599100 -/
theorem specific_arithmetic_sequence_sum :
  arithmetic_sequence_sum 2008 (-1776) 11 = 18599100 := by
  sorry

end specific_arithmetic_sequence_sum_l1596_159652


namespace rods_per_sheet_is_correct_l1596_159688

/-- Represents the number of metal rods in each metal sheet -/
def rods_per_sheet : ℕ := 10

/-- Represents the number of metal sheets in each fence panel -/
def sheets_per_panel : ℕ := 3

/-- Represents the number of metal beams in each fence panel -/
def beams_per_panel : ℕ := 2

/-- Represents the total number of fence panels -/
def total_panels : ℕ := 10

/-- Represents the number of metal rods in each metal beam -/
def rods_per_beam : ℕ := 4

/-- Represents the total number of metal rods needed for the entire fence -/
def total_rods : ℕ := 380

/-- Theorem stating that the number of rods per sheet is correct given the conditions -/
theorem rods_per_sheet_is_correct :
  rods_per_sheet * (sheets_per_panel * total_panels) + 
  rods_per_beam * (beams_per_panel * total_panels) = total_rods :=
by sorry

end rods_per_sheet_is_correct_l1596_159688


namespace three_witnesses_are_liars_l1596_159622

-- Define the type for witnesses
inductive Witness : Type
  | one
  | two
  | three
  | four

-- Define a function to represent the statement of each witness
def statement (w : Witness) : Nat :=
  match w with
  | Witness.one => 1
  | Witness.two => 2
  | Witness.three => 3
  | Witness.four => 4

-- Define a predicate to check if a witness is telling the truth
def isTruthful (w : Witness) (numLiars : Nat) : Prop :=
  statement w = numLiars

-- Theorem: Exactly three witnesses are liars
theorem three_witnesses_are_liars :
  ∃! (numLiars : Nat), 
    numLiars = 3 ∧
    (∃! (truthful : Witness), 
      isTruthful truthful numLiars ∧
      ∀ (w : Witness), w ≠ truthful → ¬(isTruthful w numLiars)) :=
by
  sorry


end three_witnesses_are_liars_l1596_159622


namespace evaluate_expression_l1596_159604

theorem evaluate_expression : 1234562 - (12 * 3 * (2 + 7)) = 1234238 := by
  sorry

end evaluate_expression_l1596_159604


namespace shorter_base_length_l1596_159662

/-- A trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midline_length : ℝ

/-- The property that the line joining the midpoints of the diagonals 
    is half the difference of the bases -/
def midline_property (t : Trapezoid) : Prop :=
  t.midline_length = (t.long_base - t.short_base) / 2

/-- Theorem stating the length of the shorter base given the conditions -/
theorem shorter_base_length (t : Trapezoid) 
  (h1 : t.long_base = 115)
  (h2 : t.midline_length = 6)
  (h3 : midline_property t) : 
  t.short_base = 103 := by
sorry

end shorter_base_length_l1596_159662


namespace intersection_dot_product_l1596_159649

/-- Given an ellipse and a hyperbola with common foci, the dot product of vectors from their intersection point to the foci is 21. -/
theorem intersection_dot_product (x y : ℝ) (F₁ F₂ P : ℝ × ℝ) : 
  x^2/25 + y^2/16 = 1 →  -- Ellipse equation
  x^2/4 - y^2/5 = 1 →    -- Hyperbola equation
  P = (x, y) →           -- P is on both curves
  (∃ c : ℝ, c > 0 ∧ 
    (F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 = (5 + c)^2 ∧ 
    (F₂.1 - P.1)^2 + (F₂.2 - P.2)^2 = (5 - c)^2 ∧ 
    (F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 = (2 + c)^2 ∧ 
    (F₂.1 - P.1)^2 + (F₂.2 - P.2)^2 = (c - 2)^2) →  -- Common foci condition
  ((F₁.1 - P.1) * (F₂.1 - P.1) + (F₁.2 - P.2) * (F₂.2 - P.2) : ℝ) = 21 :=
by sorry

end intersection_dot_product_l1596_159649


namespace negation_of_universal_proposition_l1596_159699

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x > 0) ↔ (∃ x : ℝ, x^2 - 2*x ≤ 0) := by sorry

end negation_of_universal_proposition_l1596_159699


namespace fraction_less_than_mode_l1596_159619

def data_list : List ℕ := [1, 2, 3, 4, 5, 5, 5, 5, 7, 11, 21]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

def count_less_than (l : List ℕ) (n : ℕ) : ℕ :=
  l.filter (· < n) |>.length

theorem fraction_less_than_mode :
  (count_less_than data_list (mode data_list) : ℚ) / data_list.length = 1 / 3 := by
  sorry

end fraction_less_than_mode_l1596_159619


namespace smallest_power_congruence_l1596_159623

theorem smallest_power_congruence (n : ℕ) : 
  (∀ m : ℕ, 0 < m → m < 100 → (2013 ^ m) % 1000 ≠ 1) ∧ 
  (2013 ^ 100) % 1000 = 1 :=
sorry

end smallest_power_congruence_l1596_159623


namespace parabola_vertex_fourth_quadrant_l1596_159635

/-- A parabola with equation y = -2(x+a)^2 + c -/
structure Parabola (a c : ℝ) where
  equation : ℝ → ℝ
  eq_def : ∀ x, equation x = -2 * (x + a)^2 + c

/-- The vertex of a parabola -/
def vertex (p : Parabola a c) : ℝ × ℝ := (-a, c)

/-- A point is in the fourth quadrant if its x-coordinate is positive and y-coordinate is negative -/
def in_fourth_quadrant (point : ℝ × ℝ) : Prop :=
  point.1 > 0 ∧ point.2 < 0

/-- Theorem: For a parabola y = -2(x+a)^2 + c with its vertex in the fourth quadrant, a < 0 and c < 0 -/
theorem parabola_vertex_fourth_quadrant {a c : ℝ} (p : Parabola a c) 
  (h : in_fourth_quadrant (vertex p)) : a < 0 ∧ c < 0 := by
  sorry


end parabola_vertex_fourth_quadrant_l1596_159635


namespace disjoint_subsets_remainder_l1596_159654

def T : Finset Nat := Finset.range 15

def disjoint_subsets (S : Finset Nat) : Nat :=
  (3^S.card - 2 * 2^S.card + 1) / 2

theorem disjoint_subsets_remainder (T : Finset Nat) (h : T = Finset.range 15) :
  disjoint_subsets T % 1000 = 686 := by
  sorry

end disjoint_subsets_remainder_l1596_159654


namespace divisors_of_500_l1596_159636

theorem divisors_of_500 : 
  ∃ (S : Finset Nat), 
    (∀ n ∈ S, n ∣ 500 ∧ 1 ≤ n ∧ n ≤ 500) ∧ 
    (∀ n, n ∣ 500 ∧ 1 ≤ n ∧ n ≤ 500 → n ∈ S) ∧ 
    Finset.card S = 12 := by
  sorry

end divisors_of_500_l1596_159636


namespace part_one_part_two_l1596_159687

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Theorem for the first part
theorem part_one (m : ℝ) : 
  (∀ x : ℝ, m + f x > 0) ↔ m > -2 :=
sorry

-- Theorem for the second part
theorem part_two (m : ℝ) :
  (∃ x : ℝ, m - f x > 0) ↔ m > 2 :=
sorry

end part_one_part_two_l1596_159687


namespace age_ratio_problem_l1596_159639

theorem age_ratio_problem (j e : ℕ) (h1 : j - 6 = 4 * (e - 6)) (h2 : j - 4 = 3 * (e - 4)) :
  ∃ x : ℕ, x = 14 ∧ (j + x) * 2 = (e + x) * 3 :=
sorry

end age_ratio_problem_l1596_159639


namespace keaton_apple_harvest_interval_l1596_159618

/-- Represents Keaton's farm and harvesting schedule -/
structure Farm where
  orange_harvest_interval : ℕ  -- months between orange harvests
  orange_harvest_value : ℕ     -- value of each orange harvest in dollars
  apple_harvest_value : ℕ      -- value of each apple harvest in dollars
  total_yearly_earnings : ℕ    -- total earnings per year in dollars

/-- Calculates how often Keaton can harvest his apples -/
def apple_harvest_interval (f : Farm) : ℕ :=
  12 / ((f.total_yearly_earnings - (12 / f.orange_harvest_interval * f.orange_harvest_value)) / f.apple_harvest_value)

/-- Theorem stating that Keaton can harvest his apples every 3 months -/
theorem keaton_apple_harvest_interval :
  ∀ (f : Farm),
  f.orange_harvest_interval = 2 →
  f.orange_harvest_value = 50 →
  f.apple_harvest_value = 30 →
  f.total_yearly_earnings = 420 →
  apple_harvest_interval f = 3 := by
  sorry

end keaton_apple_harvest_interval_l1596_159618


namespace complement_intersection_theorem_l1596_159614

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {1, 3, 5}

-- Define set B
def B : Set Nat := {2, 5, 7}

-- State the theorem
theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2, 7} := by sorry

end complement_intersection_theorem_l1596_159614


namespace max_boat_shipments_l1596_159642

theorem max_boat_shipments (B : ℕ) (h1 : B ≥ 120) (h2 : B % 24 = 0) :
  ∃ S : ℕ, S ≠ 24 ∧ B % S = 0 ∧ ∀ T : ℕ, T ≠ 24 → B % T = 0 → T ≤ S :=
by
  sorry

end max_boat_shipments_l1596_159642


namespace total_cookies_l1596_159671

/-- The number of baking trays Lara is using. -/
def num_trays : ℕ := 4

/-- The number of rows of cookies on each tray. -/
def rows_per_tray : ℕ := 5

/-- The number of cookies in one row. -/
def cookies_per_row : ℕ := 6

/-- Theorem: The total number of cookies Lara is baking is 120. -/
theorem total_cookies : 
  num_trays * rows_per_tray * cookies_per_row = 120 := by
  sorry

end total_cookies_l1596_159671


namespace cricket_team_handedness_l1596_159658

theorem cricket_team_handedness (total_players : Nat) (throwers : Nat) (right_handed : Nat)
    (h1 : total_players = 58)
    (h2 : throwers = 37)
    (h3 : right_handed = 51)
    (h4 : throwers ≤ right_handed) :
    (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 := by
  sorry

end cricket_team_handedness_l1596_159658


namespace chinese_spanish_difference_l1596_159626

def hours_english : ℕ := 2
def hours_chinese : ℕ := 5
def hours_spanish : ℕ := 4

theorem chinese_spanish_difference : hours_chinese - hours_spanish = 1 := by
  sorry

end chinese_spanish_difference_l1596_159626


namespace parallel_vectors_imply_x_equals_two_l1596_159620

def a : ℝ × ℝ := (1, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

theorem parallel_vectors_imply_x_equals_two (x : ℝ) :
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b x) = k • (4 • (b x) - 2 • a)) →
  x = 2 := by
sorry

end parallel_vectors_imply_x_equals_two_l1596_159620


namespace lap_time_improvement_l1596_159605

/-- Represents the performance data for a runner -/
structure Performance where
  laps : ℕ
  time : ℕ  -- time in minutes

/-- Calculates the lap time in seconds given a Performance -/
def lapTimeInSeconds (p : Performance) : ℚ :=
  (p.time * 60) / p.laps

theorem lap_time_improvement (initial : Performance) (current : Performance) 
  (h1 : initial.laps = 8) (h2 : initial.time = 36)
  (h3 : current.laps = 10) (h4 : current.time = 35) :
  lapTimeInSeconds initial - lapTimeInSeconds current = 60 := by
  sorry

end lap_time_improvement_l1596_159605


namespace sequence_inequality_l1596_159676

theorem sequence_inequality (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_ineq : ∀ n, a (n + 2) ≤ (2023 * a n) / (a n * a (n + 1) + 2023)) :
  a 2023 < 1 ∨ a 2024 < 1 := by
sorry

end sequence_inequality_l1596_159676


namespace inscribed_rectangle_area_l1596_159631

/-- The area of a rectangle inscribed in a trapezoid -/
theorem inscribed_rectangle_area (a b h x : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) (hh : 0 < h) (hx : 0 < x) (hxh : x < h) :
  let area := x * (a - b) * (h - x) / h
  area = x * (a - b) * (h - x) / h :=
by sorry

end inscribed_rectangle_area_l1596_159631


namespace minimum_value_implies_a_l1596_159640

def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-5 : ℝ) 5, f a x ≥ -3) ∧
  (∃ x ∈ Set.Icc (-5 : ℝ) 5, f a x = -3) →
  a = Real.sqrt 5 ∨ a = -Real.sqrt 5 :=
by sorry

end minimum_value_implies_a_l1596_159640


namespace unique_prime_power_condition_l1596_159689

theorem unique_prime_power_condition : ∃! p : ℕ, 
  p ≤ 1000 ∧ 
  Nat.Prime p ∧ 
  ∃ (m n : ℕ), n ≥ 2 ∧ 2 * p + 1 = m ^ n ∧
  p = 13 :=
sorry

end unique_prime_power_condition_l1596_159689


namespace equality_equivalence_l1596_159610

theorem equality_equivalence (a b c : ℝ) : 
  (a * (b - c)) / (b + c) + (b * (c - a)) / (c + a) + (c * (a - b)) / (a + b) = 0 ↔ 
  (a^2 * (b - c)) / (b + c) + (b^2 * (c - a)) / (c + a) + (c^2 * (a - b)) / (a + b) = 0 :=
by sorry

end equality_equivalence_l1596_159610


namespace imaginary_part_of_complex_fraction_l1596_159675

theorem imaginary_part_of_complex_fraction : Complex.im (5 * Complex.I / (1 + 2 * Complex.I)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l1596_159675


namespace min_value_sum_reciprocals_l1596_159645

/-- Given a quadratic function f(x) = ax^2 - x + c with range [0, +∞),
    the minimum value of 2/a + 2/c is 8 -/
theorem min_value_sum_reciprocals (a c : ℝ) : 
  (∀ x, ∃ y ≥ 0, y = a * x^2 - x + c) →
  (∃ x, a * x^2 - x + c = 0) →
  (∀ x, a * x^2 - x + c ≥ 0) →
  (2 / a + 2 / c ≥ 8) ∧ (∃ a c, 2 / a + 2 / c = 8) := by
sorry

end min_value_sum_reciprocals_l1596_159645


namespace kareem_largest_l1596_159651

def jose_calc (x : Int) : Int :=
  ((x - 2) * 3) + 5

def thuy_calc (x : Int) : Int :=
  (x * 3 - 2) + 5

def kareem_calc (x : Int) : Int :=
  ((x - 2) + 5) * 3

theorem kareem_largest (start : Int) :
  start = 15 →
  kareem_calc start > jose_calc start ∧
  kareem_calc start > thuy_calc start :=
by
  sorry

#eval jose_calc 15
#eval thuy_calc 15
#eval kareem_calc 15

end kareem_largest_l1596_159651


namespace segment_construction_l1596_159601

/-- A list of 99 natural numbers from 1 to 99 -/
def segments : List ℕ := List.range 99

/-- The sum of all segments -/
def total_length : ℕ := List.sum segments

/-- Predicate to check if a square can be formed -/
def can_form_square (segs : List ℕ) : Prop :=
  ∃ (side : ℕ), 4 * side = List.sum segs

/-- Predicate to check if a rectangle can be formed -/
def can_form_rectangle (segs : List ℕ) : Prop :=
  ∃ (length width : ℕ), length * width = List.sum segs ∧ length ≠ width

/-- Predicate to check if an equilateral triangle can be formed -/
def can_form_equilateral_triangle (segs : List ℕ) : Prop :=
  ∃ (side : ℕ), 3 * side = List.sum segs

theorem segment_construction :
  ¬ can_form_square segments ∧
  can_form_rectangle segments ∧
  can_form_equilateral_triangle segments :=
sorry

end segment_construction_l1596_159601


namespace polygon_sides_l1596_159667

theorem polygon_sides (exterior_angle : ℝ) (h : exterior_angle = 40) :
  (360 : ℝ) / exterior_angle = 9 := by
  sorry

end polygon_sides_l1596_159667


namespace line_equation_from_parametric_l1596_159664

/-- The equation of a line parameterized by (3t + 6, 5t - 7) is y = (5/3)x - 17 -/
theorem line_equation_from_parametric : 
  ∀ (t x y : ℝ), x = 3 * t + 6 ∧ y = 5 * t - 7 → y = (5/3) * x - 17 := by
  sorry

end line_equation_from_parametric_l1596_159664


namespace set_equivalence_l1596_159691

theorem set_equivalence (U A B : Set α) :
  (A ∩ B = A) ↔ (A ⊆ U ∧ B ⊆ U ∧ (Uᶜ ∩ B)ᶜ ⊆ (Uᶜ ∩ A)ᶜ) := by sorry

end set_equivalence_l1596_159691


namespace triangle_side_length_l1596_159621

-- Define the triangle DEF
structure Triangle (D E F : ℝ) where
  -- Angle sum property of a triangle
  angle_sum : D + E + F = Real.pi

-- Define the main theorem
theorem triangle_side_length 
  (D E F : ℝ) 
  (t : Triangle D E F) 
  (h1 : Real.cos (3 * D - E) + Real.sin (D + E) = 2) 
  (h2 : 6 = 6) :  -- DE = 6, but we use 6 = 6 as Lean doesn't know DE yet
  ∃ (EF : ℝ), EF = 3 * Real.sqrt (2 - Real.sqrt 2) :=
sorry

end triangle_side_length_l1596_159621


namespace candies_left_theorem_l1596_159674

/-- Calculates the number of candies left to be shared with others --/
def candies_left_to_share (initial_candies : ℕ) (siblings : ℕ) (candies_per_sibling : ℕ) (candies_to_eat : ℕ) : ℕ :=
  let candies_after_siblings := initial_candies - siblings * candies_per_sibling
  let candies_after_friend := candies_after_siblings / 2
  candies_after_friend - candies_to_eat

/-- Proves that given the initial conditions, the number of candies left to be shared with others is 19 --/
theorem candies_left_theorem :
  candies_left_to_share 100 3 10 16 = 19 := by
  sorry

end candies_left_theorem_l1596_159674


namespace m_range_l1596_159663

theorem m_range (m : ℝ) :
  (m^2 + m)^(3/5) ≤ (3 - m)^(3/5) → -3 ≤ m ∧ m ≤ 1 :=
by
  sorry

end m_range_l1596_159663


namespace evaluate_expression_l1596_159613

theorem evaluate_expression : 8^7 + 8^7 + 8^7 - 8^7 = 8^8 := by
  sorry

end evaluate_expression_l1596_159613


namespace intersection_A_B_intersection_A_complement_B_l1596_159665

-- Define the sets A and B
def A : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -2 ∨ 4 < x}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | -5 ≤ x ∧ x < -2} := by sorry

-- Theorem for the intersection of A and the complement of B
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

end intersection_A_B_intersection_A_complement_B_l1596_159665


namespace modulus_of_complex_product_l1596_159693

theorem modulus_of_complex_product : 
  let i : ℂ := Complex.I
  let z : ℂ := (3 - i) * (1 + 3*i)
  Complex.abs z = 10 := by sorry

end modulus_of_complex_product_l1596_159693


namespace right_triangle_hypotenuse_l1596_159678

/-- Given a right triangle with medians from acute angles 6 and √30, prove the hypotenuse is 2√52.8 -/
theorem right_triangle_hypotenuse (a b : ℝ) (h_right : a^2 + b^2 = (a + b)^2 / 4)
  (h_median1 : b^2 + (a/2)^2 = 30) (h_median2 : a^2 + (b/2)^2 = 36) :
  (2*a)^2 + (2*b)^2 = 4 * 52.8 := by
sorry

end right_triangle_hypotenuse_l1596_159678


namespace triangle_centroid_existence_and_property_l1596_159647

/-- Given a triangle ABC, there exists a unique point O (the centroid) that lies on all medians and divides each in a 2:1 ratio from the vertex. -/
theorem triangle_centroid_existence_and_property (A B C : EuclideanSpace ℝ (Fin 2)) :
  ∃! O : EuclideanSpace ℝ (Fin 2),
    (∃ t : ℝ, O = A + t • (midpoint ℝ B C - A)) ∧
    (∃ u : ℝ, O = B + u • (midpoint ℝ A C - B)) ∧
    (∃ v : ℝ, O = C + v • (midpoint ℝ A B - C)) ∧
    (O = A + (2/3) • (midpoint ℝ B C - A)) ∧
    (O = B + (2/3) • (midpoint ℝ A C - B)) ∧
    (O = C + (2/3) • (midpoint ℝ A B - C)) := by
  sorry

end triangle_centroid_existence_and_property_l1596_159647


namespace neither_sufficient_nor_necessary_l1596_159625

open Real

/-- A function f : ℝ → ℝ is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem neither_sufficient_nor_necessary
  (a : ℝ)
  (ha : a > 0)
  (ha_neq : a ≠ 1) :
  ¬(IsIncreasing (fun x ↦ a^x) → IsIncreasing (fun x ↦ x^a)) ∧
  ¬(IsIncreasing (fun x ↦ x^a) → IsIncreasing (fun x ↦ a^x)) := by
  sorry

end neither_sufficient_nor_necessary_l1596_159625


namespace divisible_by_4_6_9_less_than_300_l1596_159661

theorem divisible_by_4_6_9_less_than_300 : 
  (Finset.filter (fun n : ℕ => n % 4 = 0 ∧ n % 6 = 0 ∧ n % 9 = 0) (Finset.range 300)).card = 8 :=
by sorry

end divisible_by_4_6_9_less_than_300_l1596_159661


namespace number_divided_by_six_l1596_159682

theorem number_divided_by_six : ∃ n : ℝ, n / 6 = 26 ∧ n = 156 := by
  sorry

end number_divided_by_six_l1596_159682


namespace dataset_transformation_l1596_159697

/-- Represents a dataset with mean and variance -/
structure Dataset where
  mean : ℝ
  variance : ℝ

/-- Represents the transformation of adding a constant to each data point -/
def add_constant (d : Dataset) (c : ℝ) : Dataset :=
  { mean := d.mean + c,
    variance := d.variance }

theorem dataset_transformation (d : Dataset) :
  d.mean = 2.8 →
  d.variance = 3.6 →
  (add_constant d 60).mean = 62.8 ∧ (add_constant d 60).variance = 3.6 := by
  sorry

end dataset_transformation_l1596_159697


namespace termite_ridden_not_collapsing_l1596_159632

theorem termite_ridden_not_collapsing (total_homes : ℕ) (termite_ridden : ℕ) (collapsing : ℕ)
  (h1 : termite_ridden = total_homes / 3)
  (h2 : collapsing = termite_ridden * 5 / 8) :
  (termite_ridden - collapsing : ℚ) / total_homes = 1 / 8 := by
  sorry

end termite_ridden_not_collapsing_l1596_159632


namespace pizza_slices_left_l1596_159606

theorem pizza_slices_left (total_slices : ℕ) (john_slices : ℕ) (sam_multiplier : ℕ) : 
  total_slices = 12 →
  john_slices = 3 →
  sam_multiplier = 2 →
  total_slices - (john_slices + sam_multiplier * john_slices) = 3 :=
by sorry

end pizza_slices_left_l1596_159606


namespace quadratic_rational_solution_l1596_159692

/-- The quadratic equation kx^2 + 16x + k = 0 has rational solutions if and only if k = 8, where k is a positive integer. -/
theorem quadratic_rational_solution (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 16 * x + k = 0) ↔ k = 8 := by
sorry

end quadratic_rational_solution_l1596_159692


namespace sum_of_products_l1596_159656

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + x*y + y^2 = 12)
  (h2 : y^2 + y*z + z^2 = 25)
  (h3 : z^2 + x*z + x^2 = 37) :
  x*y + y*z + x*z = 20 :=
by sorry

end sum_of_products_l1596_159656


namespace ram_selection_probability_l1596_159630

/-- Given two brothers Ram and Ravi, where the probability of Ravi's selection is 1/5
    and the probability of both being selected is 0.11428571428571428,
    prove that the probability of Ram's selection is 0.5714285714285714 -/
theorem ram_selection_probability
  (p_ravi : ℝ)
  (p_both : ℝ)
  (h1 : p_ravi = 1 / 5)
  (h2 : p_both = 0.11428571428571428) :
  p_both / p_ravi = 0.5714285714285714 := by
  sorry

end ram_selection_probability_l1596_159630


namespace old_man_coins_l1596_159638

theorem old_man_coins (x y : ℕ) (h1 : x ≠ y) (h2 : x^2 - y^2 = 25 * (x - y)) : x + y = 25 := by
  sorry

end old_man_coins_l1596_159638


namespace intersection_A_B_l1596_159648

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 3}
def B : Set ℝ := {x | x^2 ≥ 4}

-- Define the interval [2, 3)
def interval_2_3 : Set ℝ := {x | 2 ≤ x ∧ x < 3}

-- Theorem statement
theorem intersection_A_B : A ∩ B = interval_2_3 := by
  sorry

end intersection_A_B_l1596_159648


namespace mean_value_point_of_cubic_minus_linear_l1596_159659

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) := x^3 - 3*x

-- Define the derivative of f(x)
def f' (x : ℝ) := 3*x^2 - 3

-- Define the mean value point property
def is_mean_value_point (f : ℝ → ℝ) (f' : ℝ → ℝ) (a b x₀ : ℝ) : Prop :=
  f b - f a = f' x₀ * (b - a)

theorem mean_value_point_of_cubic_minus_linear :
  ∃ x₀ : ℝ, is_mean_value_point f f' (-2) 2 x₀ ∧ x₀^2 = 1/3 := by
  sorry


end mean_value_point_of_cubic_minus_linear_l1596_159659


namespace man_crossing_bridge_l1596_159670

/-- Proves that a man walking at 10 km/hr takes 10 minutes to cross a 1666.6666666666665 meter bridge -/
theorem man_crossing_bridge 
  (walking_rate : ℝ) 
  (bridge_length : ℝ) 
  (h1 : walking_rate = 10) -- km/hr
  (h2 : bridge_length = 1666.6666666666665) -- meters
  : (bridge_length / (walking_rate * 1000 / 60)) = 10 := by
  sorry

#check man_crossing_bridge

end man_crossing_bridge_l1596_159670


namespace lcm_1332_888_l1596_159680

theorem lcm_1332_888 : Nat.lcm 1332 888 = 2664 := by
  sorry

end lcm_1332_888_l1596_159680


namespace factorization_equality_l1596_159673

theorem factorization_equality (a b : ℝ) : a^2 * b - 2*a*b + b = b * (a - 1)^2 := by
  sorry

end factorization_equality_l1596_159673


namespace calculation_proof_l1596_159669

theorem calculation_proof : 10 - 9 * 8 / 4 + 7 - 6 * 5 + 3 - 2 * 1 = -30 := by
  sorry

end calculation_proof_l1596_159669


namespace lisa_flight_distance_l1596_159666

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Lisa's flight distance -/
theorem lisa_flight_distance :
  let speed : ℝ := 32
  let time : ℝ := 8
  distance speed time = 256 := by sorry

end lisa_flight_distance_l1596_159666


namespace briannes_yard_length_l1596_159677

theorem briannes_yard_length (derricks_length : ℝ) (alexs_length : ℝ) (briannes_length : ℝ) : 
  derricks_length = 10 →
  alexs_length = derricks_length / 2 →
  briannes_length = 6 * alexs_length →
  briannes_length = 30 := by sorry

end briannes_yard_length_l1596_159677


namespace sin_cos_sum_zero_l1596_159683

theorem sin_cos_sum_zero : 
  Real.sin (35 * π / 6) + Real.cos (-11 * π / 3) = 0 := by
  sorry

end sin_cos_sum_zero_l1596_159683


namespace albrecht_equation_solutions_l1596_159608

theorem albrecht_equation_solutions :
  ∀ a b : ℕ+, 
    (a + 2*b - 3)^2 = a^2 + 4*b^2 - 9 ↔ 
    ((a = 2 ∧ b = 15) ∨ (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 3) ∨ (a = 15 ∧ b = 2)) :=
by sorry

end albrecht_equation_solutions_l1596_159608


namespace range_of_k_with_two_preimages_l1596_159641

def f (x : ℝ) : ℝ := -x^2 + 2*x

theorem range_of_k_with_two_preimages :
  ∀ k : ℝ, (∃ x y : ℝ, x ≠ y ∧ f x = k ∧ f y = k) → k < 1 :=
by sorry

end range_of_k_with_two_preimages_l1596_159641

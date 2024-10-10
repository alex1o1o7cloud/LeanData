import Mathlib

namespace village_population_l3874_387473

theorem village_population (partial_population : ℕ) (partial_percentage : ℚ) (total_population : ℕ) :
  partial_percentage = 9/10 →
  partial_population = 36000 →
  total_population * partial_percentage = partial_population →
  total_population = 40000 := by
sorry

end village_population_l3874_387473


namespace inequality_proof_l3874_387449

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_3 : a + b + c = 3) :
  (a^2 + 9) / (2*a^2 + (b+c)^2) + (b^2 + 9) / (2*b^2 + (c+a)^2) + (c^2 + 9) / (2*c^2 + (a+b)^2) ≤ 5 := by
  sorry

end inequality_proof_l3874_387449


namespace max_sum_under_constraints_l3874_387454

theorem max_sum_under_constraints (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 10) : 
  x + y ≤ 93 / 44 := by
  sorry

end max_sum_under_constraints_l3874_387454


namespace optimal_division_l3874_387475

theorem optimal_division (a : ℝ) (h : a > 0) :
  let f := fun (x : ℝ) => x / (a - x) + (a - x) / x
  ∃ (x : ℝ), 0 < x ∧ x < a ∧ ∀ (y : ℝ), 0 < y ∧ y < a → f x ≤ f y :=
by sorry

end optimal_division_l3874_387475


namespace largest_multiple_of_8_under_100_l3874_387456

theorem largest_multiple_of_8_under_100 : ∃ n : ℕ, n * 8 = 96 ∧ 
  (∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96) := by
  sorry

end largest_multiple_of_8_under_100_l3874_387456


namespace irrational_pi_only_l3874_387413

theorem irrational_pi_only (a b c d : ℝ) : 
  a = 1 / 7 → b = Real.pi → c = -1 → d = 0 → 
  (¬ Irrational a ∧ Irrational b ∧ ¬ Irrational c ∧ ¬ Irrational d) := by
  sorry

end irrational_pi_only_l3874_387413


namespace additional_birds_l3874_387444

theorem additional_birds (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 231)
  (h2 : final_birds = 312) :
  final_birds - initial_birds = 81 := by
  sorry

end additional_birds_l3874_387444


namespace range_of_a_l3874_387447

/-- Given a ≥ -2, prove that if C ⊆ B, then a ∈ [1/2, 3] -/
theorem range_of_a (a : ℝ) (ha : a ≥ -2) :
  let A := {x : ℝ | -2 ≤ x ∧ x ≤ a}
  let B := {y : ℝ | ∃ x ∈ A, y = 2 * x + 3}
  let C := {t : ℝ | ∃ x ∈ A, t = x^2}
  C ⊆ B → a ∈ Set.Icc (1/2 : ℝ) 3 := by
  sorry

end range_of_a_l3874_387447


namespace left_handed_rock_fans_under_25_l3874_387459

/-- Represents the number of people with specific characteristics in a workshop. -/
structure WorkshopPeople where
  total : ℕ
  leftHanded : ℕ
  rockMusicFans : ℕ
  rightHandedNotRockFans : ℕ
  under25 : ℕ
  rightHandedUnder25RockFans : ℕ

/-- Theorem stating the number of left-handed, rock music fans under 25 in the workshop. -/
theorem left_handed_rock_fans_under_25 (w : WorkshopPeople) 
  (h1 : w.total = 30)
  (h2 : w.leftHanded = 12)
  (h3 : w.rockMusicFans = 18)
  (h4 : w.rightHandedNotRockFans = 5)
  (h5 : w.under25 = 9)
  (h6 : w.rightHandedUnder25RockFans = 3)
  (h7 : w.leftHanded + (w.total - w.leftHanded) = w.total) :
  ∃ x : ℕ, x = 5 ∧ 
    x + (w.leftHanded - x) + (w.rockMusicFans - x) + w.rightHandedNotRockFans + 
    w.rightHandedUnder25RockFans + (w.total - w.leftHanded - w.rightHandedNotRockFans - 
    w.rightHandedUnder25RockFans - x) = w.total :=
  sorry


end left_handed_rock_fans_under_25_l3874_387459


namespace coin_toss_probability_l3874_387484

def toss_outcomes (n : ℕ) : ℕ := 2^n

def favorable_outcomes : ℕ := 5

theorem coin_toss_probability :
  let mina_tosses : ℕ := 2
  let liam_tosses : ℕ := 3
  let total_outcomes : ℕ := toss_outcomes mina_tosses * toss_outcomes liam_tosses
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 32 := by sorry

end coin_toss_probability_l3874_387484


namespace water_spilled_is_eight_quarts_l3874_387433

/-- Represents the water supply problem from the shipwreck scenario -/
structure WaterSupply where
  initial_people : ℕ
  initial_days : ℕ
  spill_day : ℕ
  quart_per_person_per_day : ℕ

/-- The amount of water spilled in the shipwreck scenario -/
def water_spilled (ws : WaterSupply) : ℕ :=
  ws.initial_people + 7

/-- Theorem stating that the amount of water spilled is 8 quarts -/
theorem water_spilled_is_eight_quarts (ws : WaterSupply) 
  (h1 : ws.initial_days = 13)
  (h2 : ws.quart_per_person_per_day = 1)
  (h3 : ws.spill_day = 5)
  (h4 : ws.initial_people > 0)
  : water_spilled ws = 8 := by
  sorry

#check water_spilled_is_eight_quarts

end water_spilled_is_eight_quarts_l3874_387433


namespace unique_consecutive_triangle_with_double_angle_l3874_387446

/-- Represents a triangle with side lengths (a, a+1, a+2) -/
structure ConsecutiveTriangle where
  a : ℕ
  a_pos : a > 0

/-- Calculates the cosine of an angle in a ConsecutiveTriangle using the law of cosines -/
def cos_angle (t : ConsecutiveTriangle) (side : Fin 3) : ℚ :=
  match side with
  | 0 => (t.a^2 + 6*t.a + 5) / (2*t.a^2 + 6*t.a + 4)
  | 1 => ((t.a + 1) * (t.a + 3)) / (2*t.a*(t.a + 2))
  | 2 => ((t.a - 1) * (t.a - 3)) / (2*t.a*(t.a + 1))

/-- Checks if one angle is twice another in a ConsecutiveTriangle -/
def has_double_angle (t : ConsecutiveTriangle) : Prop :=
  ∃ (i j : Fin 3), i ≠ j ∧ cos_angle t j = 2 * (cos_angle t i)^2 - 1

/-- The main theorem stating that there's a unique ConsecutiveTriangle with a double angle -/
theorem unique_consecutive_triangle_with_double_angle :
  ∃! (t : ConsecutiveTriangle), has_double_angle t :=
sorry

end unique_consecutive_triangle_with_double_angle_l3874_387446


namespace line_tangent_to_ellipse_l3874_387451

/-- The line equation y = 2mx + 2 intersects the ellipse 2x^2 + 8y^2 = 8 exactly once if and only if m^2 = 3/16 -/
theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! p : ℝ × ℝ, (2 * p.1^2 + 8 * p.2^2 = 8) ∧ (p.2 = 2 * m * p.1 + 2)) ↔ m^2 = 3/16 := by
  sorry

end line_tangent_to_ellipse_l3874_387451


namespace rectangular_solid_surface_area_l3874_387491

/-- A rectangular solid with prime edge lengths and volume 231 has surface area 262 -/
theorem rectangular_solid_surface_area : 
  ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 231 →
  2 * (a * b + b * c + a * c) = 262 := by
sorry

end rectangular_solid_surface_area_l3874_387491


namespace positive_sum_inequalities_l3874_387430

theorem positive_sum_inequalities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 4) : 
  a^2 + b^2/4 + c^2/9 ≥ 8/7 ∧ 
  1/(a+c) + 1/(a+b) + 1/(b+c) ≥ 9/8 := by
  sorry

end positive_sum_inequalities_l3874_387430


namespace polar_equations_and_intersection_ratio_l3874_387485

-- Define the line l in Cartesian coordinates
def line_l (x y : ℝ) : Prop := x = 4

-- Define the curve C in Cartesian coordinates
def curve_C (x y φ : ℝ) : Prop := x = 1 + Real.sqrt 2 * Real.cos φ ∧ y = 1 + Real.sqrt 2 * Real.sin φ

-- Define the transformation from Cartesian to polar coordinates
def to_polar (x y ρ θ : ℝ) : Prop := x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- State the theorem
theorem polar_equations_and_intersection_ratio :
  ∀ (x y ρ θ φ α : ℝ),
  (line_l x y → ρ * Real.cos θ = 4) ∧
  (curve_C x y φ → ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) ∧
  (0 < α ∧ α < Real.pi / 4 →
    ∃ (ρ_A ρ_B : ℝ),
      ρ_A = 2 * (Real.cos α + Real.sin α) ∧
      ρ_B = 4 / Real.cos α ∧
      1 / 2 < ρ_A / ρ_B ∧ ρ_A / ρ_B ≤ (Real.sqrt 2 + 1) / 4) := by
  sorry

end polar_equations_and_intersection_ratio_l3874_387485


namespace factorization_equality_l3874_387493

theorem factorization_equality (x y : ℝ) : 6 * x^3 * y^2 + 3 * x^2 * y^2 = 3 * x^2 * y^2 * (2 * x + 1) := by
  sorry

end factorization_equality_l3874_387493


namespace abs_negative_seventeen_l3874_387453

theorem abs_negative_seventeen : |(-17 : ℤ)| = 17 := by sorry

end abs_negative_seventeen_l3874_387453


namespace eagles_score_l3874_387486

/-- Given the total points and margin of victory in a basketball game, prove the losing team's score. -/
theorem eagles_score (total_points margin : ℕ) (h1 : total_points = 82) (h2 : margin = 18) :
  (total_points - margin) / 2 = 32 := by
  sorry

end eagles_score_l3874_387486


namespace sufficient_not_necessary_condition_l3874_387431

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧
  (∃ a, a ≤ 0 ∧ a^2 + a ≥ 0) := by
  sorry

end sufficient_not_necessary_condition_l3874_387431


namespace third_sibling_age_difference_l3874_387480

/-- Represents the ages of four siblings --/
structure SiblingAges where
  youngest : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ

/-- The conditions of the sibling age problem --/
def siblingAgeProblem (ages : SiblingAges) : Prop :=
  ages.youngest = 25.75 ∧
  ages.second = ages.youngest + 3 ∧
  ages.third = ages.youngest + 6 ∧
  (ages.youngest + ages.second + ages.third + ages.fourth) / 4 = 30

/-- The theorem stating that the third sibling is 6 years older than the youngest --/
theorem third_sibling_age_difference (ages : SiblingAges) :
  siblingAgeProblem ages → ages.third - ages.youngest = 6 := by
  sorry

end third_sibling_age_difference_l3874_387480


namespace ellipse_and_line_properties_l3874_387448

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A line passing through a point and intersecting an ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  intersectionPoints : Fin 2 → ℝ × ℝ

/-- The problem statement -/
theorem ellipse_and_line_properties
  (E : Ellipse)
  (l : IntersectingLine E)
  (h₁ : E.a^2 - E.b^2 = 1) -- Condition for foci at (-1,0) and (1,0)
  (h₂ : (l.intersectionPoints 0).1 + (l.intersectionPoints 1).1 +
        ((l.intersectionPoints 0).1 + 1)^2 + (l.intersectionPoints 0).2^2 +
        ((l.intersectionPoints 1).1 + 1)^2 + (l.intersectionPoints 1).2^2 = 16) -- Perimeter condition
  (h₃ : (l.intersectionPoints 0).1 * (l.intersectionPoints 1).1 +
        (l.intersectionPoints 0).2 * (l.intersectionPoints 1).2 = 0) -- Perpendicularity condition
  : (E.a = Real.sqrt 3 ∧ E.b = Real.sqrt 2) ∧
    (l.k = Real.sqrt 2 ∨ l.k = -Real.sqrt 2) :=
by sorry

end ellipse_and_line_properties_l3874_387448


namespace tree_purchase_solution_l3874_387481

/-- Represents the unit prices and purchasing schemes for tree seedlings -/
structure TreePurchase where
  osmanthus_price : ℕ
  camphor_price : ℕ
  schemes : List (ℕ × ℕ)

/-- Defines the conditions of the tree purchasing problem -/
def tree_purchase_problem (p : TreePurchase) : Prop :=
  -- First purchase condition
  10 * p.osmanthus_price + 20 * p.camphor_price = 3000 ∧
  -- Second purchase condition
  8 * p.osmanthus_price + 24 * p.camphor_price = 2800 ∧
  -- Next purchase conditions
  (∀ (o c : ℕ), (o, c) ∈ p.schemes →
    o + c = 40 ∧
    o * p.osmanthus_price + c * p.camphor_price ≤ 3800 ∧
    c ≤ 3 * o) ∧
  -- All possible schemes are included
  (∀ (o c : ℕ), o + c = 40 →
    o * p.osmanthus_price + c * p.camphor_price ≤ 3800 →
    c ≤ 3 * o →
    (o, c) ∈ p.schemes)

/-- Theorem stating the solution to the tree purchasing problem -/
theorem tree_purchase_solution :
  ∃ (p : TreePurchase),
    tree_purchase_problem p ∧
    p.osmanthus_price = 200 ∧
    p.camphor_price = 50 ∧
    p.schemes = [(10, 30), (11, 29), (12, 28)] :=
  sorry

end tree_purchase_solution_l3874_387481


namespace polar_equation_is_parabola_l3874_387411

theorem polar_equation_is_parabola :
  ∀ (r θ x y : ℝ),
  (r = 2 / (1 - Real.sin θ)) →
  (x = r * Real.cos θ) →
  (y = r * Real.sin θ) →
  ∃ (a b : ℝ), x^2 = a * y + b :=
sorry

end polar_equation_is_parabola_l3874_387411


namespace tangent_circles_count_l3874_387443

/-- Represents a line in a plane -/
structure Line where
  -- Add necessary fields for a line

/-- Represents a circle in a plane -/
structure Circle where
  -- Add necessary fields for a circle

/-- Checks if a circle is tangent to a line -/
def is_tangent (c : Circle) (l : Line) : Prop :=
  sorry

/-- Counts the number of circles tangent to all three given lines -/
def count_tangent_circles (l1 l2 l3 : Line) : Nat :=
  sorry

/-- The main theorem stating the possible values for the number of tangent circles -/
theorem tangent_circles_count (l1 l2 l3 : Line) :
  l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 →
  (count_tangent_circles l1 l2 l3 = 0 ∨
   count_tangent_circles l1 l2 l3 = 2 ∨
   count_tangent_circles l1 l2 l3 = 4) :=
by sorry

end tangent_circles_count_l3874_387443


namespace power_of_seven_mod_twelve_l3874_387403

theorem power_of_seven_mod_twelve : 7^203 % 12 = 7 := by
  sorry

end power_of_seven_mod_twelve_l3874_387403


namespace jake_balloons_l3874_387458

theorem jake_balloons (allan_balloons : ℕ) (difference : ℕ) : 
  allan_balloons = 5 → difference = 2 → allan_balloons - difference = 3 := by
  sorry

end jake_balloons_l3874_387458


namespace store_price_reduction_l3874_387427

theorem store_price_reduction (original_price : ℝ) (first_reduction : ℝ) : 
  first_reduction > 0 →
  first_reduction < 100 →
  (original_price * (1 - first_reduction / 100) * (1 - 0.14) = 0.774 * original_price) →
  first_reduction = 10 := by
  sorry

end store_price_reduction_l3874_387427


namespace consecutive_powers_of_two_divisible_by_six_l3874_387487

theorem consecutive_powers_of_two_divisible_by_six (n : ℕ) :
  6 ∣ (2^n + 2^(n+1)) := by sorry

end consecutive_powers_of_two_divisible_by_six_l3874_387487


namespace number_operation_l3874_387495

theorem number_operation (x : ℝ) : (x - 5) / 7 = 7 → (x - 34) / 10 = 2 := by
  sorry

end number_operation_l3874_387495


namespace middle_number_proof_l3874_387462

theorem middle_number_proof (a b c : ℤ) : 
  a < b ∧ b < c ∧ 
  a + b = 18 ∧ 
  a + c = 23 ∧ 
  b + c = 27 → 
  b = 11 := by
sorry

end middle_number_proof_l3874_387462


namespace picture_coverage_percentage_l3874_387478

theorem picture_coverage_percentage (poster_width poster_height picture_width picture_height : ℝ) 
  (hw_poster : poster_width = 50 ∧ poster_height = 100)
  (hw_picture : picture_width = 20 ∧ picture_height = 40) :
  (picture_width * picture_height) / (poster_width * poster_height) * 100 = 16 := by
  sorry

end picture_coverage_percentage_l3874_387478


namespace ryan_to_bill_ratio_l3874_387465

/-- Represents the number of math problems composed by each person -/
structure ProblemCounts where
  bill : ℕ
  ryan : ℕ
  frank : ℕ

/-- Represents the conditions of the problem -/
def problem_conditions (p : ProblemCounts) : Prop :=
  p.bill = 20 ∧
  p.frank = 3 * p.ryan ∧
  p.frank = 30 * 4

/-- The theorem to be proved -/
theorem ryan_to_bill_ratio 
  (p : ProblemCounts) 
  (h : problem_conditions p) : 
  p.ryan / p.bill = 2 := by
  sorry

#check ryan_to_bill_ratio

end ryan_to_bill_ratio_l3874_387465


namespace total_unique_eagle_types_l3874_387422

/-- The number of unique types of eagles across all sections -/
def uniqueEagleTypes (sectionA sectionB sectionC sectionD sectionE : ℝ)
  (overlapAB overlapBC overlapCD overlapDE overlapACE : ℝ) : ℝ :=
  sectionA + sectionB + sectionC + sectionD + sectionE - 
  (overlapAB + overlapBC + overlapCD + overlapDE - overlapACE)

/-- Theorem stating the total number of unique eagle types -/
theorem total_unique_eagle_types :
  uniqueEagleTypes 12.5 8.3 10.7 14.2 17.1 3.5 2.1 3.7 4.4 1.5 = 51.6 := by
  sorry

end total_unique_eagle_types_l3874_387422


namespace remaining_two_average_l3874_387423

theorem remaining_two_average (n₁ n₂ n₃ n₄ n₅ n₆ : ℝ) :
  (n₁ + n₂ + n₃ + n₄ + n₅ + n₆) / 6 = 4.60 →
  (n₁ + n₂) / 2 = 3.4 →
  (n₃ + n₄) / 2 = 3.8 →
  (n₅ + n₆) / 2 = 6.6 :=
by sorry

end remaining_two_average_l3874_387423


namespace sandys_change_l3874_387424

/-- Represents the cost of a drink order -/
structure DrinkOrder where
  cappuccino : ℕ
  icedTea : ℕ
  cafeLatte : ℕ
  espresso : ℕ

/-- Calculates the total cost of a drink order -/
def totalCost (order : DrinkOrder) : ℚ :=
  2 * order.cappuccino + 3 * order.icedTea + 1.5 * order.cafeLatte + 1 * order.espresso

/-- Calculates the change received from a given payment -/
def changeReceived (payment : ℚ) (order : DrinkOrder) : ℚ :=
  payment - totalCost order

/-- Sandy's specific drink order -/
def sandysOrder : DrinkOrder :=
  { cappuccino := 3
  , icedTea := 2
  , cafeLatte := 2
  , espresso := 2 }

/-- Theorem stating that Sandy receives $3 in change -/
theorem sandys_change :
  changeReceived 20 sandysOrder = 3 := by
  sorry

end sandys_change_l3874_387424


namespace sum_of_zeros_less_than_two_ln_a_l3874_387408

/-- Given a function f(x) = e^x - ax + a, where a ∈ ℝ, if f has two zeros, their sum is less than 2 ln a -/
theorem sum_of_zeros_less_than_two_ln_a (a : ℝ) (x₁ x₂ : ℝ) :
  let f := fun x => Real.exp x - a * x + a
  (f x₁ = 0) → (f x₂ = 0) → (x₁ + x₂ < 2 * Real.log a) := by
  sorry

end sum_of_zeros_less_than_two_ln_a_l3874_387408


namespace arithmetic_mean_sequence_l3874_387402

theorem arithmetic_mean_sequence (a b c d e f g : ℝ) 
  (hb : b = (a + c) / 2)
  (hc : c = (b + d) / 2)
  (hd : d = (c + e) / 2)
  (he : e = (d + f) / 2)
  (hf : f = (e + g) / 2) :
  d = (a + g) / 2 := by
sorry

end arithmetic_mean_sequence_l3874_387402


namespace smallest_n_congruence_l3874_387460

theorem smallest_n_congruence (n : ℕ+) : (23 * n.val ≡ 5678 [ZMOD 11]) ↔ n = 2 := by
  sorry

end smallest_n_congruence_l3874_387460


namespace max_dominoes_8x9_board_l3874_387419

/-- Represents a checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a domino -/
structure Domino :=
  (length : ℕ)
  (width : ℕ)

/-- Calculates the maximum number of non-overlapping dominoes on a checkerboard -/
def max_dominoes (board : Checkerboard) (domino : Domino) (initial_placement : ℕ) : ℕ :=
  sorry

theorem max_dominoes_8x9_board :
  let board : Checkerboard := ⟨8, 9⟩
  let domino : Domino := ⟨2, 1⟩
  let initial_placement : ℕ := 6
  max_dominoes board domino initial_placement = 34 :=
by sorry

end max_dominoes_8x9_board_l3874_387419


namespace simplify_expression_l3874_387437

theorem simplify_expression : 8 * (15 / 11) * (-25 / 40) = -15 / 11 := by
  sorry

end simplify_expression_l3874_387437


namespace original_average_theorem_l3874_387490

theorem original_average_theorem (S : Finset ℝ) (f : ℝ → ℝ) :
  S.card = 7 →
  (∀ x ∈ S, f x = 5 * x) →
  (S.sum f) / S.card = 75 →
  (S.sum id) / S.card = 15 :=
by
  sorry

end original_average_theorem_l3874_387490


namespace walk_time_calculation_l3874_387496

/-- Represents the walking times between different locations in minutes -/
structure WalkingTimes where
  parkOfficeToHiddenLake : ℝ
  hiddenLakeToParkOffice : ℝ
  parkOfficeToLakeParkRestaurant : ℝ

/-- Represents the wind effect on walking times -/
structure WindEffect where
  favorableReduction : ℝ
  adverseIncrease : ℝ

theorem walk_time_calculation (w : WindEffect) (t : WalkingTimes) : 
  w.favorableReduction = 0.2 →
  w.adverseIncrease = 0.25 →
  t.parkOfficeToHiddenLake * (1 - w.favorableReduction) = 15 →
  t.hiddenLakeToParkOffice * (1 + w.adverseIncrease) = 7 →
  t.parkOfficeToHiddenLake * (1 - w.favorableReduction) + 
    t.hiddenLakeToParkOffice * (1 + w.adverseIncrease) + 
    t.parkOfficeToLakeParkRestaurant * (1 - w.favorableReduction) = 32 →
  t.parkOfficeToLakeParkRestaurant = 12.5 := by
  sorry

#check walk_time_calculation

end walk_time_calculation_l3874_387496


namespace triangles_5_4_l3874_387455

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of triangles formed by points on two parallel lines -/
def triangles_on_parallel_lines (points_on_line_a points_on_line_b : ℕ) : ℕ :=
  choose points_on_line_a 2 * choose points_on_line_b 1 +
  choose points_on_line_a 1 * choose points_on_line_b 2

/-- Theorem: The number of triangles formed by 5 points on one line and 4 points on a parallel line -/
theorem triangles_5_4 : triangles_on_parallel_lines 5 4 = choose 5 2 * choose 4 1 + choose 5 1 * choose 4 2 := by
  sorry

end triangles_5_4_l3874_387455


namespace basketball_average_points_l3874_387476

/-- Given a basketball player who scored 60 points in 5 games, 
    prove that their average points per game is 12. -/
theorem basketball_average_points (total_points : ℕ) (num_games : ℕ) 
  (h1 : total_points = 60) (h2 : num_games = 5) : 
  total_points / num_games = 12 := by
  sorry

end basketball_average_points_l3874_387476


namespace staircase_perimeter_l3874_387498

/-- Represents a staircase-shaped region -/
structure StaircaseRegion where
  -- Eight sides of length 1
  unit_sides : Fin 8 → ℝ
  unit_sides_length : ∀ i, unit_sides i = 1
  -- Area of the region
  area : ℝ
  area_value : area = 53
  -- Other properties of the staircase shape are implicit

/-- The perimeter of a staircase region -/
def perimeter (s : StaircaseRegion) : ℝ := sorry

/-- Theorem stating that the perimeter of the given staircase region is 32 -/
theorem staircase_perimeter (s : StaircaseRegion) : perimeter s = 32 := by
  sorry

end staircase_perimeter_l3874_387498


namespace rational_solutions_are_integer_l3874_387409

theorem rational_solutions_are_integer (a b : ℤ) :
  ∃ (x y : ℚ), y - 2*x = a ∧ y^2 - x*y + x^2 = b →
  ∃ (x' y' : ℤ), (x' : ℚ) = x ∧ (y' : ℚ) = y := by
sorry

end rational_solutions_are_integer_l3874_387409


namespace lattice_triangle_properties_l3874_387438

/-- A lattice point in the xy-plane -/
structure LatticePoint where
  x : Int
  y : Int

/-- A triangle with vertices at lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Count of lattice points on a side (excluding endpoints) -/
def latticePointsOnSide (P Q : LatticePoint) : Nat :=
  sorry

/-- Area of a triangle with lattice point vertices -/
def triangleArea (t : LatticeTriangle) : Int :=
  sorry

theorem lattice_triangle_properties (t : LatticeTriangle) :
  (latticePointsOnSide t.A t.B % 2 = 1 ∧ latticePointsOnSide t.A t.C % 2 = 1 →
    latticePointsOnSide t.B t.C % 2 = 1) ∧
  (latticePointsOnSide t.A t.B = 3 ∧ latticePointsOnSide t.A t.C = 3 →
    ∃ k : Int, triangleArea t = 8 * k) :=
  sorry

end lattice_triangle_properties_l3874_387438


namespace perpendicular_condition_false_l3874_387483

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpLine : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpPlane : Plane → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (subset : Line → Plane → Prop)

theorem perpendicular_condition_false
  (α β : Plane) (b : Line)
  (h_diff : α ≠ β)
  (h_subset : subset b β) :
  ¬(∀ (α β : Plane) (b : Line),
    α ≠ β →
    subset b β →
    (perpLine b α → perpPlane α β) ∧
    ¬(perpPlane α β → perpLine b α)) :=
sorry

end perpendicular_condition_false_l3874_387483


namespace samanthas_number_l3874_387482

theorem samanthas_number (x : ℚ) : 5 * ((3 * x + 6) / 2) = 100 → x = 34 / 3 := by
  sorry

end samanthas_number_l3874_387482


namespace bag_weight_problem_l3874_387428

theorem bag_weight_problem (w1 w2 w3 : ℝ) : 
  w1 / w2 = 4 / 5 ∧ w2 / w3 = 5 / 6 ∧ w1 + w3 = w2 + 45 → w1 = 36 := by
  sorry

end bag_weight_problem_l3874_387428


namespace equal_cheese_division_l3874_387461

/-- Represents an equilateral triangle cheese -/
structure EquilateralTriangleCheese where
  side_length : ℝ
  area : ℝ

/-- Represents a division of the cheese -/
structure CheeseDivision where
  num_pieces : ℕ
  piece_area : ℝ

/-- The number of people to divide the cheese among -/
def num_people : ℕ := 5

theorem equal_cheese_division 
  (cheese : EquilateralTriangleCheese) 
  (division : CheeseDivision) :
  division.num_pieces = 25 ∧
  division.piece_area * division.num_pieces = cheese.area ∧
  division.num_pieces % num_people = 0 →
  ∃ (pieces_per_person : ℕ), 
    pieces_per_person * num_people = division.num_pieces ∧
    pieces_per_person = 5 := by
  sorry

end equal_cheese_division_l3874_387461


namespace polynomial_equality_l3874_387415

theorem polynomial_equality (m n : ℤ) : 
  (∀ x : ℤ, (x - 4) * (x + 8) = x^2 + m*x + n) → 
  (m = 4 ∧ n = -32) := by
sorry

end polynomial_equality_l3874_387415


namespace correct_assignment_count_l3874_387469

/-- The number of ways to assign doctors and nurses to schools -/
def assignment_methods (doctors nurses schools : ℕ) : ℕ :=
  if doctors = 2 ∧ nurses = 4 ∧ schools = 2 then 12 else 0

/-- Theorem stating that there are 12 different assignment methods -/
theorem correct_assignment_count :
  assignment_methods 2 4 2 = 12 := by
  sorry

end correct_assignment_count_l3874_387469


namespace sufficient_condition_for_product_greater_than_four_l3874_387474

theorem sufficient_condition_for_product_greater_than_four (a b : ℝ) :
  a > 2 → b > 2 → a * b > 4 := by sorry

end sufficient_condition_for_product_greater_than_four_l3874_387474


namespace coffee_consumption_l3874_387477

def vacation_duration : ℕ := 40
def pods_per_box : ℕ := 30
def cost_per_box : ℚ := 8
def total_spent : ℚ := 32

def cups_per_day : ℚ := total_spent / cost_per_box * pods_per_box / vacation_duration

theorem coffee_consumption : cups_per_day = 3 := by sorry

end coffee_consumption_l3874_387477


namespace symmetry_of_exponential_graphs_l3874_387400

theorem symmetry_of_exponential_graphs :
  ∀ (a b : ℝ), b = 3^a ↔ -b = -(3^(-a)) := by sorry

end symmetry_of_exponential_graphs_l3874_387400


namespace rectangular_solid_surface_area_l3874_387407

/-- Surface area of a rectangular solid -/
def surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: Total surface area of a rectangular solid with given dimensions -/
theorem rectangular_solid_surface_area (a : ℝ) :
  surface_area a (a + 2) (a - 1) = 6 * a^2 + 4 * a - 4 := by
  sorry

end rectangular_solid_surface_area_l3874_387407


namespace flour_calculation_l3874_387417

/-- The amount of flour Katie needs in pounds -/
def katie_flour : ℕ := 3

/-- The additional amount of flour Sheila needs compared to Katie in pounds -/
def sheila_extra_flour : ℕ := 2

/-- The total amount of flour needed by both Katie and Sheila -/
def total_flour : ℕ := katie_flour + (katie_flour + sheila_extra_flour)

theorem flour_calculation :
  total_flour = 8 :=
sorry

end flour_calculation_l3874_387417


namespace total_students_l3874_387440

theorem total_students (A B C : ℕ) : 
  B = A - 8 →
  C = 5 * B →
  B = 25 →
  A + B + C = 183 := by
sorry

end total_students_l3874_387440


namespace quadratic_radical_combination_l3874_387464

theorem quadratic_radical_combination (a : ℝ) : 
  (∃ k : ℝ, (k * Real.sqrt 2)^2 = a + 1) → a = 1 := by
sorry

end quadratic_radical_combination_l3874_387464


namespace consecutive_numbers_proof_l3874_387472

theorem consecutive_numbers_proof (x y z : ℤ) : 
  (x = y + 1) →  -- x, y are consecutive
  (y = z + 1) →  -- y, z are consecutive
  (x > y) →      -- x > y
  (y > z) →      -- y > z
  (2*x + 3*y + 3*z = 5*y + 11) →  -- given equation
  (z = 3) →      -- given value of z
  (3*y = 12) :=  -- conclusion to prove
by
  sorry

end consecutive_numbers_proof_l3874_387472


namespace ping_pong_rackets_sold_l3874_387429

theorem ping_pong_rackets_sold (total_amount : ℝ) (average_price : ℝ) (h1 : total_amount = 490) (h2 : average_price = 9.8) :
  total_amount / average_price = 50 := by
  sorry

end ping_pong_rackets_sold_l3874_387429


namespace fermat_like_equation_implies_power_l3874_387492

theorem fermat_like_equation_implies_power (n p x y k : ℕ) : 
  Odd n → 
  n > 1 → 
  Nat.Prime p → 
  Odd p → 
  x^n + y^n = p^k → 
  ∃ t : ℕ, n = p^t := by
sorry

end fermat_like_equation_implies_power_l3874_387492


namespace quadratic_function_property_l3874_387405

theorem quadratic_function_property (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = -1/3 * x₁^2 + 5 →
  y₂ = -1/3 * x₂^2 + 5 →
  0 < x₁ →
  x₁ < x₂ →
  y₂ < y₁ ∧ y₁ < 5 :=
by sorry

end quadratic_function_property_l3874_387405


namespace franks_boxes_l3874_387441

/-- The number of boxes Frank filled with toys -/
def filled_boxes : ℕ := 8

/-- The number of boxes Frank has left empty -/
def empty_boxes : ℕ := 5

/-- The total number of boxes Frank had initially -/
def total_boxes : ℕ := filled_boxes + empty_boxes

theorem franks_boxes : total_boxes = 13 := by
  sorry

end franks_boxes_l3874_387441


namespace rectangular_field_perimeter_l3874_387416

theorem rectangular_field_perimeter : ∀ (length breadth : ℝ),
  breadth = 0.6 * length →
  length * breadth = 37500 →
  2 * (length + breadth) = 800 := by
  sorry

end rectangular_field_perimeter_l3874_387416


namespace digit_placement_theorem_l3874_387471

def number_of_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  Nat.choose n k * Nat.factorial m

theorem digit_placement_theorem :
  number_of_arrangements 6 2 4 = 360 :=
sorry

end digit_placement_theorem_l3874_387471


namespace probability_a2_selected_l3874_387410

-- Define the sets of students
def english_students : Finset (Fin 2) := Finset.univ
def japanese_students : Finset (Fin 3) := Finset.univ

-- Define the total number of possible outcomes
def total_outcomes : ℕ := (english_students.card * japanese_students.card)

-- Define the number of outcomes where A₂ is selected
def a2_outcomes : ℕ := japanese_students.card

-- Theorem statement
theorem probability_a2_selected :
  (a2_outcomes : ℚ) / total_outcomes = 1 / 2 := by
  sorry


end probability_a2_selected_l3874_387410


namespace parallel_lines_imply_a_equals_one_l3874_387450

-- Define the direction vectors
def v1 (a : ℝ) : Fin 3 → ℝ := ![2*a, 3, 2]
def v2 : Fin 3 → ℝ := ![2, 3, 2]

-- Define the condition for parallel lines
def are_parallel (a : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ i, v1 a i = k * v2 i

-- Theorem statement
theorem parallel_lines_imply_a_equals_one :
  are_parallel 1 → ∀ a : ℝ, are_parallel a → a = 1 := by
  sorry

end parallel_lines_imply_a_equals_one_l3874_387450


namespace inequality_proof_l3874_387489

theorem inequality_proof (a b c : ℝ) (h1 : a > -b) (h2 : -b > 0) (h3 : c < 0) :
  a * (1 - c) > b * (c - 1) := by
  sorry

end inequality_proof_l3874_387489


namespace fashion_show_duration_l3874_387466

/-- The total time for a fashion show runway -/
def fashion_show_time (num_models : ℕ) (bathing_suits_per_model : ℕ) (evening_wear_per_model : ℕ) (time_per_trip : ℕ) : ℕ :=
  (num_models * (bathing_suits_per_model + evening_wear_per_model)) * time_per_trip

/-- Theorem: The fashion show with 6 models, 2 bathing suits and 3 evening wear per model, and 2 minutes per trip takes 60 minutes -/
theorem fashion_show_duration :
  fashion_show_time 6 2 3 2 = 60 := by
  sorry

end fashion_show_duration_l3874_387466


namespace clock_ticks_theorem_l3874_387470

/-- Represents the number of ticks and time between first and last ticks for a clock -/
structure ClockTicks where
  num_ticks : ℕ
  time_between : ℕ

/-- Calculates the number of ticks given the time between first and last ticks -/
def calculate_ticks (reference : ClockTicks) (time : ℕ) : ℕ :=
  let interval := reference.time_between / (reference.num_ticks - 1)
  (time / interval) + 1

theorem clock_ticks_theorem (reference : ClockTicks) (time : ℕ) :
  reference.num_ticks = 8 ∧ reference.time_between = 42 ∧ time = 30 →
  calculate_ticks reference time = 6 :=
by
  sorry

#check clock_ticks_theorem

end clock_ticks_theorem_l3874_387470


namespace contrapositive_example_l3874_387439

theorem contrapositive_example : 
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔ 
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) :=
by sorry

end contrapositive_example_l3874_387439


namespace tangent_line_to_ln_curve_l3874_387425

theorem tangent_line_to_ln_curve (k : ℝ) :
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = 1 / x) → k = 1 / Real.exp 1 := by
  sorry

end tangent_line_to_ln_curve_l3874_387425


namespace star_equality_l3874_387426

/-- Binary operation ★ on ordered pairs of integers -/
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

/-- Theorem stating that if (5,4) ★ (1,1) = (x,y) ★ (4,3), then x = 8 -/
theorem star_equality (x y : ℤ) :
  star 5 4 1 1 = star x y 4 3 → x = 8 := by
  sorry

end star_equality_l3874_387426


namespace common_difference_is_half_l3874_387404

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition : a 1 + a 6 + a 11 = 6
  fourth_term : a 4 = 1

/-- The common difference of an arithmetic sequence is 1/2 given the conditions -/
theorem common_difference_is_half (seq : ArithmeticSequence) : 
  ∃ d : ℚ, (∀ n : ℕ, seq.a (n + 1) - seq.a n = d) ∧ d = 1/2 := by
  sorry

end common_difference_is_half_l3874_387404


namespace gcd_power_two_minus_one_l3874_387442

theorem gcd_power_two_minus_one : Nat.gcd (2^1025 - 1) (2^1056 - 1) = 2^31 - 1 := by
  sorry

end gcd_power_two_minus_one_l3874_387442


namespace cos_135_degrees_l3874_387432

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_135_degrees_l3874_387432


namespace juan_milk_needed_l3874_387412

/-- The number of cookies that can be baked with one half-gallon of milk -/
def cookies_per_half_gallon : ℕ := 48

/-- The number of cookies Juan wants to bake -/
def cookies_to_bake : ℕ := 40

/-- The amount of milk needed for baking, in half-gallons -/
def milk_needed : ℕ := 1

theorem juan_milk_needed :
  cookies_to_bake ≤ cookies_per_half_gallon → milk_needed = 1 := by
  sorry

end juan_milk_needed_l3874_387412


namespace cannot_end_with_two_l3874_387401

-- Define the initial set of numbers
def initial_numbers : List Nat := List.range 2017

-- Define the operation of taking the difference
def difference_operation (a b : Nat) : Nat := Int.natAbs (a - b)

-- Define the property of maintaining odd sum parity
def maintains_odd_sum_parity (numbers : List Nat) : Prop :=
  List.sum numbers % 2 = 1

-- Define the final state we want to disprove
def final_state (numbers : List Nat) : Prop :=
  numbers = [2]

-- Theorem statement
theorem cannot_end_with_two :
  ¬ ∃ (final_numbers : List Nat),
    (maintains_odd_sum_parity initial_numbers →
     maintains_odd_sum_parity final_numbers) ∧
    final_state final_numbers :=
by sorry

end cannot_end_with_two_l3874_387401


namespace digit_sum_problem_l3874_387414

theorem digit_sum_problem (a b c d : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) →  -- digits are less than 10
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →  -- digits are different
  (c + a = 10) →  -- condition from right column
  (b + c + 1 = 10) →  -- condition from middle column
  (a + d + 1 = 11) →  -- condition from left column
  (a + b + c + d = 19) :=
by sorry

end digit_sum_problem_l3874_387414


namespace log_sum_equals_two_l3874_387435

theorem log_sum_equals_two :
  2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end log_sum_equals_two_l3874_387435


namespace third_degree_polynomial_property_l3874_387445

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- The property that |g(x)| = 15 for x ∈ {0, 1, 2, 4, 5, 6} -/
def HasSpecificValues (g : ThirdDegreePolynomial) : Prop :=
  ∀ x ∈ ({0, 1, 2, 4, 5, 6} : Set ℝ), |g x| = 15

theorem third_degree_polynomial_property (g : ThirdDegreePolynomial) 
  (h : HasSpecificValues g) : |g (-1)| = 75 := by
  sorry

end third_degree_polynomial_property_l3874_387445


namespace inequality_solution_l3874_387420

theorem inequality_solution (x : ℝ) : 
  (|(7 - 2*x) / 4| < 3) ↔ (-5/2 < x ∧ x < 19/2) :=
by sorry

end inequality_solution_l3874_387420


namespace point_n_from_m_l3874_387421

/-- Given two points M and N in a 2D plane, prove that N can be obtained
    from M by moving 4 units upward. -/
theorem point_n_from_m (M N : ℝ × ℝ) : 
  M = (-1, -1) → N = (-1, 3) → N.2 - M.2 = 4 := by
  sorry

end point_n_from_m_l3874_387421


namespace unique_recovery_l3874_387499

theorem unique_recovery (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_operations : ∃ (x y : ℝ) (x_pos : x > 0) (y_pos : y > 0),
    ({a, b, c, d} : Set ℝ) = {x + y, x - y, x / y, x * y}) :
  ∃! (x y : ℝ), x > 0 ∧ y > 0 ∧ ({a, b, c, d} : Set ℝ) = {x + y, x - y, x / y, x * y} :=
sorry

end unique_recovery_l3874_387499


namespace jack_email_difference_l3874_387468

/-- Given the number of emails Jack received at different times of the day,
    prove that he received 2 more emails in the morning than in the afternoon. -/
theorem jack_email_difference (morning afternoon evening : ℕ) 
    (h1 : morning = 5)
    (h2 : afternoon = 3)
    (h3 : evening = 16) :
    morning - afternoon = 2 := by
  sorry

end jack_email_difference_l3874_387468


namespace square_sum_and_reciprocal_square_l3874_387436

theorem square_sum_and_reciprocal_square (x : ℝ) (h : x + 2/x = 6) :
  x^2 + 4/x^2 = 32 := by
  sorry

end square_sum_and_reciprocal_square_l3874_387436


namespace solve_for_y_l3874_387434

theorem solve_for_y (x y : ℝ) (h1 : x^2 = y + 7) (h2 : x = 6) : y = 29 := by
  sorry

end solve_for_y_l3874_387434


namespace directory_page_numbering_l3874_387497

/-- Calculate the total number of digits needed to number pages in a directory --/
def totalDigits (totalPages : ℕ) : ℕ :=
  let singleDigitPages := min totalPages 9
  let doubleDigitPages := min (max (totalPages - 9) 0) 90
  let tripleDigitPages := max (totalPages - 99) 0
  singleDigitPages * 1 + doubleDigitPages * 2 + tripleDigitPages * 3

/-- Theorem: A directory with 710 pages requires 2022 digits to number all pages --/
theorem directory_page_numbering :
  totalDigits 710 = 2022 := by sorry

end directory_page_numbering_l3874_387497


namespace test_score_calculation_l3874_387457

theorem test_score_calculation (total_questions correct_answers incorrect_answers score : ℕ) : 
  total_questions = 100 →
  correct_answers + incorrect_answers = total_questions →
  score = correct_answers - 2 * incorrect_answers →
  score = 70 →
  correct_answers = 90 := by
sorry

end test_score_calculation_l3874_387457


namespace water_speed_calculation_l3874_387406

/-- 
Given a person who can swim at 4 km/h in still water and takes 7 hours to swim 14 km against a current,
prove that the speed of the water is 2 km/h.
-/
theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) :
  still_water_speed = 4 →
  distance = 14 →
  time = 7 →
  ∃ (water_speed : ℝ), water_speed = 2 ∧ still_water_speed - water_speed = distance / time :=
by sorry

end water_speed_calculation_l3874_387406


namespace fraction_problem_l3874_387488

theorem fraction_problem (f : ℚ) : f * 300 = (3/5 * 125) + 45 → f = 2/5 := by
  sorry

end fraction_problem_l3874_387488


namespace max_y_value_l3874_387467

/-- Given that x and y are negative integers satisfying y = 10x / (10 - x), 
    the maximum value of y is -5 -/
theorem max_y_value (x y : ℤ) 
  (h1 : x < 0) 
  (h2 : y < 0) 
  (h3 : y = 10 * x / (10 - x)) : 
  (∀ z : ℤ, z < 0 ∧ ∃ w : ℤ, w < 0 ∧ z = 10 * w / (10 - w) → z ≤ -5) ∧ 
  (∃ u : ℤ, u < 0 ∧ -5 = 10 * u / (10 - u)) :=
sorry

end max_y_value_l3874_387467


namespace factorization_equality_l3874_387452

theorem factorization_equality (x y a b : ℝ) :
  9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a + 2 * b) * (3 * a - 2 * b) :=
by sorry

end factorization_equality_l3874_387452


namespace quadratic_factorization_sum_l3874_387418

/-- 
Given a quadratic expression 4x^2 - 8x + 6, when factorized in the form a(x - h)^2 + k,
the sum of a, h, and k is equal to 7.
-/
theorem quadratic_factorization_sum (x : ℝ) :
  ∃ (a h k : ℝ), 
    (4 * x^2 - 8 * x + 6 = a * (x - h)^2 + k) ∧ 
    (a + h + k = 7) := by
  sorry

end quadratic_factorization_sum_l3874_387418


namespace function_inequalities_l3874_387479

theorem function_inequalities (p q r s : ℝ) (h : p * s - q * r < 0) :
  let f := fun x => (p * x + q) / (r * x + s)
  ∀ x₁ x₂ ε : ℝ,
    ε > 0 →
    (x₁ < x₂ ∧ x₂ < -s/r → f x₁ > f x₂) ∧
    (-s/r < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
    (x₁ < x₂ ∧ x₂ < -s/r → f (x₁ - ε) - f x₁ < f (x₂ - ε) - f x₂) ∧
    (-s/r < x₁ ∧ x₁ < x₂ → f x₁ - f (x₁ + ε) > f x₂ - f (x₂ + ε)) :=
by sorry

end function_inequalities_l3874_387479


namespace ab_value_l3874_387463

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧
    Real.sqrt (log a) = m ∧
    Real.sqrt (log b) = n ∧
    log (Real.sqrt a) = m^2 / 2 ∧
    log (Real.sqrt b) = n^2 / 2 ∧
    m + n + m^2 / 2 + n^2 / 2 = 100) →
  a * b = 10^164 := by
sorry

end ab_value_l3874_387463


namespace inequality_holds_iff_p_in_interval_l3874_387494

theorem inequality_holds_iff_p_in_interval (p : ℝ) :
  (∀ x : ℝ, -9 < (3 * x^2 + p * x - 6) / (x^2 - x + 1) ∧ 
             (3 * x^2 + p * x - 6) / (x^2 - x + 1) < 6) ↔ 
  -3 < p ∧ p < 6 := by sorry

end inequality_holds_iff_p_in_interval_l3874_387494

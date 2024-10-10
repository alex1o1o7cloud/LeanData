import Mathlib

namespace quadratic_inequality_solution_l1805_180523

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 - 5 * x + 2 > 0) ↔ (x < 2/3 ∨ x > 1) :=
by sorry

end quadratic_inequality_solution_l1805_180523


namespace complement_intersection_theorem_l1805_180531

def A : Set ℤ := {x | x^2 ≤ 16}
def B : Set ℤ := {x | -1 ≤ x ∧ x < 4}

theorem complement_intersection_theorem : 
  (A \ (A ∩ B)) = {-4, -3, -2, 4} := by sorry

end complement_intersection_theorem_l1805_180531


namespace no_infinite_prime_sequence_with_property_l1805_180578

-- Define the property for the sequence
def isPrimeSequenceWithProperty (p : ℕ → ℕ) : Prop :=
  (∀ n, Nat.Prime (p n)) ∧ 
  (∀ n, (p (n + 1) : ℤ) - 2 * (p n : ℤ) = 1 ∨ (p (n + 1) : ℤ) - 2 * (p n : ℤ) = -1)

-- State the theorem
theorem no_infinite_prime_sequence_with_property :
  ¬ ∃ p : ℕ → ℕ, isPrimeSequenceWithProperty p :=
sorry

end no_infinite_prime_sequence_with_property_l1805_180578


namespace num_dogs_in_pool_l1805_180569

-- Define the total number of legs/paws in the pool
def total_legs : ℕ := 24

-- Define the number of humans in the pool
def num_humans : ℕ := 2

-- Define the number of legs per human
def legs_per_human : ℕ := 2

-- Define the number of legs per dog
def legs_per_dog : ℕ := 4

-- Theorem to prove
theorem num_dogs_in_pool : 
  (total_legs - num_humans * legs_per_human) / legs_per_dog = 5 := by
  sorry


end num_dogs_in_pool_l1805_180569


namespace monotonic_decreasing_interval_of_f_l1805_180500

def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

theorem monotonic_decreasing_interval_of_f :
  ∀ a b : ℝ, a = -1 ∧ b = 11 →
  (∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f y < f x) ∧
  ¬(∃ c d : ℝ, (c < a ∨ b < d) ∧
    (∀ x y : ℝ, c < x ∧ x < y ∧ y < d → f y < f x)) :=
by sorry

end monotonic_decreasing_interval_of_f_l1805_180500


namespace steve_juice_consumption_l1805_180516

theorem steve_juice_consumption (don_juice : ℚ) (steve_fraction : ℚ) :
  don_juice = 1/4 →
  steve_fraction = 3/4 →
  steve_fraction * don_juice = 3/16 := by
sorry

end steve_juice_consumption_l1805_180516


namespace vector_magnitude_l1805_180587

theorem vector_magnitude (a b : ℝ × ℝ) :
  ‖a‖ = 1 →
  ‖b‖ = 2 →
  a - b = (Real.sqrt 3, Real.sqrt 2) →
  ‖a + 2 • b‖ = Real.sqrt 17 := by
  sorry

end vector_magnitude_l1805_180587


namespace perpendicular_planes_counterexample_l1805_180585

/-- A type representing a plane in 3D space -/
structure Plane :=
  (normal : ℝ × ℝ × ℝ)
  (point : ℝ × ℝ × ℝ)

/-- Perpendicularity relation between planes -/
def perpendicular (p q : Plane) : Prop :=
  ∃ (k : ℝ), p.normal = k • q.normal

theorem perpendicular_planes_counterexample :
  ∃ (α β γ : Plane),
    α ≠ β ∧ β ≠ γ ∧ α ≠ γ ∧
    perpendicular α β ∧
    perpendicular β γ ∧
    ¬ perpendicular α γ :=
sorry

end perpendicular_planes_counterexample_l1805_180585


namespace functional_equation_solution_l1805_180521

/-- Given a function g : ℝ → ℝ satisfying the functional equation
    2g(x) - 3g(1/x) = x^2 for all x ≠ 0, prove that g(2) = 8.25 -/
theorem functional_equation_solution (g : ℝ → ℝ) 
    (h : ∀ x : ℝ, x ≠ 0 → 2 * g x - 3 * g (1/x) = x^2) : 
  g 2 = 8.25 := by
  sorry

end functional_equation_solution_l1805_180521


namespace division_result_l1805_180510

theorem division_result : (0.075 : ℚ) / (0.005 : ℚ) = 15 := by
  sorry

end division_result_l1805_180510


namespace inequality_system_solution_l1805_180548

theorem inequality_system_solution (p : ℝ) : 19 * p < 10 ∧ p > (1/2 : ℝ) → (1/2 : ℝ) < p ∧ p < 10/19 := by
  sorry

end inequality_system_solution_l1805_180548


namespace max_value_theorem_l1805_180568

theorem max_value_theorem (p q r s : ℝ) (h : p^2 + q^2 + r^2 - s^2 + 4 = 0) :
  ∃ (M : ℝ), M = -2 * Real.sqrt 2 ∧ ∀ (p' q' r' s' : ℝ), 
    p'^2 + q'^2 + r'^2 - s'^2 + 4 = 0 → 
    3*p' + 2*q' + r' - 4*abs s' ≤ M :=
by sorry

end max_value_theorem_l1805_180568


namespace sum_of_four_consecutive_integers_divisible_by_two_l1805_180557

theorem sum_of_four_consecutive_integers_divisible_by_two :
  ∀ n : ℤ, ∃ k : ℤ, (n - 1) + n + (n + 1) + (n + 2) = 2 * k :=
by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l1805_180557


namespace bhavan_score_percentage_l1805_180598

theorem bhavan_score_percentage (max_score : ℝ) (amar_percent : ℝ) (chetan_percent : ℝ) (average_mark : ℝ) :
  max_score = 900 →
  amar_percent = 64 →
  chetan_percent = 44 →
  average_mark = 432 →
  ∃ bhavan_percent : ℝ,
    bhavan_percent = 36 ∧
    3 * average_mark = (amar_percent / 100 * max_score) + (bhavan_percent / 100 * max_score) + (chetan_percent / 100 * max_score) :=
by sorry

end bhavan_score_percentage_l1805_180598


namespace different_suit_card_combinations_l1805_180501

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards per suit in a standard deck -/
def cards_per_suit : ℕ := standard_deck_size / number_of_suits

/-- The number of cards to be chosen -/
def cards_to_choose : ℕ := 4

-- Theorem statement
theorem different_suit_card_combinations :
  (number_of_suits.choose cards_to_choose) * (cards_per_suit ^ cards_to_choose) = 28561 := by
  sorry

end different_suit_card_combinations_l1805_180501


namespace inscribed_polygon_division_l1805_180536

-- Define a polygon inscribed around a circle
structure InscribedPolygon where
  vertices : List (ℝ × ℝ)
  center : ℝ × ℝ
  radius : ℝ
  is_inscribed : ∀ v ∈ vertices, dist center v = radius

-- Define a line passing through a point
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Define the area of a polygon
def area (p : InscribedPolygon) : ℝ := sorry

-- Define the perimeter of a polygon
def perimeter (p : InscribedPolygon) : ℝ := sorry

-- Define the two parts of a polygon divided by a line
def divided_parts (p : InscribedPolygon) (l : Line) : (InscribedPolygon × InscribedPolygon) := sorry

theorem inscribed_polygon_division (p : InscribedPolygon) (l : Line) 
  (h : l.point = p.center) : 
  let (p1, p2) := divided_parts p l
  (area p1 = area p2) ∧ (perimeter p1 = perimeter p2) := by
  sorry

end inscribed_polygon_division_l1805_180536


namespace time_ratio_in_countries_l1805_180552

/- Given conditions -/
def total_trip_duration : ℕ := 10
def time_in_first_country : ℕ := 2

/- Theorem to prove -/
theorem time_ratio_in_countries :
  (total_trip_duration - time_in_first_country) / time_in_first_country = 4 := by
  sorry

end time_ratio_in_countries_l1805_180552


namespace simplified_fraction_l1805_180565

theorem simplified_fraction (a : ℤ) (ha : a > 0) :
  let expr := (a + 1) / a - a / (a + 1)
  let simplified := (2 * a + 1) / (a * (a + 1))
  expr = simplified ∧ (a = 2023 → 2 * a + 1 = 4047) := by
  sorry

#eval 2 * 2023 + 1

end simplified_fraction_l1805_180565


namespace circle1_properties_circle2_and_circle3_properties_l1805_180591

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 1)^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 13
def circle3 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 13

-- Define the line equations
def line1 (x y : ℝ) : Prop := x - 2*y - 2 = 0
def line2 (x y : ℝ) : Prop := 2*x + 3*y - 10 = 0

-- Theorem for the first circle
theorem circle1_properties :
  (∀ x y, circle1 x y → line1 x y) ∧
  circle1 0 4 ∧
  circle1 4 6 := by sorry

-- Theorem for the second and third circles
theorem circle2_and_circle3_properties :
  (∀ x y, (circle2 x y ∨ circle3 x y) → (x - 2)^2 + (y - 2)^2 = 13) ∧
  (∃ x y, (circle2 x y ∨ circle3 x y) ∧ line2 x y ∧ x = 2 ∧ y = 2) := by sorry

end circle1_properties_circle2_and_circle3_properties_l1805_180591


namespace inequality_proof_l1805_180540

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end inequality_proof_l1805_180540


namespace exactly_four_triples_l1805_180553

/-- The number of ordered triples (a, b, c) of positive integers satisfying the given LCM conditions -/
def count_triples : ℕ := 4

/-- Predicate to check if a triple (a, b, c) satisfies the LCM conditions -/
def satisfies_conditions (a b c : ℕ+) : Prop :=
  Nat.lcm a b = 90 ∧ Nat.lcm a c = 980 ∧ Nat.lcm b c = 630

/-- The main theorem stating that there are exactly 4 triples satisfying the conditions -/
theorem exactly_four_triples :
  (∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), s.card = count_triples ∧
    ∀ t, t ∈ s ↔ satisfies_conditions t.1 t.2.1 t.2.2) :=
sorry

end exactly_four_triples_l1805_180553


namespace euler_totient_even_bound_l1805_180590

theorem euler_totient_even_bound (n : ℕ) (h : Even n) (h_pos : n > 0) : 
  (Finset.filter (fun x => Nat.gcd n x = 1) (Finset.range n)).card ≤ n / 2 := by
  sorry

end euler_totient_even_bound_l1805_180590


namespace square_symmetry_count_l1805_180584

/-- Represents the symmetry operations on a square -/
inductive SquareSymmetry
| reflect : SquareSymmetry
| rotate : SquareSymmetry

/-- Represents a sequence of symmetry operations -/
def SymmetrySequence := List SquareSymmetry

/-- Checks if a sequence of symmetry operations results in the identity transformation -/
def is_identity (seq : SymmetrySequence) : Prop :=
  sorry

/-- Counts the number of valid symmetry sequences of a given length -/
def count_valid_sequences (n : Nat) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem square_symmetry_count :
  count_valid_sequences 2016 % 100000 = 20000 :=
sorry

end square_symmetry_count_l1805_180584


namespace intersection_M_complement_N_empty_l1805_180532

def M : Set ℝ := {x | |x - 1| < 1}
def N : Set ℝ := {x | x^2 - 2*x < 3}

theorem intersection_M_complement_N_empty :
  M ∩ (Set.univ \ N) = ∅ := by sorry

end intersection_M_complement_N_empty_l1805_180532


namespace two_numbers_product_sum_l1805_180539

theorem two_numbers_product_sum (n : Nat) : n = 45 →
  ∃ x y : Nat, x ∈ Finset.range (n + 1) ∧ 
             y ∈ Finset.range (n + 1) ∧ 
             x < y ∧
             (Finset.sum (Finset.range (n + 1)) id - x - y = x * y) ∧
             y - x = 9 := by
  sorry

end two_numbers_product_sum_l1805_180539


namespace mean_calculation_l1805_180575

theorem mean_calculation (x : ℝ) : 
  (28 + x + 70 + 88 + 104) / 5 = 67 →
  (50 + 62 + 97 + 124 + x) / 5 = 75.6 :=
by sorry

end mean_calculation_l1805_180575


namespace num_valid_assignments_is_72_l1805_180589

/-- Represents a valid assignment of doctors to positions -/
structure DoctorAssignment where
  assignments : Fin 5 → Fin 4
  all_positions_filled : ∀ p : Fin 4, ∃ d : Fin 5, assignments d = p
  first_two_different : assignments 0 ≠ assignments 1

/-- The number of valid doctor assignments -/
def num_valid_assignments : ℕ := sorry

/-- Theorem stating that the number of valid assignments is 72 -/
theorem num_valid_assignments_is_72 : num_valid_assignments = 72 := by sorry

end num_valid_assignments_is_72_l1805_180589


namespace total_investment_proof_l1805_180502

def bank_investment : ℝ := 6000
def bond_investment : ℝ := 6000
def bank_interest_rate : ℝ := 0.05
def bond_return_rate : ℝ := 0.09
def annual_income : ℝ := 660

theorem total_investment_proof :
  bank_investment + bond_investment = 12000 :=
by sorry

end total_investment_proof_l1805_180502


namespace zeros_in_decimal_representation_l1805_180586

theorem zeros_in_decimal_representation (n : ℕ) : 
  (∃ k : ℕ, (1 : ℚ) / (25^10 : ℚ) = (1 : ℚ) / (10^k : ℚ)) ∧ 
  (∀ m : ℕ, m < 20 → (1 : ℚ) / (25^10 : ℚ) < (1 : ℚ) / (10^m : ℚ)) ∧
  (1 : ℚ) / (25^10 : ℚ) ≥ (1 : ℚ) / (10^20 : ℚ) :=
by sorry

end zeros_in_decimal_representation_l1805_180586


namespace basketball_handshakes_l1805_180541

/-- The number of handshakes in a basketball game with specific conditions -/
theorem basketball_handshakes :
  let team_size : ℕ := 6
  let num_teams : ℕ := 2
  let num_referees : ℕ := 3
  let opposing_team_handshakes := team_size * team_size
  let same_team_handshakes := num_teams * (team_size * (team_size - 1) / 2)
  let player_referee_handshakes := (num_teams * team_size) * num_referees
  opposing_team_handshakes + same_team_handshakes + player_referee_handshakes = 102 :=
by sorry

end basketball_handshakes_l1805_180541


namespace equation_solutions_l1805_180549

theorem equation_solutions :
  (∃ x : ℝ, 7 * x + 2 * (3 * x - 3) = 20 ∧ x = 2) ∧
  (∃ x : ℝ, (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 3 ∧ x = 67 / 23) := by
  sorry

end equation_solutions_l1805_180549


namespace bakery_sugar_amount_l1805_180571

/-- Given the ratios of ingredients in a bakery storage room, prove the amount of sugar. -/
theorem bakery_sugar_amount 
  (sugar flour baking_soda : ℝ)
  (h1 : sugar / flour = 5 / 6)
  (h2 : flour / baking_soda = 10)
  (h3 : flour / (baking_soda + 60) = 8) :
  sugar = 2000 := by
  sorry

end bakery_sugar_amount_l1805_180571


namespace slope_product_theorem_l1805_180509

theorem slope_product_theorem (m n p : ℝ) : 
  m ≠ 0 ∧ n ≠ 0 ∧ p ≠ 0 →  -- none of the lines are horizontal
  (∃ θ₁ θ₂ θ₃ : ℝ, 
    θ₁ = 3 * θ₂ ∧  -- L₁ makes three times the angle with the horizontal as L₂
    θ₃ = θ₁ / 2 ∧  -- L₃ makes half the angle of L₁
    m = Real.tan θ₁ ∧ 
    n = Real.tan θ₂ ∧ 
    p = Real.tan θ₃) →
  m = 3 * n →  -- L₁ has 3 times the slope of L₂
  m = 5 * p →  -- L₁ has 5 times the slope of L₃
  m * n * p = Real.sqrt 3 / 15 := by
sorry

end slope_product_theorem_l1805_180509


namespace orphanage_donation_percentage_l1805_180562

theorem orphanage_donation_percentage (total_income : ℝ) 
  (children_percentage : ℝ) (num_children : ℕ) (wife_percentage : ℝ) 
  (remaining_amount : ℝ) :
  total_income = 1200000 →
  children_percentage = 0.2 →
  num_children = 3 →
  wife_percentage = 0.3 →
  remaining_amount = 60000 →
  let distributed_percentage := children_percentage * num_children + wife_percentage
  let distributed_amount := distributed_percentage * total_income
  let amount_before_donation := total_income - distributed_amount
  let donation_amount := amount_before_donation - remaining_amount
  donation_amount / amount_before_donation = 0.5 := by sorry

end orphanage_donation_percentage_l1805_180562


namespace tan_cos_sum_identity_l1805_180504

theorem tan_cos_sum_identity : 
  Real.tan (30 * π / 180) * Real.cos (60 * π / 180) + 
  Real.tan (45 * π / 180) * Real.cos (30 * π / 180) = 
  2 * Real.sqrt 3 / 3 := by
  sorry

end tan_cos_sum_identity_l1805_180504


namespace choose_four_from_nine_l1805_180529

theorem choose_four_from_nine (n : ℕ) (k : ℕ) : n = 9 ∧ k = 4 → Nat.choose n k = 126 := by
  sorry

end choose_four_from_nine_l1805_180529


namespace hexagon_division_l1805_180558

/- Define a hexagon -/
def Hexagon : Type := Unit

/- Define a legal point in the hexagon -/
inductive LegalPoint : Type
| vertex : LegalPoint
| intersection : LegalPoint → LegalPoint → LegalPoint

/- Define a legal triangle in the hexagon -/
structure LegalTriangle :=
(p1 p2 p3 : LegalPoint)

/- Define a division of the hexagon -/
def Division := List LegalTriangle

/- The main theorem to prove -/
theorem hexagon_division (n : Nat) (h : n ≥ 6) : 
  ∃ (d : Division), d.length = n := by sorry

end hexagon_division_l1805_180558


namespace largest_constant_inequality_l1805_180592

theorem largest_constant_inequality (x y : ℝ) :
  (∃ (C : ℝ), ∀ (x y : ℝ), x^2 + y^2 + 1 ≥ C * (x + y)) ∧
  (∀ (D : ℝ), (∀ (x y : ℝ), x^2 + y^2 + 1 ≥ D * (x + y)) → D ≤ Real.sqrt 2) :=
by sorry

end largest_constant_inequality_l1805_180592


namespace range_of_a_for_decreasing_f_l1805_180547

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 2

-- State the theorem
theorem range_of_a_for_decreasing_f :
  (∀ a : ℝ, (∀ x : ℝ, (∀ y : ℝ, x < y → f a x > f a y)) ↔ a ∈ Set.Iic (-3)) := by
  sorry

end range_of_a_for_decreasing_f_l1805_180547


namespace exist_six_lines_equal_angles_l1805_180579

/-- A line in 3D space represented by a point and a direction vector -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Angle between two lines in 3D space -/
def angle (l1 l2 : Line3D) : ℝ := sorry

/-- A set of 6 lines in 3D space -/
def SixLines : Type := Fin 6 → Line3D

/-- Predicate to check if all pairs of lines are non-parallel -/
def all_non_parallel (lines : SixLines) : Prop :=
  ∀ i j, i ≠ j → lines i ≠ lines j

/-- Predicate to check if all pairwise angles are equal -/
def all_angles_equal (lines : SixLines) : Prop :=
  ∀ i j k l, i ≠ j → k ≠ l → angle (lines i) (lines j) = angle (lines k) (lines l)

/-- Theorem stating the existence of 6 lines satisfying the conditions -/
theorem exist_six_lines_equal_angles : 
  ∃ (lines : SixLines), all_non_parallel lines ∧ all_angles_equal lines :=
sorry

end exist_six_lines_equal_angles_l1805_180579


namespace circles_externally_tangent_l1805_180527

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r1 r2 d : ℝ) : Prop := d = r1 + r2

/-- Given two circles with radii 2 and 3, and the distance between their centers is 5,
    prove that they are externally tangent -/
theorem circles_externally_tangent :
  let r1 : ℝ := 2
  let r2 : ℝ := 3
  let d : ℝ := 5
  externally_tangent r1 r2 d := by
sorry

end circles_externally_tangent_l1805_180527


namespace cubic_polynomial_negative_one_bound_l1805_180580

/-- A polynomial of degree 3 with three distinct positive roots -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  roots : Fin 3 → ℝ
  roots_positive : ∀ i, roots i > 0
  roots_distinct : ∀ i j, i ≠ j → roots i ≠ roots j
  is_root : ∀ i, (roots i)^3 + a*(roots i)^2 + b*(roots i) - 1 = 0

/-- The polynomial P(x) = x^3 + ax^2 + bx - 1 -/
def P (poly : CubicPolynomial) (x : ℝ) : ℝ :=
  x^3 + poly.a * x^2 + poly.b * x - 1

theorem cubic_polynomial_negative_one_bound (poly : CubicPolynomial) : P poly (-1) < -8 := by
  sorry

end cubic_polynomial_negative_one_bound_l1805_180580


namespace pedestrian_meets_cart_time_l1805_180511

/-- Represents a participant in the scenario -/
inductive Participant
| Pedestrian
| Cyclist
| Cart
| Car

/-- Represents a time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Represents an event involving two participants -/
structure Event where
  participant1 : Participant
  participant2 : Participant
  time : Time

/-- The scenario with all participants and events -/
structure Scenario where
  cyclist_overtakes_pedestrian : Event
  pedestrian_meets_car : Event
  cyclist_meets_cart : Event
  cyclist_meets_car : Event
  car_meets_cyclist : Event
  car_meets_pedestrian : Event
  car_overtakes_cart : Event

def is_valid_scenario (s : Scenario) : Prop :=
  s.cyclist_overtakes_pedestrian.time = Time.mk 10 0 ∧
  s.pedestrian_meets_car.time = Time.mk 11 0 ∧
  s.cyclist_meets_cart.time.hours - s.cyclist_overtakes_pedestrian.time.hours = 
    s.cyclist_meets_car.time.hours - s.cyclist_meets_cart.time.hours ∧
  s.cyclist_meets_cart.time.minutes - s.cyclist_overtakes_pedestrian.time.minutes = 
    s.cyclist_meets_car.time.minutes - s.cyclist_meets_cart.time.minutes ∧
  s.car_meets_pedestrian.time.hours - s.car_meets_cyclist.time.hours = 
    s.car_overtakes_cart.time.hours - s.car_meets_pedestrian.time.hours ∧
  s.car_meets_pedestrian.time.minutes - s.car_meets_cyclist.time.minutes = 
    s.car_overtakes_cart.time.minutes - s.car_meets_pedestrian.time.minutes

theorem pedestrian_meets_cart_time (s : Scenario) (h : is_valid_scenario s) :
  ∃ (t : Event), t.participant1 = Participant.Pedestrian ∧ 
                 t.participant2 = Participant.Cart ∧ 
                 t.time = Time.mk 10 40 :=
sorry

end pedestrian_meets_cart_time_l1805_180511


namespace five_variable_inequality_two_is_smallest_constant_l1805_180525

theorem five_variable_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) > 2 :=
by sorry

theorem two_is_smallest_constant :
  ∀ ε > 0, ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) + 
    Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) + 
    Real.sqrt (e / (a + b + c + d)) < 2 + ε :=
by sorry

end five_variable_inequality_two_is_smallest_constant_l1805_180525


namespace triangle_formation_check_l1805_180555

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem triangle_formation_check :
  ¬(can_form_triangle 3 3 6) ∧
  ¬(can_form_triangle 2 3 6) ∧
  can_form_triangle 5 8 12 ∧
  ¬(can_form_triangle 4 7 11) :=
sorry

end triangle_formation_check_l1805_180555


namespace min_x_prime_factorization_sum_l1805_180507

theorem min_x_prime_factorization_sum (x y p q : ℕ+) (e f : ℕ) : 
  (∀ x' y' : ℕ+, 13 * x'^7 = 19 * y'^17 → x ≤ x') →
  13 * x^7 = 19 * y^17 →
  x = p^e * q^f →
  p.val.Prime ∧ q.val.Prime →
  p + q + e + f = 44 := by
  sorry

end min_x_prime_factorization_sum_l1805_180507


namespace smallest_number_l1805_180514

theorem smallest_number (S : Set ℤ) : S = {-2, -1, 0, 1} → ∀ x ∈ S, -2 ≤ x :=
by
  sorry

end smallest_number_l1805_180514


namespace monday_walking_speed_l1805_180520

/-- Represents Jonathan's exercise routine for a week -/
structure ExerciseRoutine where
  monday_speed : ℝ
  wednesday_speed : ℝ
  friday_speed : ℝ
  distance_per_day : ℝ
  total_time : ℝ

/-- Theorem stating that Jonathan's Monday walking speed is 2 miles per hour -/
theorem monday_walking_speed (routine : ExerciseRoutine) 
  (h1 : routine.wednesday_speed = 3)
  (h2 : routine.friday_speed = 6)
  (h3 : routine.distance_per_day = 6)
  (h4 : routine.total_time = 6)
  (h5 : routine.distance_per_day / routine.monday_speed + 
        routine.distance_per_day / routine.wednesday_speed + 
        routine.distance_per_day / routine.friday_speed = routine.total_time) :
  routine.monday_speed = 2 := by
  sorry

#check monday_walking_speed

end monday_walking_speed_l1805_180520


namespace octal_376_equals_decimal_254_l1805_180577

def octal_to_decimal (octal : ℕ) : ℕ :=
  (octal / 100) * 8^2 + ((octal / 10) % 10) * 8^1 + (octal % 10) * 8^0

theorem octal_376_equals_decimal_254 : octal_to_decimal 376 = 254 := by
  sorry

end octal_376_equals_decimal_254_l1805_180577


namespace carnival_tickets_l1805_180544

theorem carnival_tickets (tickets : ℕ) (extra : ℕ) : 
  let F := Nat.minFac (tickets + extra)
  F ∣ (tickets + extra) ∧ ¬(F ∣ tickets) →
  F = 3 :=
by
  sorry

#check carnival_tickets 865 8

end carnival_tickets_l1805_180544


namespace cut_triangular_prism_has_27_edges_l1805_180582

/-- Represents a triangular prism with corners cut off -/
structure CutTriangularPrism where
  /-- The number of vertices in the original triangular prism -/
  original_vertices : Nat
  /-- The number of edges in the original triangular prism -/
  original_edges : Nat
  /-- The number of new edges created by each corner cut -/
  new_edges_per_cut : Nat
  /-- Assertion that the cuts remove each corner entirely -/
  corners_removed : Prop
  /-- Assertion that the cuts do not intersect elsewhere on the prism -/
  cuts_dont_intersect : Prop

/-- The number of edges in a triangular prism with corners cut off -/
def num_edges_after_cuts (prism : CutTriangularPrism) : Nat :=
  prism.original_edges + prism.original_vertices * prism.new_edges_per_cut

/-- Theorem stating that a triangular prism with corners cut off has 27 edges -/
theorem cut_triangular_prism_has_27_edges (prism : CutTriangularPrism)
  (h1 : prism.original_vertices = 6)
  (h2 : prism.original_edges = 9)
  (h3 : prism.new_edges_per_cut = 3)
  (h4 : prism.corners_removed)
  (h5 : prism.cuts_dont_intersect) :
  num_edges_after_cuts prism = 27 := by
  sorry


end cut_triangular_prism_has_27_edges_l1805_180582


namespace x_interval_l1805_180517

theorem x_interval (x : ℝ) (h1 : 1/x < 3) (h2 : 1/x > -4) (h3 : 2*x - 1 > 0) : x > 1/2 := by
  sorry

end x_interval_l1805_180517


namespace f_intersects_twice_l1805_180526

/-- An even function that is monotonically increasing for positive x and satisfies f(1) * f(2) < 0 -/
def f : ℝ → ℝ :=
  sorry

/-- f is an even function -/
axiom f_even : ∀ x, f (-x) = f x

/-- f is monotonically increasing for positive x -/
axiom f_increasing : ∀ x y, 0 < x → x < y → f x < f y

/-- f(1) * f(2) < 0 -/
axiom f_sign_change : f 1 * f 2 < 0

/-- The number of intersection points between f and the x-axis -/
def num_intersections : ℕ :=
  sorry

/-- Theorem: The number of intersection points between f and the x-axis is 2 -/
theorem f_intersects_twice : num_intersections = 2 :=
  sorry

end f_intersects_twice_l1805_180526


namespace mom_tshirt_count_l1805_180518

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom buys -/
def packages_bought : ℕ := 71

/-- The total number of t-shirts Mom will have -/
def total_shirts : ℕ := shirts_per_package * packages_bought

theorem mom_tshirt_count : total_shirts = 426 := by
  sorry

end mom_tshirt_count_l1805_180518


namespace belle_treat_cost_l1805_180588

/-- The cost of feeding Belle treats for a week -/
def weekly_cost : ℚ := 21

/-- The number of dog biscuits Belle eats daily -/
def daily_biscuits : ℕ := 4

/-- The number of rawhide bones Belle eats daily -/
def daily_bones : ℕ := 2

/-- The cost of each rawhide bone in dollars -/
def bone_cost : ℚ := 1

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The cost of each dog biscuit in dollars -/
def biscuit_cost : ℚ := 1/4

theorem belle_treat_cost : 
  weekly_cost = days_in_week * (daily_biscuits * biscuit_cost + daily_bones * bone_cost) :=
by sorry

end belle_treat_cost_l1805_180588


namespace sqrt_fraction_simplification_l1805_180572

theorem sqrt_fraction_simplification :
  (Real.sqrt (7^2 + 24^2)) / (Real.sqrt (49 + 16)) = (25 * Real.sqrt 65) / 65 := by
  sorry

end sqrt_fraction_simplification_l1805_180572


namespace monotonicity_intervals_max_value_on_interval_min_value_on_interval_l1805_180560

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval of interest
def interval : Set ℝ := Set.Icc (-3) 2

-- Statement for intervals of monotonicity
theorem monotonicity_intervals (x : ℝ) :
  (∀ y z, y < x → x < z → y < -1 → z < -1 → f y < f z) ∧
  (∀ y z, y < x → x < z → -1 < y → z < 1 → f y > f z) ∧
  (∀ y z, y < x → x < z → 1 < y → f y < f z) :=
sorry

-- Statement for maximum value on the interval
theorem max_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 2 :=
sorry

-- Statement for minimum value on the interval
theorem min_value_on_interval :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≥ f x ∧ f x = -18 :=
sorry

end monotonicity_intervals_max_value_on_interval_min_value_on_interval_l1805_180560


namespace chicken_problem_l1805_180545

theorem chicken_problem (total chickens_colten : ℕ) 
  (h_total : total = 383)
  (h_colten : chickens_colten = 37) : 
  ∃ (chickens_skylar chickens_quentin : ℕ),
    chickens_skylar = 3 * chickens_colten - 4 ∧
    chickens_quentin = 2 * chickens_skylar + 32 ∧
    chickens_quentin + chickens_skylar + chickens_colten = total :=
by
  sorry

#check chicken_problem

end chicken_problem_l1805_180545


namespace arithmetic_sequence_sum_l1805_180594

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a b c : ℕ → ℝ) :
  ArithmeticSequence a ∧ ArithmeticSequence b ∧ ArithmeticSequence c →
  a 1 + b 1 + c 1 = 0 →
  a 2 + b 2 + c 2 = 1 →
  a 2015 + b 2015 + c 2015 = 2014 := by
  sorry

end arithmetic_sequence_sum_l1805_180594


namespace runner_position_l1805_180503

theorem runner_position (track_circumference : ℝ) (distance_run : ℝ) : 
  track_circumference = 100 →
  distance_run = 10560 →
  ∃ (n : ℕ) (remainder : ℝ), 
    distance_run = n * track_circumference + remainder ∧
    75 < remainder ∧ remainder ≤ 100 :=
by sorry

end runner_position_l1805_180503


namespace rain_given_east_wind_l1805_180554

/-- Given that:
    1. The probability of an east wind in April is 8/30
    2. The probability of both an east wind and rain in April is 7/30
    Prove that the probability of rain in April given an east wind is 7/8 -/
theorem rain_given_east_wind (p_east : ℚ) (p_east_and_rain : ℚ) 
  (h1 : p_east = 8/30) (h2 : p_east_and_rain = 7/30) :
  p_east_and_rain / p_east = 7/8 := by
  sorry

end rain_given_east_wind_l1805_180554


namespace tracis_road_trip_l1805_180573

theorem tracis_road_trip (D : ℝ) : 
  (1/3 : ℝ) * D + (1/4 : ℝ) * (2/3 : ℝ) * D + 300 = D → D = 600 :=
by sorry

end tracis_road_trip_l1805_180573


namespace box_length_is_twelve_l1805_180595

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.height * d.width * d.length

/-- Theorem: If 40 building blocks with given dimensions can fit into a box with given height and width,
    then the length of the box is 12 inches -/
theorem box_length_is_twelve
  (box : Dimensions)
  (block : Dimensions)
  (h1 : box.height = 8)
  (h2 : box.width = 10)
  (h3 : block.height = 3)
  (h4 : block.width = 2)
  (h5 : block.length = 4)
  (h6 : volume box ≥ 40 * volume block) :
  box.length = 12 :=
sorry

end box_length_is_twelve_l1805_180595


namespace rain_probability_l1805_180542

theorem rain_probability (p : ℝ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 5) :
  1 - (1 - p)^n = 1023/1024 := by
  sorry

end rain_probability_l1805_180542


namespace trapezoid_area_increase_l1805_180508

/-- Represents a trapezoid with a given height -/
structure Trapezoid where
  height : ℝ

/-- Calculates the increase in area when both bases of a trapezoid are increased by a given amount -/
def area_increase (t : Trapezoid) (base_increase : ℝ) : ℝ :=
  t.height * base_increase

/-- Theorem: The area increase of a trapezoid with height 6 cm when both bases are increased by 4 cm is 24 square centimeters -/
theorem trapezoid_area_increase :
  let t : Trapezoid := { height := 6 }
  area_increase t 4 = 24 := by
  sorry

end trapezoid_area_increase_l1805_180508


namespace smallest_value_l1805_180599

theorem smallest_value (y : ℝ) (h : y = 8) :
  let a := 5 / (y - 1)
  let b := 5 / (y + 1)
  let c := 5 / y
  let d := (5 + y) / 10
  let e := y - 5
  b < a ∧ b < c ∧ b < d ∧ b < e :=
by sorry

end smallest_value_l1805_180599


namespace veranda_area_l1805_180583

/-- The area of a veranda surrounding a rectangular room. -/
theorem veranda_area (room_length room_width veranda_length_side veranda_width_side : ℝ)
  (h1 : room_length = 19)
  (h2 : room_width = 12)
  (h3 : veranda_length_side = 2.5)
  (h4 : veranda_width_side = 3) :
  (room_length + 2 * veranda_length_side) * (room_width + 2 * veranda_width_side) - 
  room_length * room_width = 204 := by
  sorry

end veranda_area_l1805_180583


namespace abs_a_minus_b_equals_eight_l1805_180513

theorem abs_a_minus_b_equals_eight (a b : ℚ) 
  (h : |a + b| + (b - 4)^2 = 0) : 
  |a - b| = 8 := by
sorry

end abs_a_minus_b_equals_eight_l1805_180513


namespace doughnuts_left_l1805_180576

/-- The number of doughnuts in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of doughnuts in the box -/
def boxDozens : ℕ := 2

/-- The number of doughnuts eaten -/
def eatenDoughnuts : ℕ := 8

/-- Theorem: Given a box with 2 dozen doughnuts and 8 doughnuts eaten, 
    the number of doughnuts left is 16 -/
theorem doughnuts_left : 
  boxDozens * dozen - eatenDoughnuts = 16 := by
  sorry

end doughnuts_left_l1805_180576


namespace smallest_n_for_candy_purchase_l1805_180561

theorem smallest_n_for_candy_purchase : ∃ n : ℕ+, 
  (∀ m : ℕ+, (15 * m).gcd 10 = 10 ∧ (15 * m).gcd 16 = 16 ∧ (15 * m).gcd 18 = 18 → n ≤ m) ∧
  (15 * n).gcd 10 = 10 ∧ (15 * n).gcd 16 = 16 ∧ (15 * n).gcd 18 = 18 ∧
  n = 48 :=
by sorry

end smallest_n_for_candy_purchase_l1805_180561


namespace camping_site_problem_l1805_180564

theorem camping_site_problem (total : ℕ) (two_weeks_ago : ℕ) (difference : ℕ) :
  total = 150 →
  two_weeks_ago = 40 →
  difference = 10 →
  ∃ (three_weeks_ago last_week : ℕ),
    three_weeks_ago + two_weeks_ago + last_week = total ∧
    two_weeks_ago = three_weeks_ago + difference ∧
    last_week = 80 :=
by
  sorry

#check camping_site_problem

end camping_site_problem_l1805_180564


namespace sticker_pages_l1805_180566

theorem sticker_pages (stickers_per_page : ℕ) (remaining_stickers : ℕ) : 
  (stickers_per_page = 20 ∧ remaining_stickers = 220) → 
  ∃ (initial_pages : ℕ), 
    initial_pages * stickers_per_page - stickers_per_page = remaining_stickers ∧ 
    initial_pages = 12 :=
by
  sorry

end sticker_pages_l1805_180566


namespace smallest_with_16_divisors_exactly_16_divisors_210_smallest_positive_integer_with_16_divisors_l1805_180563

def number_of_divisors (n : ℕ) : ℕ :=
  (Nat.divisors n).card

theorem smallest_with_16_divisors : 
  ∀ n : ℕ, n > 0 → number_of_divisors n = 16 → n ≥ 210 :=
by
  sorry

theorem exactly_16_divisors_210 : number_of_divisors 210 = 16 :=
by
  sorry

theorem smallest_positive_integer_with_16_divisors : 
  ∀ n : ℕ, n > 0 → number_of_divisors n = 16 → n ≥ 210 ∧ number_of_divisors 210 = 16 :=
by
  sorry

end smallest_with_16_divisors_exactly_16_divisors_210_smallest_positive_integer_with_16_divisors_l1805_180563


namespace stating_sum_of_digits_special_product_l1805_180567

/-- 
Represents the product of numbers of the form (10^k - 1) where k is a power of 2 up to 2^n.
-/
def specialProduct (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * (10^(2^i) - 1)) 9

/-- 
Represents the sum of digits of a natural number in decimal notation.
-/
def sumOfDigits (m : ℕ) : ℕ :=
  sorry

/-- 
Theorem stating that the sum of digits of the special product is equal to 9 · 2^n.
-/
theorem sum_of_digits_special_product (n : ℕ) : 
  sumOfDigits (specialProduct n) = 9 * 2^n := by
  sorry

end stating_sum_of_digits_special_product_l1805_180567


namespace kevin_watermelon_weight_l1805_180506

theorem kevin_watermelon_weight :
  let first_watermelon : ℝ := 9.91
  let second_watermelon : ℝ := 4.11
  let total_weight : ℝ := first_watermelon + second_watermelon
  total_weight = 14.02 := by sorry

end kevin_watermelon_weight_l1805_180506


namespace arithmetic_sequence_sum_l1805_180535

/-- Given an arithmetic sequence {a_n} where a₂ = 3a₅ - 6, prove that S₉ = 27 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- sum formula for arithmetic sequence
  a 2 = 3 * a 5 - 6 →                   -- given condition
  S 9 = 27 :=
by sorry

end arithmetic_sequence_sum_l1805_180535


namespace y_squared_eq_three_x_squared_plus_one_l1805_180551

/-- Sequence x defined recursively -/
def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * x (n + 1) - x n

/-- Sequence y defined recursively -/
def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * y (n + 1) - y n

/-- Main theorem: For all natural numbers n, y(n)² = 3x(n)² + 1 -/
theorem y_squared_eq_three_x_squared_plus_one (n : ℕ) : (y n)^2 = 3*(x n)^2 + 1 := by
  sorry

end y_squared_eq_three_x_squared_plus_one_l1805_180551


namespace average_difference_l1805_180522

theorem average_difference (x : ℝ) : 
  (10 + 30 + 50) / 3 = (20 + 40 + x) / 3 + 8 → x = 6 := by
sorry

end average_difference_l1805_180522


namespace number_puzzle_l1805_180534

theorem number_puzzle (x : ℤ) (h : x - 69 = 37) : x + 55 = 161 := by
  sorry

end number_puzzle_l1805_180534


namespace mayoral_election_votes_l1805_180515

theorem mayoral_election_votes (candidate_x candidate_y other_candidate : ℕ) : 
  candidate_x = candidate_y + (candidate_y / 2) →
  candidate_y = other_candidate - (other_candidate * 2 / 5) →
  candidate_x = 22500 →
  other_candidate = 25000 := by
sorry

end mayoral_election_votes_l1805_180515


namespace quadratic_always_two_roots_l1805_180581

theorem quadratic_always_two_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (∀ x : ℝ, x^2 - m*x + m - 2 = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end quadratic_always_two_roots_l1805_180581


namespace optimal_triangle_game_l1805_180533

open Real

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- A point inside a triangle -/
def pointInside (t : Triangle) (X : ℝ × ℝ) : Prop := sorry

/-- The sum of areas of three triangles formed by connecting a point to three pairs of points on the sides of the original triangle -/
def sumOfAreas (t : Triangle) (X : ℝ × ℝ) : ℝ := sorry

theorem optimal_triangle_game (t : Triangle) (h : t.area = 1) :
  ∃ (X : ℝ × ℝ), pointInside t X ∧ sumOfAreas t X = 1/3 ∧
  ∀ (Y : ℝ × ℝ), pointInside t Y → sumOfAreas t Y ≥ 1/3 := by sorry

end optimal_triangle_game_l1805_180533


namespace circle_radius_theorem_l1805_180574

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if two circles touch externally -/
def touch_externally (c1 c2 : Circle) : Prop := sorry

/-- Checks if a point lies on a circle -/
def on_circle (p : Point) (c : Circle) : Prop := sorry

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop := sorry

theorem circle_radius_theorem (C1 C2 : Circle) (P Q R : Point) :
  C1.radius = 12 →
  touch_externally C1 C2 →
  on_circle P C1 →
  on_circle P C2 →
  on_circle Q C1 →
  on_circle R C2 →
  collinear P Q R →
  distance P Q = 7 →
  distance P R = 17 →
  C2.radius = 10 := by sorry

end circle_radius_theorem_l1805_180574


namespace average_weight_solution_l1805_180524

def average_weight_problem (d e f : ℝ) : Prop :=
  (d + e + f) / 3 = 42 ∧
  (e + f) / 2 = 41 ∧
  e = 26 →
  (d + e) / 2 = 35

theorem average_weight_solution :
  ∀ d e f : ℝ, average_weight_problem d e f :=
by
  sorry

end average_weight_solution_l1805_180524


namespace x_value_proof_l1805_180570

theorem x_value_proof (x : ℝ) (h1 : x ≠ 0) (h2 : Real.sqrt ((5 * x) / 3) = x) : x = 5 / 3 := by
  sorry

end x_value_proof_l1805_180570


namespace y_plus_z_negative_l1805_180538

theorem y_plus_z_negative (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hy : 0 < y ∧ y < 1) 
  (hz : -2 < z ∧ z < -1) : 
  y + z < 0 := by
  sorry

end y_plus_z_negative_l1805_180538


namespace quadruplet_babies_l1805_180556

theorem quadruplet_babies (total_babies : ℕ) 
  (h_total : total_babies = 1500)
  (h_triplets : ∃ c : ℕ, 5 * c = number_of_triplet_sets)
  (h_twins : number_of_twin_sets = 2 * number_of_triplet_sets)
  (h_quintuplets : number_of_quintuplet_sets = number_of_quadruplet_sets / 2)
  (h_sum : 2 * number_of_twin_sets + 3 * number_of_triplet_sets + 
           4 * number_of_quadruplet_sets + 5 * number_of_quintuplet_sets = total_babies) :
  4 * number_of_quadruplet_sets = 145 :=
by sorry

-- Define variables
variable (number_of_twin_sets number_of_triplet_sets number_of_quadruplet_sets number_of_quintuplet_sets : ℕ)

end quadruplet_babies_l1805_180556


namespace bus_stop_problem_l1805_180543

/-- The number of children who got on the bus at a stop -/
def children_at_stop (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem bus_stop_problem (initial : ℕ) (final : ℕ) 
  (h1 : initial = 18) 
  (h2 : final = 25) :
  children_at_stop initial final = 7 := by
  sorry

end bus_stop_problem_l1805_180543


namespace total_surveys_completed_l1805_180550

def regular_rate : ℚ := 10
def cellphone_rate : ℚ := regular_rate * (1 + 30 / 100)
def cellphone_surveys : ℕ := 60
def total_earnings : ℚ := 1180

theorem total_surveys_completed :
  ∃ (regular_surveys : ℕ),
    (regular_surveys : ℚ) * regular_rate + 
    (cellphone_surveys : ℚ) * cellphone_rate = total_earnings ∧
    regular_surveys + cellphone_surveys = 100 :=
by sorry

end total_surveys_completed_l1805_180550


namespace terrier_hush_interval_terrier_hush_interval_is_two_l1805_180505

/-- The interval at which a terrier's owner hushes it, given the following conditions:
  - The poodle barks twice for every one time the terrier barks.
  - The terrier's owner says "hush" six times before the dogs stop barking.
  - The poodle barked 24 times. -/
theorem terrier_hush_interval : ℕ :=
  let poodle_barks : ℕ := 24
  let poodle_to_terrier_ratio : ℕ := 2
  let total_hushes : ℕ := 6
  let terrier_barks : ℕ := poodle_barks / poodle_to_terrier_ratio
  terrier_barks / total_hushes

/-- Proof that the terrier_hush_interval is equal to 2 -/
theorem terrier_hush_interval_is_two : terrier_hush_interval = 2 := by
  sorry

end terrier_hush_interval_terrier_hush_interval_is_two_l1805_180505


namespace gcf_of_75_and_125_l1805_180597

theorem gcf_of_75_and_125 : Nat.gcd 75 125 = 25 := by
  sorry

end gcf_of_75_and_125_l1805_180597


namespace abs_square_not_always_equal_to_value_l1805_180528

theorem abs_square_not_always_equal_to_value : ¬ ∀ a : ℝ, |a^2| = a := by
  sorry

end abs_square_not_always_equal_to_value_l1805_180528


namespace delicious_delhi_bill_l1805_180593

/-- Calculates the total bill for a meal at Delicious Delhi restaurant --/
def calculate_bill (
  samosa_price : ℚ)
  (pakora_price : ℚ)
  (lassi_price : ℚ)
  (biryani_price : ℚ)
  (naan_price : ℚ)
  (samosa_quantity : ℕ)
  (pakora_quantity : ℕ)
  (lassi_quantity : ℕ)
  (biryani_quantity : ℕ)
  (naan_quantity : ℕ)
  (biryani_discount_rate : ℚ)
  (service_fee_rate : ℚ)
  (tip_rate : ℚ)
  (tax_rate : ℚ) : ℚ :=
  sorry

theorem delicious_delhi_bill :
  calculate_bill 2 3 2 (11/2) (3/2) 3 4 1 2 1 (1/10) (3/100) (1/5) (2/25) = 4125/100 :=
sorry

end delicious_delhi_bill_l1805_180593


namespace angle_P_measure_l1805_180519

/-- A quadrilateral with specific angle relationships -/
structure SpecialQuadrilateral where
  P : ℝ  -- Angle P in degrees
  Q : ℝ  -- Angle Q in degrees
  R : ℝ  -- Angle R in degrees
  S : ℝ  -- Angle S in degrees
  angle_relation : P = 3*Q ∧ P = 4*R ∧ P = 6*S
  sum_360 : P + Q + R + S = 360

/-- The measure of angle P in a SpecialQuadrilateral is 206 degrees -/
theorem angle_P_measure (quad : SpecialQuadrilateral) : 
  ⌊quad.P⌋ = 206 := by sorry

end angle_P_measure_l1805_180519


namespace equation_solution_l1805_180596

theorem equation_solution (m : ℝ) : 
  (∃ x : ℝ, x = 3 ∧ 4 * (x - 1) - m * x + 6 = 8) → 
  m^2 + 2*m - 3 = 5 := by
sorry

end equation_solution_l1805_180596


namespace congruence_problem_l1805_180512

theorem congruence_problem (x : ℤ) : 
  (5 * x + 11) % 19 = 3 → (3 * x + 7) % 19 = 6 := by sorry

end congruence_problem_l1805_180512


namespace polynomial_factorization_l1805_180546

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = 
  (a - b)^2 * (b - c)^2 * (c - a)^2 * (a + b + c) := by
sorry

end polynomial_factorization_l1805_180546


namespace volume_of_rotated_composite_shape_l1805_180559

/-- The volume of a solid formed by rotating a composite shape about the x-axis -/
theorem volume_of_rotated_composite_shape (π : ℝ) :
  let rectangle1_height : ℝ := 6
  let rectangle1_width : ℝ := 1
  let rectangle2_height : ℝ := 2
  let rectangle2_width : ℝ := 4
  let semicircle_diameter : ℝ := 2
  
  let volume_cylinder1 : ℝ := π * rectangle1_height^2 * rectangle1_width
  let volume_cylinder2 : ℝ := π * rectangle2_height^2 * rectangle2_width
  let volume_hemisphere : ℝ := (2/3) * π * (semicircle_diameter/2)^3
  
  let total_volume : ℝ := volume_cylinder1 + volume_cylinder2 + volume_hemisphere
  
  total_volume = 52 * (2/3) * π :=
by sorry

end volume_of_rotated_composite_shape_l1805_180559


namespace fractional_equation_root_l1805_180537

theorem fractional_equation_root (x m : ℝ) : 
  (∃ x, x / (x - 3) - 2 = (m - 1) / (x - 3) ∧ x ≠ 3) → m = 4 := by
  sorry

end fractional_equation_root_l1805_180537


namespace triangle_problem_l1805_180530

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a * sin (2 * B) = Real.sqrt 3 * b * sin A →
  cos A = 1 / 3 →
  B = π / 6 ∧ sin C = (2 * Real.sqrt 6 + 1) / 6 :=
by sorry

end triangle_problem_l1805_180530

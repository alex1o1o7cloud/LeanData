import Mathlib

namespace ramanujan_hardy_game_l3666_366692

theorem ramanujan_hardy_game (h r : ℂ) : 
  h * r = 48 - 12*I ∧ h = 6 + 2*I → r = 39/5 - 21/5*I :=
by sorry

end ramanujan_hardy_game_l3666_366692


namespace daeyoung_pencils_l3666_366645

/-- Given the conditions of Daeyoung's purchase, prove that he bought 3 pencils. -/
theorem daeyoung_pencils :
  ∀ (E P : ℕ),
  E + P = 8 →
  300 * E + 500 * P = 3000 →
  E ≥ 1 →
  P ≥ 1 →
  P = 3 :=
by
  sorry

end daeyoung_pencils_l3666_366645


namespace tower_count_mod_1000_l3666_366624

/-- A function that calculates the number of towers for n cubes -/
def tower_count (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 32
  | m + 4 => 4 * tower_count (m + 3)

/-- The theorem stating that the number of towers for 10 cubes is congruent to 288 mod 1000 -/
theorem tower_count_mod_1000 :
  tower_count 10 ≡ 288 [MOD 1000] :=
sorry

end tower_count_mod_1000_l3666_366624


namespace set_union_equality_l3666_366648

-- Define the sets M and N
def M : Set ℝ := {x | x^2 < 4*x}
def N : Set ℝ := {x | |x - 1| ≥ 3}

-- Define the union set
def unionSet : Set ℝ := {x | x ≤ -2 ∨ x > 0}

-- Theorem statement
theorem set_union_equality : M ∪ N = unionSet := by
  sorry

end set_union_equality_l3666_366648


namespace function_properties_l3666_366678

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1 / Real.exp x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  Real.log ((3 - a) * Real.exp x + 1) - Real.log (3 * a) - 2 * x

theorem function_properties :
  (∃ (m : ℝ), m = 2 ∧ ∀ x : ℝ, x ≥ 0 → f x ≥ m) ∧
  (∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≥ 0 ∧ x₂ ≥ 0 → g a x₁ ≤ f x₂ - 2) →
    a ≥ 1 ∧ a ≤ 3) := by sorry

end function_properties_l3666_366678


namespace edward_work_hours_l3666_366672

theorem edward_work_hours (hourly_rate : ℝ) (max_regular_hours : ℕ) (total_earnings : ℝ) :
  hourly_rate = 7 →
  max_regular_hours = 40 →
  total_earnings = 210 →
  ∃ (hours_worked : ℕ), hours_worked = 30 ∧ (hours_worked : ℝ) * hourly_rate = total_earnings :=
by sorry

end edward_work_hours_l3666_366672


namespace num_orderings_eq_1554_l3666_366646

/-- The number of designs --/
def n : ℕ := 12

/-- The set of all design labels --/
def designs : Finset ℕ := Finset.range n

/-- The set of completed designs --/
def completed : Finset ℕ := {10, 11}

/-- The set of designs that could still be in the pile --/
def remaining : Finset ℕ := (designs \ completed).filter (· ≤ 9)

/-- The number of possible orderings for completing the remaining designs --/
def num_orderings : ℕ :=
  Finset.sum (Finset.powerset remaining) (fun S => S.card + 2)

theorem num_orderings_eq_1554 : num_orderings = 1554 := by
  sorry

end num_orderings_eq_1554_l3666_366646


namespace expression_factorization_l3666_366619

theorem expression_factorization (x : ℝ) : 
  (8 * x^4 + 34 * x^3 - 120 * x + 150) - (-2 * x^4 + 12 * x^3 - 5 * x + 10) = 
  5 * x * (2 * x^3 + (22/5) * x^2 - 23 * x + 28) := by
sorry

end expression_factorization_l3666_366619


namespace carly_grape_lollipops_l3666_366652

/-- The number of grape lollipops in Carly's collection --/
def grape_lollipops (total : ℕ) (cherry : ℕ) (non_cherry_flavors : ℕ) : ℕ :=
  (total - cherry) / non_cherry_flavors

/-- Theorem stating the number of grape lollipops in Carly's collection --/
theorem carly_grape_lollipops : 
  grape_lollipops 42 (42 / 2) 3 = 7 := by
  sorry

end carly_grape_lollipops_l3666_366652


namespace necessary_not_sufficient_condition_l3666_366640

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem necessary_not_sufficient_condition (a b : ℝ) :
  let z : ℂ := ⟨a, b⟩
  (is_pure_imaginary z → a = 0) ∧
  ¬(a = 0 → is_pure_imaginary z) :=
sorry

end necessary_not_sufficient_condition_l3666_366640


namespace decreasing_quadratic_condition_l3666_366663

-- Define the function f(x) = x^2 + mx + 1
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

-- Define the property of f being decreasing on an interval
def isDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Theorem statement
theorem decreasing_quadratic_condition (m : ℝ) :
  isDecreasingOn (f m) 0 5 → m ≤ -10 :=
sorry

end decreasing_quadratic_condition_l3666_366663


namespace angle_B_not_right_angle_sin_C_over_sin_A_range_l3666_366643

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  2 * Real.sin t.C * Real.sin (t.B - t.A) = 2 * Real.sin t.A * Real.sin t.C - Real.sin t.B ^ 2

-- Theorem 1: Angle B cannot be a right angle
theorem angle_B_not_right_angle (t : Triangle) (h : satisfies_condition t) : 
  t.B ≠ π / 2 := by sorry

-- Theorem 2: Range of sin(C)/sin(A) for acute triangles
theorem sin_C_over_sin_A_range (t : Triangle) (h1 : satisfies_condition t) 
  (h2 : t.A < π / 2 ∧ t.B < π / 2 ∧ t.C < π / 2) : 
  1 / 3 < Real.sin t.C / Real.sin t.A ∧ Real.sin t.C / Real.sin t.A < 5 / 3 := by sorry

end angle_B_not_right_angle_sin_C_over_sin_A_range_l3666_366643


namespace sophie_germain_identity_l3666_366633

theorem sophie_germain_identity (a b : ℝ) : 
  a^4 + 4*b^4 = (a^2 - 2*a*b + 2*b^2) * (a^2 + 2*a*b + 2*b^2) := by
  sorry

end sophie_germain_identity_l3666_366633


namespace unfenced_side_length_is_ten_l3666_366666

/-- Represents a rectangular yard with fencing on three sides -/
structure FencedYard where
  length : ℝ
  width : ℝ
  area : ℝ
  fenceLength : ℝ

/-- The unfenced side length of a rectangular yard -/
def unfencedSideLength (yard : FencedYard) : ℝ := yard.length

/-- Theorem stating the conditions and the result to be proved -/
theorem unfenced_side_length_is_ten
  (yard : FencedYard)
  (area_constraint : yard.area = 200)
  (fence_constraint : yard.fenceLength = 50)
  (rectangle_constraint : yard.area = yard.length * yard.width)
  (fence_sides_constraint : yard.fenceLength = 2 * yard.width + yard.length) :
  unfencedSideLength yard = 10 := by sorry

end unfenced_side_length_is_ten_l3666_366666


namespace shopkeeper_cheating_profit_l3666_366655

/-- The percentage by which the shopkeeper increases the weight when buying from the supplier -/
def supplier_increase_percent : ℝ := 20

/-- The profit percentage the shopkeeper aims to achieve -/
def target_profit_percent : ℝ := 32

/-- The percentage by which the shopkeeper increases the weight when selling to the customer -/
def customer_increase_percent : ℝ := 26.67

theorem shopkeeper_cheating_profit (initial_weight : ℝ) (h : initial_weight > 0) :
  let actual_weight := initial_weight * (1 + supplier_increase_percent / 100)
  let selling_weight := actual_weight * (1 + customer_increase_percent / 100)
  (selling_weight - actual_weight) / initial_weight * 100 = target_profit_percent := by
sorry

end shopkeeper_cheating_profit_l3666_366655


namespace i_times_one_minus_i_squared_eq_two_l3666_366658

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem stating that i(1-i)² = 2 -/
theorem i_times_one_minus_i_squared_eq_two : i * (1 - i)^2 = 2 := by
  sorry

end i_times_one_minus_i_squared_eq_two_l3666_366658


namespace ratio_MBQ_ABQ_l3666_366671

-- Define the points
variable (A B C P Q M : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- BP and BQ bisect ∠ABC
axiom bisect_ABC : angle A B P = angle P B C ∧ angle A B Q = angle Q B C

-- BM trisects ∠PBQ
axiom trisect_PBQ : angle P B M = angle M B Q ∧ 3 * angle M B Q = angle P B Q

-- Theorem to prove
theorem ratio_MBQ_ABQ : angle M B Q / angle A B Q = 1 / 4 := by sorry

end ratio_MBQ_ABQ_l3666_366671


namespace sine_inequality_solution_set_l3666_366607

theorem sine_inequality_solution_set 
  (a : ℝ) 
  (h1 : -1 < a) 
  (h2 : a < 0) 
  (θ : ℝ) 
  (h3 : θ = Real.arcsin a) : 
  {x : ℝ | ∃ (n : ℤ), (2*n - 1)*π - θ < x ∧ x < 2*n*π + θ} = 
  {x : ℝ | Real.sin x < a} := by
  sorry

end sine_inequality_solution_set_l3666_366607


namespace flower_problem_l3666_366605

theorem flower_problem (total : ℕ) (roses_fraction : ℚ) (tulips : ℕ) (carnations : ℕ) : 
  total = 40 →
  roses_fraction = 2 / 5 →
  tulips = 10 →
  carnations = total - (roses_fraction * total).num - tulips →
  carnations = 14 := by
sorry

end flower_problem_l3666_366605


namespace overtime_compensation_l3666_366601

def total_employees : ℕ := 350
def men_pay_rate : ℚ := 10
def women_pay_rate : ℚ := 815/100

theorem overtime_compensation 
  (total_men : ℕ) 
  (men_accepted : ℕ) 
  (h1 : total_men ≤ total_employees) 
  (h2 : men_accepted ≤ total_men) 
  (h3 : ∀ (m : ℕ), m ≤ total_men → 
    men_pay_rate * m + women_pay_rate * (total_employees - m) = 
    men_pay_rate * men_accepted + women_pay_rate * (total_employees - total_men)) :
  women_pay_rate * (total_employees - total_men) = 122250/100 := by
  sorry

end overtime_compensation_l3666_366601


namespace polynomial_coefficient_sum_l3666_366610

/-- Given a polynomial equation, prove the sum of specific coefficients --/
theorem polynomial_coefficient_sum :
  ∀ (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ),
  (∀ x : ℝ, a + a₁ * (x + 2) + a₂ * (x + 2)^2 + a₃ * (x + 2)^3 + a₄ * (x + 2)^4 + 
             a₅ * (x + 2)^5 + a₆ * (x + 2)^6 + a₇ * (x + 2)^7 + a₈ * (x + 2)^8 + 
             a₉ * (x + 2)^9 + a₁₀ * (x + 2)^10 + a₁₁ * (x + 2)^11 + a₁₂ * (x + 2)^12 = 
             (x^2 - 2*x - 2)^6) →
  2*a₂ + 6*a₃ + 12*a₄ + 20*a₅ + 30*a₆ + 42*a₇ + 56*a₈ + 72*a₉ + 90*a₁₀ + 110*a₁₁ + 132*a₁₂ = 492 :=
by sorry

end polynomial_coefficient_sum_l3666_366610


namespace square_of_three_times_sqrt_two_l3666_366612

theorem square_of_three_times_sqrt_two : (3 * Real.sqrt 2) ^ 2 = 18 := by
  sorry

end square_of_three_times_sqrt_two_l3666_366612


namespace steves_return_speed_l3666_366686

/-- Proves that given a round trip of 60 km (30 km each way), where the return speed is twice 
    the outbound speed, and the total travel time is 6 hours, the return speed is 15 km/h. -/
theorem steves_return_speed 
  (distance : ℝ) 
  (total_time : ℝ) 
  (speed_to_work : ℝ) 
  (speed_from_work : ℝ) : 
  distance = 30 →
  total_time = 6 →
  speed_from_work = 2 * speed_to_work →
  distance / speed_to_work + distance / speed_from_work = total_time →
  speed_from_work = 15 := by
  sorry


end steves_return_speed_l3666_366686


namespace intersection_distance_l3666_366673

/-- The distance between the points of intersection of x^2 + y = 12 and x + y = 12 is √2 -/
theorem intersection_distance : 
  ∃ (p₁ p₂ : ℝ × ℝ), 
    (p₁.1^2 + p₁.2 = 12 ∧ p₁.1 + p₁.2 = 12) ∧ 
    (p₂.1^2 + p₂.2 = 12 ∧ p₂.1 + p₂.2 = 12) ∧ 
    p₁ ≠ p₂ ∧
    Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) = Real.sqrt 2 := by
  sorry

end intersection_distance_l3666_366673


namespace prove_b_value_l3666_366664

theorem prove_b_value (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 35 * b) : b = 63 := by
  sorry

end prove_b_value_l3666_366664


namespace average_candies_sikyung_l3666_366641

def sikyung_group : Finset ℕ := {16, 22, 30, 26, 18, 20}

theorem average_candies_sikyung : 
  (sikyung_group.sum id) / sikyung_group.card = 22 := by
  sorry

end average_candies_sikyung_l3666_366641


namespace simple_interest_principal_l3666_366661

/-- Simple interest calculation -/
theorem simple_interest_principal (interest rate time principal : ℝ) :
  interest = principal * rate * time →
  rate = 0.09 →
  time = 1 →
  interest = 900 →
  principal = 10000 := by
sorry

end simple_interest_principal_l3666_366661


namespace geometric_sequence_ratio_l3666_366635

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  sum_property : a 2 + a 8 = 15
  product_property : a 3 * a 7 = 36

/-- The theorem stating the possible values of a_19 / a_13 -/
theorem geometric_sequence_ratio 
  (seq : GeometricSequence) : 
  seq.a 19 / seq.a 13 = 1/4 ∨ seq.a 19 / seq.a 13 = 4 := by
  sorry

end geometric_sequence_ratio_l3666_366635


namespace range_of_fraction_l3666_366609

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∃ (z : ℝ), z = x / y ∧ 1/6 ≤ z ∧ z ≤ 4/3 :=
sorry

end range_of_fraction_l3666_366609


namespace probability_odd_and_multiple_of_3_l3666_366683

/-- Represents a fair die with n sides -/
structure Die (n : ℕ) where
  sides : Finset (Fin n)
  fair : sides.card = n

/-- The event of rolling an odd number on a die -/
def oddEvent (d : Die n) : Finset (Fin n) :=
  d.sides.filter (λ x => x.val % 2 = 1)

/-- The event of rolling a multiple of 3 on a die -/
def multipleOf3Event (d : Die n) : Finset (Fin n) :=
  d.sides.filter (λ x => x.val % 3 = 0)

/-- The probability of an event occurring on a fair die -/
def probability (d : Die n) (event : Finset (Fin n)) : ℚ :=
  event.card / d.sides.card

theorem probability_odd_and_multiple_of_3 
  (d8 : Die 8) 
  (d12 : Die 12) : 
  probability d8 (oddEvent d8) * probability d12 (multipleOf3Event d12) = 1/6 := by
sorry

end probability_odd_and_multiple_of_3_l3666_366683


namespace intersection_M_N_l3666_366676

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 2 < 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l3666_366676


namespace c_value_l3666_366685

theorem c_value (a b c : ℚ) : 
  8 = (2 / 100) * a → 
  2 = (8 / 100) * b → 
  c = b / a → 
  c = 1 / 16 := by
sorry

end c_value_l3666_366685


namespace max_value_inequality_l3666_366650

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a^2 + b^2 + 2*c^2 = 1) : 
  Real.sqrt 2 * a * b + 2 * b * c + 7 * a * c ≤ 2 * Real.sqrt 2 := by
  sorry

end max_value_inequality_l3666_366650


namespace solve_equation_l3666_366618

theorem solve_equation (C : ℝ) (h : 5 * C - 6 = 34) : C = 8 := by
  sorry

end solve_equation_l3666_366618


namespace inequality_proof_l3666_366696

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_condition : a + b + c = 1) : 
  (a + 1/a) * (b + 1/b) * (c + 1/c) ≥ 1000/27 := by
sorry

end inequality_proof_l3666_366696


namespace quadratic_factorization_l3666_366622

/-- Given a quadratic expression 2y^2 - 5y - 12 that can be factored as (2y + a)(y + b) 
    where a and b are integers, prove that a - b = 7 -/
theorem quadratic_factorization (a b : ℤ) : 
  (∀ y, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b)) → a - b = 7 := by
  sorry

end quadratic_factorization_l3666_366622


namespace lcm_of_denominators_l3666_366623

theorem lcm_of_denominators (a b c d e : ℕ) (ha : a = 3) (hb : b = 4) (hc : c = 6) (hd : d = 8) (he : e = 9) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = 72 := by
  sorry

end lcm_of_denominators_l3666_366623


namespace max_area_cyclic_quadrilateral_l3666_366660

/-- The maximum area of a cyclic quadrilateral with side lengths 1, 4, 7, and 8 is 18 -/
theorem max_area_cyclic_quadrilateral :
  let a : ℝ := 1
  let b : ℝ := 4
  let c : ℝ := 7
  let d : ℝ := 8
  let s : ℝ := (a + b + c + d) / 2
  let area : ℝ := Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d))
  area = 18 := by sorry

end max_area_cyclic_quadrilateral_l3666_366660


namespace basketball_lineups_l3666_366697

def total_players : ℕ := 12
def players_per_lineup : ℕ := 5
def point_guards_per_lineup : ℕ := 1

def number_of_lineups : ℕ :=
  total_players * (Nat.choose (total_players - 1) (players_per_lineup - 1))

theorem basketball_lineups :
  number_of_lineups = 3960 := by
  sorry

end basketball_lineups_l3666_366697


namespace simplify_fraction_l3666_366651

theorem simplify_fraction : (66 : ℚ) / 4356 = 1 / 66 := by sorry

end simplify_fraction_l3666_366651


namespace largest_fraction_l3666_366688

theorem largest_fraction (a b c d e : ℚ) 
  (ha : a = 3/10) (hb : b = 9/20) (hc : c = 12/25) (hd : d = 27/50) (he : e = 49/100) :
  d = max a (max b (max c (max d e))) :=
sorry

end largest_fraction_l3666_366688


namespace xy_value_given_equation_l3666_366684

theorem xy_value_given_equation :
  ∀ x y : ℝ, 2*x^2 + 2*x*y + y^2 - 6*x + 9 = 0 → x^y = 1/27 := by
  sorry

end xy_value_given_equation_l3666_366684


namespace family_probability_l3666_366665

theorem family_probability : 
  let p_boy : ℝ := 1/2
  let p_girl : ℝ := 1/2
  let num_children : ℕ := 4
  let p_all_boys : ℝ := p_boy ^ num_children
  let p_all_girls : ℝ := p_girl ^ num_children
  let p_at_least_one_of_each : ℝ := 1 - p_all_boys - p_all_girls
  p_at_least_one_of_each = 7/8 := by
sorry

end family_probability_l3666_366665


namespace isosceles_triangle_angle_measure_l3666_366657

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define an isosceles triangle
def Isosceles (t : Triangle A B C) : Prop :=
  ‖A - B‖ = ‖B - C‖

-- Define the angle measure function
def AngleMeasure (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem isosceles_triangle_angle_measure 
  (A B C D : ℝ × ℝ) 
  (t : Triangle A B C) 
  (h_isosceles : Isosceles t) 
  (h_angle_C : AngleMeasure B C A = 50) :
  AngleMeasure C B D = 115 := by
  sorry

end isosceles_triangle_angle_measure_l3666_366657


namespace distance_between_trees_l3666_366647

/-- Given a yard with trees planted at equal distances, calculate the distance between consecutive trees. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) : 
  yard_length = 360 ∧ num_trees = 31 → 
  (yard_length / (num_trees - 1 : ℝ)) = 12 := by
  sorry

end distance_between_trees_l3666_366647


namespace prob_A_given_B_value_l3666_366617

/-- The number of people visiting tourist spots -/
def num_people : ℕ := 4

/-- The number of tourist spots -/
def num_spots : ℕ := 4

/-- Event A: All 4 people visit different spots -/
def event_A : Prop := True

/-- Event B: Xiao Zhao visits a spot alone -/
def event_B : Prop := True

/-- The number of ways for 3 people to visit 3 spots -/
def ways_3_people_3_spots : ℕ := 3 * 3 * 3

/-- The number of ways for Xiao Zhao to visit a spot alone -/
def ways_xiao_zhao_alone : ℕ := num_spots * ways_3_people_3_spots

/-- The number of ways 4 people can visit different spots -/
def ways_all_different : ℕ := 4 * 3 * 2 * 1

/-- The probability of event A given event B -/
def prob_A_given_B : ℚ := ways_all_different / ways_xiao_zhao_alone

theorem prob_A_given_B_value : prob_A_given_B = 2 / 9 := by
  sorry

end prob_A_given_B_value_l3666_366617


namespace parallel_line_slope_l3666_366634

theorem parallel_line_slope (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  let m := -a / b
  (∀ x y, a * x + b * y = c) → m = -1/2 :=
by sorry

end parallel_line_slope_l3666_366634


namespace quadratic_root_value_l3666_366693

theorem quadratic_root_value (k : ℚ) : 
  ((-25 - Real.sqrt 369) / 12 : ℝ) ∈ {x : ℝ | 6 * x^2 + 25 * x + k = 0} → k = 32/3 := by
  sorry

end quadratic_root_value_l3666_366693


namespace roof_dimension_difference_l3666_366637

theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 5 * width →
  width * length = 720 →
  length - width = 48 := by
sorry

end roof_dimension_difference_l3666_366637


namespace rhombus_area_l3666_366649

/-- The area of a rhombus with side length √125 and diagonal difference 8 is 60.5 -/
theorem rhombus_area (side : ℝ) (diag_diff : ℝ) (area : ℝ) : 
  side = Real.sqrt 125 →
  diag_diff = 8 →
  area = (side^2 * Real.sqrt (4 - (diag_diff / side)^2)) / 2 →
  area = 60.5 := by sorry

end rhombus_area_l3666_366649


namespace min_value_expression_min_value_attainable_l3666_366695

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * a^3 + 16 * b^3 + 25 * c^3 + 1 / (5 * a * b * c) ≥ 4 * Real.sqrt 3 :=
by sorry

theorem min_value_attainable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  4 * a^3 + 16 * b^3 + 25 * c^3 + 1 / (5 * a * b * c) = 4 * Real.sqrt 3 :=
by sorry

end min_value_expression_min_value_attainable_l3666_366695


namespace equation_one_solution_equation_two_no_solution_l3666_366682

-- Problem 1
theorem equation_one_solution (x : ℝ) :
  (2 / x + 1 / (x * (x - 2)) = 5 / (2 * x)) ↔ x = 4 :=
sorry

-- Problem 2
theorem equation_two_no_solution :
  ¬∃ (x : ℝ), (5 * x - 4) / (x - 2) = (4 * x + 10) / (3 * x - 6) - 1 :=
sorry

end equation_one_solution_equation_two_no_solution_l3666_366682


namespace expression_value_at_one_fifth_l3666_366694

theorem expression_value_at_one_fifth :
  let x : ℚ := 1/5
  (x^2 - 4) / (x^2 - 2*x) = 11 :=
by sorry

end expression_value_at_one_fifth_l3666_366694


namespace condition_relationship_l3666_366674

theorem condition_relationship (x : ℝ) : 
  (∀ x, abs x < 1 → x > -1) ∧ 
  (∃ x, x > -1 ∧ ¬(abs x < 1)) :=
sorry

end condition_relationship_l3666_366674


namespace mystery_book_shelves_l3666_366681

theorem mystery_book_shelves (books_per_shelf : ℕ) (picture_book_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 6 →
  picture_book_shelves = 4 →
  total_books = 54 →
  (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf = 5 :=
by sorry

end mystery_book_shelves_l3666_366681


namespace geometric_mean_max_value_l3666_366613

theorem geometric_mean_max_value (a b : ℝ) (h : a^2 = (1 + 2*b) * (1 - 2*b)) :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ x, x = (8*a*b)/(|a| + 2*|b|) → x ≤ M :=
sorry

end geometric_mean_max_value_l3666_366613


namespace triangle_side_length_l3666_366670

theorem triangle_side_length (A B C : ℝ × ℝ) :
  let AC := Real.sqrt 3
  let AB := 2
  let angle_B := 60 * Real.pi / 180
  let BC := Real.sqrt ((AC^2 + AB^2) - 2 * AC * AB * Real.cos angle_B)
  AC = Real.sqrt 3 ∧ AB = 2 ∧ angle_B = 60 * Real.pi / 180 →
  BC = 1 := by
sorry

end triangle_side_length_l3666_366670


namespace radio_dealer_profit_l3666_366630

theorem radio_dealer_profit (n d : ℕ) (h_d_pos : d > 0) : 
  (3 * (d / n / 3) + (n - 3) * (d / n + 10) - d = 100) → n ≥ 13 :=
by
  sorry

end radio_dealer_profit_l3666_366630


namespace john_swimming_laps_l3666_366690

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem john_swimming_laps :
  base7ToBase10 [3, 1, 2, 5] = 1823 := by
  sorry

end john_swimming_laps_l3666_366690


namespace johnny_total_planks_l3666_366659

/-- Represents the number of planks needed for a table surface -/
def surface_planks (table_type : String) : ℕ :=
  match table_type with
  | "small" => 3
  | "medium" => 5
  | "large" => 7
  | _ => 0

/-- Represents the number of planks needed for table legs -/
def leg_planks : ℕ := 4

/-- Calculates the total planks needed for a given number of tables of a specific type -/
def planks_for_table_type (table_type : String) (num_tables : ℕ) : ℕ :=
  num_tables * (surface_planks table_type + leg_planks)

/-- Theorem: The total number of planks needed for Johnny's tables is 50 -/
theorem johnny_total_planks : 
  planks_for_table_type "small" 3 + 
  planks_for_table_type "medium" 2 + 
  planks_for_table_type "large" 1 = 50 := by
  sorry


end johnny_total_planks_l3666_366659


namespace inequality_proof_l3666_366615

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : c > 1) : 
  a * b^c > b * a^c := by
sorry

end inequality_proof_l3666_366615


namespace probability_three_digit_ending_4_divisible_by_3_l3666_366638

/-- A three-digit positive integer ending in 4 -/
def ThreeDigitEndingIn4 : Type := { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 10 = 4 }

/-- The count of three-digit positive integers ending in 4 -/
def totalCount : ℕ := 90

/-- The count of three-digit positive integers ending in 4 that are divisible by 3 -/
def divisibleBy3Count : ℕ := 33

/-- The probability that a three-digit positive integer ending in 4 is divisible by 3 -/
def probabilityDivisibleBy3 : ℚ := divisibleBy3Count / totalCount

theorem probability_three_digit_ending_4_divisible_by_3 :
  probabilityDivisibleBy3 = 11 / 30 := by sorry

end probability_three_digit_ending_4_divisible_by_3_l3666_366638


namespace quadratic_root_value_l3666_366603

theorem quadratic_root_value (k : ℝ) : 
  (∀ x : ℝ, 5 * x^2 + 20 * x + k = 0 ↔ x = (-20 + Real.sqrt 60) / 10 ∨ x = (-20 - Real.sqrt 60) / 10) 
  → k = 17 := by
sorry

end quadratic_root_value_l3666_366603


namespace line_through_points_l3666_366675

/-- Given a line x = 3y + 5 passing through points (m, n) and (m + 2, n + q), prove that q = 2/3 -/
theorem line_through_points (m n : ℝ) : 
  (∃ q : ℝ, m = 3 * n + 5 ∧ m + 2 = 3 * (n + q) + 5) → 
  (∃ q : ℝ, q = 2/3) :=
by sorry

end line_through_points_l3666_366675


namespace expression_as_square_of_binomial_l3666_366616

/-- Represents the expression (-4b-3a)(-3a+4b) -/
def expression (a b : ℝ) : ℝ := (-4*b - 3*a) * (-3*a + 4*b)

/-- Represents the square of binomial form (x - y)(x + y) = x^2 - y^2 -/
def squareOfBinomialForm (x y : ℝ) : ℝ := x^2 - y^2

/-- Theorem stating that the given expression can be rewritten in a form 
    related to the square of a binomial -/
theorem expression_as_square_of_binomial (a b : ℝ) : 
  ∃ (x y : ℝ), expression a b = squareOfBinomialForm x y := by
  sorry

end expression_as_square_of_binomial_l3666_366616


namespace circle_radius_l3666_366628

/-- A circle with center (0, k) where k < -6 is tangent to y = x, y = -x, and y = -6.
    Its radius is 6√2. -/
theorem circle_radius (k : ℝ) (h : k < -6) :
  let center := (0 : ℝ × ℝ)
  let radius := Real.sqrt 2 * 6
  (∀ p : ℝ × ℝ, (p.1 = p.2 ∨ p.1 = -p.2 ∨ p.2 = -6) →
    ‖p - center‖ = radius) →
  radius = Real.sqrt 2 * 6 :=
by sorry


end circle_radius_l3666_366628


namespace total_football_games_l3666_366642

theorem total_football_games (games_attended : ℕ) (games_missed : ℕ) : 
  games_attended = 3 → games_missed = 4 → games_attended + games_missed = 7 :=
by
  sorry

#check total_football_games

end total_football_games_l3666_366642


namespace systems_solutions_l3666_366626

theorem systems_solutions :
  (∃ x y : ℝ, x = 2*y - 1 ∧ 3*x + 4*y = 17 ∧ x = 3 ∧ y = 2) ∧
  (∃ x y : ℝ, 2*x - y = 0 ∧ 3*x - 2*y = 5 ∧ x = -5 ∧ y = -10) :=
by sorry

end systems_solutions_l3666_366626


namespace chocolate_eggs_duration_l3666_366632

/-- The number of chocolate eggs Maddy has -/
def N : ℕ := 40

/-- The number of eggs Maddy eats per weekday -/
def eggs_per_day : ℕ := 2

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weeks the chocolate eggs will last -/
def weeks_lasted : ℕ := N / (eggs_per_day * weekdays)

theorem chocolate_eggs_duration : weeks_lasted = 4 := by
  sorry

end chocolate_eggs_duration_l3666_366632


namespace lisa_coffee_consumption_l3666_366621

/-- The number of cups of coffee Lisa drank -/
def cups_of_coffee : ℕ := sorry

/-- The amount of caffeine in milligrams per cup of coffee -/
def caffeine_per_cup : ℕ := 80

/-- Lisa's daily caffeine limit in milligrams -/
def daily_limit : ℕ := 200

/-- The amount of caffeine Lisa consumed over her daily limit in milligrams -/
def excess_caffeine : ℕ := 40

/-- Theorem stating that Lisa drank 3 cups of coffee -/
theorem lisa_coffee_consumption : cups_of_coffee = 3 := by sorry

end lisa_coffee_consumption_l3666_366621


namespace cricket_run_rate_theorem_l3666_366602

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  targetRuns : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let runsInFirstPart := game.firstPartRunRate * game.firstPartOvers
  let remainingRuns := game.targetRuns - runsInFirstPart
  remainingRuns / remainingOvers

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstPartOvers = 10)
  (h3 : game.firstPartRunRate = 6.2)
  (h4 : game.targetRuns = 282) :
  requiredRunRate game = 5.5 := by
  sorry

#eval requiredRunRate { totalOvers := 50, firstPartOvers := 10, firstPartRunRate := 6.2, targetRuns := 282 }

end cricket_run_rate_theorem_l3666_366602


namespace bella_stamp_difference_l3666_366627

/-- Calculates the difference between truck stamps and rose stamps -/
def stamp_difference (snowflake : ℕ) (truck_surplus : ℕ) (total : ℕ) : ℕ :=
  let truck := snowflake + truck_surplus
  let rose := total - (snowflake + truck)
  truck - rose

/-- Proves that the difference between truck stamps and rose stamps is 13 -/
theorem bella_stamp_difference :
  stamp_difference 11 9 38 = 13 := by
  sorry

end bella_stamp_difference_l3666_366627


namespace arithmetic_sequence_sum_l3666_366667

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₂ + a₃ + a₁₀ + a₁₁ = 36, a₃ + a₁₀ = 18 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 2 + a 3 + a 10 + a 11 = 36 →
  a 3 + a 10 = 18 := by
  sorry

end arithmetic_sequence_sum_l3666_366667


namespace mashed_potatoes_bacon_difference_l3666_366604

/-- The number of students who suggested adding bacon -/
def bacon_students : ℕ := 269

/-- The number of students who suggested adding mashed potatoes -/
def mashed_potatoes_students : ℕ := 330

/-- The number of students who suggested adding tomatoes -/
def tomatoes_students : ℕ := 76

/-- The theorem stating the difference between the number of students who suggested
    mashed potatoes and those who suggested bacon -/
theorem mashed_potatoes_bacon_difference :
  mashed_potatoes_students - bacon_students = 61 := by
  sorry

end mashed_potatoes_bacon_difference_l3666_366604


namespace even_painted_faces_5x5x1_l3666_366653

/-- Represents a 3D rectangular block -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with an even number of painted faces in a given block -/
def count_even_painted_faces (b : Block) : ℕ :=
  sorry

/-- The theorem stating that a 5x5x1 block has 12 cubes with an even number of painted faces -/
theorem even_painted_faces_5x5x1 :
  let b : Block := { length := 5, width := 5, height := 1 }
  count_even_painted_faces b = 12 := by
  sorry

end even_painted_faces_5x5x1_l3666_366653


namespace transformation_is_rotation_and_scaling_l3666_366656

def rotation_90 : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def scaling_2 : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]
def transformation : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]

theorem transformation_is_rotation_and_scaling :
  transformation = scaling_2 * rotation_90 :=
sorry

end transformation_is_rotation_and_scaling_l3666_366656


namespace parallelogram_perimeter_l3666_366668

-- Define a parallelogram
structure Parallelogram :=
  (A B C D : ℝ × ℝ)
  (is_parallelogram : sorry)

-- Define the length of a side
def side_length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter of a parallelogram
def perimeter (p : Parallelogram) : ℝ :=
  side_length p.A p.B + side_length p.B p.C +
  side_length p.C p.D + side_length p.D p.A

-- Theorem statement
theorem parallelogram_perimeter (ABCD : Parallelogram)
  (h1 : side_length ABCD.A ABCD.B = 14)
  (h2 : side_length ABCD.B ABCD.C = 16) :
  perimeter ABCD = 60 := by
  sorry

end parallelogram_perimeter_l3666_366668


namespace fish_length_theorem_l3666_366629

theorem fish_length_theorem (x : ℚ) :
  (1 / 3 : ℚ) * x + (1 / 4 : ℚ) * x + 3 = x → x = 36 / 5 := by
  sorry

end fish_length_theorem_l3666_366629


namespace cubic_root_product_l3666_366636

theorem cubic_root_product (x : ℝ) : 
  (∃ p q r : ℝ, x^3 - 15*x^2 + 60*x - 36 = (x - p) * (x - q) * (x - r)) → 
  (∃ p q r : ℝ, x^3 - 15*x^2 + 60*x - 36 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 36) :=
by sorry

end cubic_root_product_l3666_366636


namespace max_value_of_expression_l3666_366611

theorem max_value_of_expression (x y z : Real) 
  (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) (hz : 0 < z ∧ z ≤ 1) :
  let A := (Real.sqrt (8 * x^4 + y) + Real.sqrt (8 * y^4 + z) + Real.sqrt (8 * z^4 + x) - 3) / (x + y + z)
  A ≤ 2 ∧ ∃ x y z, (0 < x ∧ x ≤ 1) ∧ (0 < y ∧ y ≤ 1) ∧ (0 < z ∧ z ≤ 1) ∧ A = 2 :=
by sorry

end max_value_of_expression_l3666_366611


namespace empty_set_implies_m_zero_l3666_366654

theorem empty_set_implies_m_zero (m : ℝ) : (∀ x : ℝ, m * x ≠ 1) → m = 0 := by
  sorry

end empty_set_implies_m_zero_l3666_366654


namespace quadratic_function_uniqueness_l3666_366698

-- Define a quadratic function
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x

-- State the theorem
theorem quadratic_function_uniqueness (g : ℝ → ℝ) :
  (∃ a b : ℝ, g = QuadraticFunction a b) →  -- g is quadratic
  g 1 = 1 →                                 -- g(1) = 1
  g (-1) = 5 →                              -- g(-1) = 5
  g 0 = 0 →                                 -- g(0) = 0 (passes through origin)
  g = QuadraticFunction 3 (-2) :=           -- g(x) = 3x^2 - 2x
by
  sorry


end quadratic_function_uniqueness_l3666_366698


namespace binary_to_quaternary_conversion_l3666_366639

/-- Represents a number in a given base --/
structure BaseNumber where
  digits : List Nat
  base : Nat

/-- Converts a base 2 number to base 10 --/
def binaryToDecimal (bn : BaseNumber) : Nat :=
  bn.digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 2^i) 0

/-- Converts a base 10 number to base 4 --/
def decimalToQuaternary (n : Nat) : BaseNumber :=
  let rec toDigits (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else toDigits (m / 4) ((m % 4) :: acc)
  { digits := toDigits n [], base := 4 }

/-- The main theorem --/
theorem binary_to_quaternary_conversion :
  let binary := BaseNumber.mk [1,0,1,1,0,0,1,0,1] 2
  let quaternary := BaseNumber.mk [2,3,0,1,1] 4
  decimalToQuaternary (binaryToDecimal binary) = quaternary := by
  sorry

end binary_to_quaternary_conversion_l3666_366639


namespace common_remainder_l3666_366631

theorem common_remainder : ∃ r : ℕ, 
  r < 9 ∧ r < 11 ∧ r < 17 ∧
  (3374 % 9 = r) ∧ (3374 % 11 = r) ∧ (3374 % 17 = r) ∧
  r = 8 := by
  sorry

end common_remainder_l3666_366631


namespace combined_height_of_tamara_and_kim_l3666_366608

/-- Given Tamara's height is 3 times Kim's height less 4 inches and Tamara is 68 inches tall,
    prove that the combined height of Tamara and Kim is 92 inches. -/
theorem combined_height_of_tamara_and_kim (kim_height : ℕ) : 
  (3 * kim_height - 4 = 68) → (68 + kim_height = 92) := by
  sorry

end combined_height_of_tamara_and_kim_l3666_366608


namespace no_function_satisfies_condition_l3666_366662

theorem no_function_satisfies_condition : 
  ¬∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f (x + f y) = f x + Real.sin y := by
  sorry

end no_function_satisfies_condition_l3666_366662


namespace lizzys_final_money_l3666_366699

/-- Calculates the final amount of money Lizzy has after a series of transactions -/
def lizzys_money (
  mother_gave : ℕ)
  (father_gave : ℕ)
  (candy_cost : ℕ)
  (uncle_gave : ℕ)
  (toy_price : ℕ)
  (discount_percent : ℕ)
  (change_dollars : ℕ)
  (change_cents : ℕ) : ℕ :=
  let initial := mother_gave + father_gave
  let after_candy := initial - candy_cost + uncle_gave
  let discounted_price := toy_price - (toy_price * discount_percent / 100)
  let after_toy := after_candy - discounted_price
  let final := after_toy + change_dollars * 100 + change_cents
  final

theorem lizzys_final_money :
  lizzys_money 80 40 50 70 90 20 1 10 = 178 := by
  sorry

end lizzys_final_money_l3666_366699


namespace halloween_candy_theorem_l3666_366680

/-- The number of candy pieces left after combining and eating some. -/
def candy_left (katie_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  katie_candy + sister_candy - eaten_candy

/-- Theorem stating the number of candy pieces left in the given scenario. -/
theorem halloween_candy_theorem :
  candy_left 8 23 8 = 23 := by
  sorry

end halloween_candy_theorem_l3666_366680


namespace quadratic_equation_coefficients_l3666_366644

/-- A quadratic equation with reciprocal roots whose sum is four times their product -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  roots_reciprocal : ∃ (r : ℝ), r ≠ 0 ∧ r + 1/r = -b/a
  sum_four_times_product : -b/a = 4 * (c/a)

/-- The coefficients of the quadratic equation satisfy a = c and b = -4a -/
theorem quadratic_equation_coefficients (eq : QuadraticEquation) : eq.a = eq.c ∧ eq.b = -4 * eq.a := by
  sorry

end quadratic_equation_coefficients_l3666_366644


namespace valentines_day_cards_l3666_366677

theorem valentines_day_cards (boys girls : ℕ) : 
  boys * girls = boys + girls + 18 → boys * girls = 40 := by
  sorry

end valentines_day_cards_l3666_366677


namespace line_angle_and_triangle_conditions_l3666_366687

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def l₁ : Line := { a := 2, b := -1, c := -10 }
def l₂ : Line := { a := 4, b := 3, c := -10 }
def l₃ (a : ℝ) : Line := { a := a, b := 2, c := -8 }

/-- The angle between two lines -/
def angle_between (l1 l2 : Line) : ℝ := sorry

/-- Whether three lines can form a triangle -/
def can_form_triangle (l1 l2 l3 : Line) : Prop := sorry

theorem line_angle_and_triangle_conditions :
  (angle_between l₁ l₂ = Real.arctan 2) ∧
  (∀ a : ℝ, ¬(can_form_triangle l₁ l₂ (l₃ a)) ↔ (a = -4 ∨ a = 8/3 ∨ a = 3)) := by sorry

end line_angle_and_triangle_conditions_l3666_366687


namespace lagrange_mvt_example_l3666_366614

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 6*x + 1

-- State the theorem
theorem lagrange_mvt_example :
  ∃ c ∈ Set.Ioo (-1 : ℝ) 3,
    (f 3 - f (-1)) / (3 - (-1)) = 2*c + 6 :=
by
  sorry


end lagrange_mvt_example_l3666_366614


namespace absolute_value_fraction_l3666_366620

theorem absolute_value_fraction (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 10*a*b) :
  |((a - b) / (a + b))| = Real.sqrt (2/3) := by
  sorry

end absolute_value_fraction_l3666_366620


namespace range_of_a_l3666_366600

theorem range_of_a (x a : ℝ) : 
  (∀ x, (x^2 + 2*x - 3 > 0 → x > a) ∧ 
   (x^2 + 2*x - 3 ≤ 0 → x ≤ a) ∧ 
   ∃ x, x^2 + 2*x - 3 > 0 ∧ x > a) →
  a ≥ 1 :=
by sorry

end range_of_a_l3666_366600


namespace seedling_ratio_l3666_366679

theorem seedling_ratio (first_day : ℕ) (total : ℕ) : 
  first_day = 200 → total = 1200 → 
  (total - first_day) / first_day = 5 := by
  sorry

end seedling_ratio_l3666_366679


namespace total_cost_of_flowers_l3666_366689

/-- The cost of a single flower in dollars -/
def flower_cost : ℕ := 3

/-- The number of roses bought -/
def roses_bought : ℕ := 2

/-- The number of daisies bought -/
def daisies_bought : ℕ := 2

/-- The total number of flowers bought -/
def total_flowers : ℕ := roses_bought + daisies_bought

/-- The theorem stating the total cost of the flowers -/
theorem total_cost_of_flowers : total_flowers * flower_cost = 12 := by
  sorry

end total_cost_of_flowers_l3666_366689


namespace initial_girls_count_l3666_366606

theorem initial_girls_count (b g : ℚ) : 
  (3 * (g - 20) = b) →
  (6 * (b - 60) = g - 20) →
  g = 700 / 17 := by
  sorry

end initial_girls_count_l3666_366606


namespace choose_marbles_eq_990_l3666_366625

/-- The number of ways to choose 5 marbles out of 15, where exactly 2 are chosen from a set of 4 special marbles -/
def choose_marbles : ℕ :=
  let total_marbles : ℕ := 15
  let special_marbles : ℕ := 4
  let choose_total : ℕ := 5
  let choose_special : ℕ := 2
  let normal_marbles : ℕ := total_marbles - special_marbles
  let choose_normal : ℕ := choose_total - choose_special
  (Nat.choose special_marbles choose_special) * (Nat.choose normal_marbles choose_normal)

theorem choose_marbles_eq_990 : choose_marbles = 990 := by
  sorry

end choose_marbles_eq_990_l3666_366625


namespace trees_planted_per_cut_l3666_366669

/-- Proves that the number of new trees planted for each tree cut is 5 --/
theorem trees_planted_per_cut (initial_trees : ℕ) (cut_percentage : ℚ) (final_trees : ℕ) : 
  initial_trees = 400 → 
  cut_percentage = 1/5 →
  final_trees = 720 →
  (final_trees - (initial_trees - initial_trees * cut_percentage)) / (initial_trees * cut_percentage) = 5 := by
  sorry

end trees_planted_per_cut_l3666_366669


namespace hyperbola_k_range_l3666_366691

-- Define the hyperbola equation
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (k - 3) + y^2 / (2 - k) = 1

-- Define the condition for foci on y-axis
def foci_on_y_axis (k : ℝ) : Prop :=
  2 - k > 0 ∧ k - 3 < 0

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, hyperbola_equation x y k) ∧ foci_on_y_axis k → k < 2 :=
by sorry

end hyperbola_k_range_l3666_366691

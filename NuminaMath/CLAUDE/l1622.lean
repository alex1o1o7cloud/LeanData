import Mathlib

namespace NUMINAMATH_CALUDE_sum_a_b_equals_negative_two_l1622_162273

theorem sum_a_b_equals_negative_two (a b : ℝ) :
  (a - 2)^2 + |b + 4| = 0 → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_negative_two_l1622_162273


namespace NUMINAMATH_CALUDE_penguins_fed_correct_l1622_162220

/-- The number of penguins that have already gotten a fish -/
def penguins_fed (total_penguins : ℕ) (penguins_to_feed : ℕ) : ℕ :=
  total_penguins - penguins_to_feed

theorem penguins_fed_correct (total_fish : ℕ) (total_penguins : ℕ) (penguins_to_feed : ℕ) :
  total_fish = 68 →
  total_penguins = 36 →
  penguins_to_feed = 17 →
  penguins_fed total_penguins penguins_to_feed = 19 :=
by
  sorry

#eval penguins_fed 36 17

end NUMINAMATH_CALUDE_penguins_fed_correct_l1622_162220


namespace NUMINAMATH_CALUDE_ratio_of_sums_l1622_162282

/-- Represents an arithmetic progression --/
structure ArithmeticProgression where
  firstTerm : ℕ
  difference : ℕ
  length : ℕ

/-- Calculates the sum of an arithmetic progression --/
def sumOfArithmeticProgression (ap : ArithmeticProgression) : ℕ :=
  ap.length * (2 * ap.firstTerm + (ap.length - 1) * ap.difference) / 2

/-- Generates a list of arithmetic progressions for the first group --/
def firstGroup : List ArithmeticProgression :=
  List.range 15
    |> List.map (fun i => ArithmeticProgression.mk (i + 1) (2 * (i + 1)) 10)

/-- Generates a list of arithmetic progressions for the second group --/
def secondGroup : List ArithmeticProgression :=
  List.range 15
    |> List.map (fun i => ArithmeticProgression.mk (i + 1) (2 * i + 1) 10)

/-- Calculates the sum of all elements in a group of arithmetic progressions --/
def sumOfGroup (group : List ArithmeticProgression) : ℕ :=
  group.map sumOfArithmeticProgression |> List.sum

theorem ratio_of_sums : 
  (sumOfGroup firstGroup : ℚ) / (sumOfGroup secondGroup : ℚ) = 160 / 151 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sums_l1622_162282


namespace NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_C_l1622_162229

theorem correct_calculation : 2 * Real.sqrt 3 - Real.sqrt 3 = Real.sqrt 3 :=
by sorry

theorem incorrect_calculation_A : ¬(Real.sqrt 3 + Real.sqrt 2 = Real.sqrt 5) :=
by sorry

theorem incorrect_calculation_B : ¬(Real.sqrt 3 * Real.sqrt 5 = 15) :=
by sorry

theorem incorrect_calculation_C : ¬(Real.sqrt 32 / Real.sqrt 8 = 2 ∨ Real.sqrt 32 / Real.sqrt 8 = -2) :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_C_l1622_162229


namespace NUMINAMATH_CALUDE_certain_number_proof_l1622_162261

theorem certain_number_proof (x : ℕ+) (h : (55 * x.val) % 7 = 6) : x.val % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1622_162261


namespace NUMINAMATH_CALUDE_savings_from_discount_l1622_162232

def initial_price : ℚ := 475
def discounted_price : ℚ := 199

theorem savings_from_discount :
  initial_price - discounted_price = 276 := by
  sorry

end NUMINAMATH_CALUDE_savings_from_discount_l1622_162232


namespace NUMINAMATH_CALUDE_quadratic_equation_one_solution_positive_n_value_l1622_162265

theorem quadratic_equation_one_solution (n : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + n * x + 16 = 0) → n = 16 ∨ n = -16 :=
by sorry

theorem positive_n_value (n : ℝ) :
  (∃! x : ℝ, 4 * x^2 + n * x + 16 = 0) ∧ n > 0 → n = 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_solution_positive_n_value_l1622_162265


namespace NUMINAMATH_CALUDE_theater_ticket_income_theater_income_proof_l1622_162251

/-- Calculate the total ticket income for a theater -/
theorem theater_ticket_income 
  (total_seats : ℕ) 
  (adult_price child_price : ℚ) 
  (children_count : ℕ) : ℚ :=
  let adult_count : ℕ := total_seats - children_count
  let adult_income : ℚ := adult_count * adult_price
  let child_income : ℚ := children_count * child_price
  adult_income + child_income

/-- Prove that the total ticket income for the given theater scenario is $510.00 -/
theorem theater_income_proof :
  theater_ticket_income 200 3 (3/2) 60 = 510 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_income_theater_income_proof_l1622_162251


namespace NUMINAMATH_CALUDE_original_group_size_l1622_162238

theorem original_group_size (n : ℕ) (W : ℝ) : 
  W = n * 35 →
  W + 40 = (n + 1) * 36 →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_original_group_size_l1622_162238


namespace NUMINAMATH_CALUDE_special_triangle_property_l1622_162206

noncomputable section

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the angles of the triangle
def angle_A (t : Triangle) : ℝ := Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))
def angle_B (t : Triangle) : ℝ := Real.arccos ((t.c^2 + t.a^2 - t.b^2) / (2 * t.c * t.a))
def angle_C (t : Triangle) : ℝ := Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

-- The main theorem
theorem special_triangle_property (t : Triangle) 
  (h : t.b * (t.a + t.b) * (t.b + t.c) = t.a^3 + t.b * (t.a^2 + t.c^2) + t.c^3) :
  1 / (Real.sqrt (angle_A t) + Real.sqrt (angle_B t)) + 
  1 / (Real.sqrt (angle_B t) + Real.sqrt (angle_C t)) = 
  2 / (Real.sqrt (angle_C t) + Real.sqrt (angle_A t)) :=
sorry

end

end NUMINAMATH_CALUDE_special_triangle_property_l1622_162206


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1622_162201

/-- Represents a geometric sequence with common ratio greater than 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : q > 1
  h_geometric : ∀ n, a (n + 1) = q * a n

/-- Given conditions for the geometric sequence -/
def SequenceConditions (seq : GeometricSequence) : Prop :=
  seq.a 3 * seq.a 7 = 72 ∧ seq.a 2 + seq.a 8 = 27

theorem geometric_sequence_problem (seq : GeometricSequence) 
  (h : SequenceConditions seq) : seq.a 12 = 96 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1622_162201


namespace NUMINAMATH_CALUDE_amanda_peaches_difference_l1622_162249

/-- The number of peaches each person has -/
structure Peaches where
  jill : ℕ
  steven : ℕ
  jake : ℕ
  amanda : ℕ

/-- The conditions of the problem -/
def peach_conditions (p : Peaches) : Prop :=
  p.jill = 12 ∧
  p.steven = p.jill + 15 ∧
  p.jake = p.steven - 16 ∧
  p.amanda = 2 * p.jill

/-- The average number of peaches Jake, Steven, and Jill have -/
def average (p : Peaches) : ℚ :=
  (p.jake + p.steven + p.jill : ℚ) / 3

/-- The theorem to be proved -/
theorem amanda_peaches_difference (p : Peaches) (h : peach_conditions p) :
  p.amanda - average p = 7.33 := by
  sorry

end NUMINAMATH_CALUDE_amanda_peaches_difference_l1622_162249


namespace NUMINAMATH_CALUDE_equation_proof_l1622_162242

theorem equation_proof : (8/3 + 3/2) / (15/4) - 0.4 = 32/45 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1622_162242


namespace NUMINAMATH_CALUDE_serenity_new_shoes_l1622_162240

/-- The number of pairs of shoes Serenity bought -/
def pairs_bought : ℕ := 3

/-- The number of shoes in each pair -/
def shoes_per_pair : ℕ := 2

/-- The total number of new shoes Serenity has now -/
def total_new_shoes : ℕ := pairs_bought * shoes_per_pair

theorem serenity_new_shoes : total_new_shoes = 6 := by sorry

end NUMINAMATH_CALUDE_serenity_new_shoes_l1622_162240


namespace NUMINAMATH_CALUDE_fashion_show_duration_l1622_162287

def fashion_show_time (num_models : ℕ) (bathing_suits_per_model : ℕ) (evening_wear_per_model : ℕ) (time_per_trip : ℕ) : ℕ :=
  num_models * (bathing_suits_per_model + evening_wear_per_model) * time_per_trip

theorem fashion_show_duration :
  fashion_show_time 6 2 3 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_fashion_show_duration_l1622_162287


namespace NUMINAMATH_CALUDE_new_team_average_l1622_162241

/-- Represents a cricket team with ages and average calculations. -/
structure CricketTeam where
  initialMembers : Nat
  captainAge : Nat
  wicketKeeperAge : Nat
  initialAverage : ℚ
  ageDecrease : ℚ

/-- Calculates the new average age of the team after changes. -/
def newAverageAge (team : CricketTeam) : ℚ :=
  team.initialAverage - team.ageDecrease

/-- Theorem stating the new average age of the team after changes. -/
theorem new_team_average (team : CricketTeam) 
  (h1 : team.initialMembers = 11)
  (h2 : team.captainAge = 24)
  (h3 : team.wicketKeeperAge = 31)
  (h4 : (team.initialAverage * team.initialMembers - team.captainAge - team.wicketKeeperAge) / (team.initialMembers - 2) = team.initialAverage - 1)
  (h5 : team.ageDecrease = 1/2) :
  newAverageAge team = 45/2 := by
  sorry

#eval newAverageAge { initialMembers := 11, captainAge := 24, wicketKeeperAge := 31, initialAverage := 23, ageDecrease := 1/2 }

end NUMINAMATH_CALUDE_new_team_average_l1622_162241


namespace NUMINAMATH_CALUDE_major_axis_length_l1622_162205

/-- Represents an ellipse formed by the intersection of a plane and a right circular cylinder. -/
structure IntersectionEllipse where
  cylinder_radius : ℝ
  major_axis : ℝ
  minor_axis : ℝ

/-- The theorem stating the length of the major axis given the conditions. -/
theorem major_axis_length 
  (e : IntersectionEllipse) 
  (h1 : e.cylinder_radius = 2)
  (h2 : e.minor_axis = 2 * e.cylinder_radius)
  (h3 : e.major_axis = e.minor_axis * (1 + 0.75)) :
  e.major_axis = 7 := by
  sorry


end NUMINAMATH_CALUDE_major_axis_length_l1622_162205


namespace NUMINAMATH_CALUDE_product_equals_32_l1622_162222

theorem product_equals_32 : 
  (1 / 2) * 4 * (1 / 8) * 16 * (1 / 32) * 64 * (1 / 128) * 256 * (1 / 512) * 1024 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_32_l1622_162222


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l1622_162236

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C*(x + y + z)) ↔ C ≤ 2 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l1622_162236


namespace NUMINAMATH_CALUDE_set_equality_implies_values_l1622_162290

theorem set_equality_implies_values (x y : ℝ) : 
  ({x, y^2, 1} : Set ℝ) = ({1, 2*x, y} : Set ℝ) → x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_values_l1622_162290


namespace NUMINAMATH_CALUDE_gcd_3150_9800_l1622_162221

theorem gcd_3150_9800 : Nat.gcd 3150 9800 = 350 := by sorry

end NUMINAMATH_CALUDE_gcd_3150_9800_l1622_162221


namespace NUMINAMATH_CALUDE_identify_counterfeit_coins_l1622_162253

/-- Represents a coin which can be either real or counterfeit -/
inductive Coin
| Real
| CounterfeitLight
| CounterfeitHeavy

/-- Represents the result of a weighing -/
inductive WeighingResult
| LeftHeavier
| RightHeavier
| Equal

/-- Represents a set of five coins -/
def CoinSet := Fin 5 → Coin

/-- Represents a weighing operation on the balance scale -/
def Weighing := List Nat → List Nat → WeighingResult

/-- The main theorem stating that it's possible to identify counterfeit coins in three weighings -/
theorem identify_counterfeit_coins 
  (coins : CoinSet) 
  (h1 : ∃ (i j : Fin 5), i ≠ j ∧ coins i = Coin.CounterfeitLight ∧ coins j = Coin.CounterfeitHeavy) 
  (h2 : ∀ (i : Fin 5), coins i ≠ Coin.CounterfeitLight → coins i ≠ Coin.CounterfeitHeavy → coins i = Coin.Real) :
  ∃ (w1 w2 w3 : Weighing), 
    ∀ (i j : Fin 5), 
      (coins i = Coin.CounterfeitLight ∧ coins j = Coin.CounterfeitHeavy) → 
      ∃ (f : Weighing → Weighing → Weighing → Fin 5 × Fin 5), 
        f w1 w2 w3 = (i, j) :=
sorry

end NUMINAMATH_CALUDE_identify_counterfeit_coins_l1622_162253


namespace NUMINAMATH_CALUDE_thirtieth_day_production_l1622_162237

/-- Represents the daily cloth production in feet -/
def cloth_sequence (n : ℕ) : ℚ :=
  5 + (n - 1) * ((390 - 30 * 5) / (30 * 29 / 2))

/-- The sum of the cloth_sequence for the first 30 days -/
def total_cloth : ℚ := 390

/-- The theorem states that the 30th term of the cloth_sequence is 21 -/
theorem thirtieth_day_production : cloth_sequence 30 = 21 := by sorry

end NUMINAMATH_CALUDE_thirtieth_day_production_l1622_162237


namespace NUMINAMATH_CALUDE_not_square_among_powers_l1622_162292

theorem not_square_among_powers : 
  (∃ n : ℕ, 1^6 = n^2) ∧
  (∃ n : ℕ, 3^4 = n^2) ∧
  (∃ n : ℕ, 4^3 = n^2) ∧
  (∃ n : ℕ, 5^2 = n^2) ∧
  (¬ ∃ n : ℕ, 2^5 = n^2) := by
  sorry

end NUMINAMATH_CALUDE_not_square_among_powers_l1622_162292


namespace NUMINAMATH_CALUDE_transaction_difference_prove_transaction_difference_l1622_162269

theorem transaction_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun mabel_transactions anthony_transactions cal_transactions jade_transactions =>
    mabel_transactions = 90 →
    anthony_transactions = mabel_transactions + mabel_transactions / 10 →
    cal_transactions = anthony_transactions * 2 / 3 →
    jade_transactions = 80 →
    jade_transactions - cal_transactions = 14

#check transaction_difference

theorem prove_transaction_difference :
  ∃ (mabel anthony cal jade : ℕ),
    transaction_difference mabel anthony cal jade :=
by
  sorry

end NUMINAMATH_CALUDE_transaction_difference_prove_transaction_difference_l1622_162269


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l1622_162298

theorem inequality_system_solution_range (k : ℝ) : 
  (∀ x : ℤ, (x^2 - x - 2 > 0 ∧ 2*x^2 + (2*k+5)*x + 5*k < 0) ↔ x = -2) →
  -3 ≤ k ∧ k < 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l1622_162298


namespace NUMINAMATH_CALUDE_equation_solutions_l1622_162255

theorem equation_solutions :
  (∀ x : ℝ, (x - 1) * (x + 3) = x - 1 ↔ x = 1 ∨ x = -2) ∧
  (∀ x : ℝ, 2 * x^2 - 6 * x = -3 ↔ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1622_162255


namespace NUMINAMATH_CALUDE_remainder_7n_mod_5_l1622_162243

theorem remainder_7n_mod_5 (n : ℤ) (h : n % 5 = 3) : (7 * n) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_5_l1622_162243


namespace NUMINAMATH_CALUDE_equation_solution_l1622_162278

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - y^2 - 4.5 = 0) ↔ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1622_162278


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_7_and_4_l1622_162285

theorem smallest_common_multiple_of_7_and_4 : ∃ (n : ℕ), n > 0 ∧ n % 7 = 0 ∧ n % 4 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 7 = 0 ∧ m % 4 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_7_and_4_l1622_162285


namespace NUMINAMATH_CALUDE_point_P_coordinates_l1622_162281

/-- The mapping f that transforms a point (x, y) to (x+y, 2x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, 2 * p.1 - p.2)

/-- Theorem stating that if P is mapped to (5, 1) under f, then P has coordinates (2, 3) -/
theorem point_P_coordinates (P : ℝ × ℝ) (h : f P = (5, 1)) : P = (2, 3) := by
  sorry


end NUMINAMATH_CALUDE_point_P_coordinates_l1622_162281


namespace NUMINAMATH_CALUDE_M_subset_N_l1622_162275

def M : Set ℝ := {-1, 1}
def N : Set ℝ := {x | (1 / x) < 2}

theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l1622_162275


namespace NUMINAMATH_CALUDE_odd_integers_divisibility_l1622_162296

theorem odd_integers_divisibility (a b : ℕ) : 
  Odd a → Odd b → a > 0 → b > 0 → (2 * a * b + 1) ∣ (a^2 + b^2 + 1) → a = b := by
  sorry

end NUMINAMATH_CALUDE_odd_integers_divisibility_l1622_162296


namespace NUMINAMATH_CALUDE_simplify_fraction_l1622_162216

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) : 
  (2 / (x^2 - 2*x + 1) - 1 / (x^2 - x)) / ((x + 1) / (2*x^2 - 2*x)) = 2 / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1622_162216


namespace NUMINAMATH_CALUDE_dave_lisa_slices_l1622_162209

/-- Represents the number of slices in a pizza -/
structure Pizza where
  small : ℕ
  large : ℕ

/-- Represents the number of pizzas purchased -/
structure PizzaOrder where
  small : ℕ
  large : ℕ

/-- Represents the number of slices eaten by each person -/
structure SlicesEaten where
  george : ℕ
  bob : ℕ
  susie : ℕ
  bill : ℕ
  fred : ℕ
  mark : ℕ
  ann : ℕ
  kelly : ℕ

def pizza_sizes : Pizza := ⟨4, 8⟩
def george_order : PizzaOrder := ⟨4, 3⟩
def slices_eaten : SlicesEaten := ⟨3, 4, 2, 3, 3, 3, 2, 4⟩

def total_slices (p : Pizza) (o : PizzaOrder) : ℕ :=
  p.small * o.small + p.large * o.large

def total_eaten (s : SlicesEaten) : ℕ :=
  s.george + s.bob + s.susie + s.bill + s.fred + s.mark + s.ann + s.kelly

theorem dave_lisa_slices :
  (total_slices pizza_sizes george_order - total_eaten slices_eaten) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_dave_lisa_slices_l1622_162209


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1622_162224

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

/-- Theorem: If the nth term of the arithmetic sequence is 2014, then n is 672 -/
theorem arithmetic_sequence_nth_term (n : ℕ) :
  arithmetic_sequence n = 2014 → n = 672 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1622_162224


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1622_162263

theorem quadratic_equation_roots : 
  let f : ℝ → ℝ := λ x => x^2 - 1 - 3
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1622_162263


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1622_162231

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_5 + a_8 = 24, then a_6 + a_7 = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 5 + a 8 = 24) : 
  a 6 + a 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1622_162231


namespace NUMINAMATH_CALUDE_point_C_coordinates_l1622_162299

-- Define points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 5)

-- Define vector AB
def vecAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define vector AC in terms of AB
def vecAC : ℝ × ℝ := (2 * vecAB.1, 2 * vecAB.2)

-- Define point C
def C : ℝ × ℝ := (A.1 + vecAC.1, A.2 + vecAC.2)

-- Theorem to prove
theorem point_C_coordinates : C = (-3, 9) := by
  sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l1622_162299


namespace NUMINAMATH_CALUDE_unique_a_for_equal_roots_l1622_162202

theorem unique_a_for_equal_roots :
  ∃! a : ℝ, ∀ x : ℝ, x^2 - (a + 1) * x + a = 0 → (∃! y : ℝ, y^2 - (a + 1) * y + a = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_a_for_equal_roots_l1622_162202


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l1622_162271

/-- The equation of a circle symmetric to another circle with respect to the line y = x -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∃ (a b r : ℝ), (x + a)^2 + (y - b)^2 = r^2) →  -- Original circle
  (∃ (c d : ℝ), (x - c)^2 + (y + d)^2 = 5) :=     -- Symmetric circle
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l1622_162271


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l1622_162291

theorem algebraic_expression_simplification (x : ℝ) :
  x = 2 * Real.cos (45 * π / 180) - 1 →
  (x / (x^2 + 2*x + 1) - 1 / (2*x + 2)) / ((x - 1) / (4*x + 4)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l1622_162291


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1622_162219

theorem fraction_subtraction : 
  (2 + 6 + 8) / (1 + 2 + 3) - (1 + 2 + 3) / (2 + 6 + 8) = 55 / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1622_162219


namespace NUMINAMATH_CALUDE_sophia_stamp_collection_value_l1622_162257

/-- Given a collection of stamps with equal value, calculate the total value. -/
def stamp_collection_value (total_stamps : ℕ) (sample_stamps : ℕ) (sample_value : ℕ) : ℕ :=
  total_stamps * (sample_value / sample_stamps)

/-- Theorem: Sophia's stamp collection is worth 120 dollars. -/
theorem sophia_stamp_collection_value :
  stamp_collection_value 24 8 40 = 120 := by
  sorry

#eval stamp_collection_value 24 8 40

end NUMINAMATH_CALUDE_sophia_stamp_collection_value_l1622_162257


namespace NUMINAMATH_CALUDE_unique_solution_l1622_162288

/-- Represents a three-digit number (abc) -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9
  h3 : c ≥ 0 ∧ c ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of the five rearranged numbers plus the original number -/
def N : Nat := 3306

/-- The equation that needs to be satisfied -/
def satisfiesEquation (n : ThreeDigitNumber) : Prop :=
  N + n.toNat = 222 * (n.a + n.b + n.c)

theorem unique_solution :
  ∃! n : ThreeDigitNumber, satisfiesEquation n ∧ n.toNat = 753 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1622_162288


namespace NUMINAMATH_CALUDE_unique_excellent_beats_all_l1622_162272

-- Define the type for players
variable {Player : Type}

-- Define the relation for "beats"
variable (beats : Player → Player → Prop)

-- Define what it means to be an excellent player
def is_excellent (A : Player) : Prop :=
  ∀ B : Player, B ≠ A → (beats A B ∨ ∃ C : Player, beats C B ∧ beats A C)

-- State the theorem
theorem unique_excellent_beats_all
  (players : Set Player)
  (h_nonempty : Set.Nonempty players)
  (h_no_self_play : ∀ A : Player, ¬beats A A)
  (h_all_play : ∀ A B : Player, A ∈ players → B ∈ players → A ≠ B → (beats A B ∨ beats B A))
  (h_unique_excellent : ∃! A : Player, A ∈ players ∧ is_excellent beats A) :
  ∃ A : Player, A ∈ players ∧ is_excellent beats A ∧ ∀ B : Player, B ∈ players → B ≠ A → beats A B :=
sorry

end NUMINAMATH_CALUDE_unique_excellent_beats_all_l1622_162272


namespace NUMINAMATH_CALUDE_eva_last_when_start_vasya_l1622_162223

/-- Represents the children in the circle -/
inductive Child : Type
| Anya : Child
| Borya : Child
| Vasya : Child
| Gena : Child
| Dasha : Child
| Eva : Child
| Zhenya : Child

/-- The number of children in the circle -/
def num_children : Nat := 7

/-- The step size for elimination -/
def step_size : Nat := 3

/-- Function to determine the last remaining child given a starting position -/
def last_remaining (start : Child) : Child :=
  sorry

/-- Theorem stating that starting from Vasya results in Eva being the last remaining -/
theorem eva_last_when_start_vasya :
  last_remaining Child.Vasya = Child.Eva :=
sorry

end NUMINAMATH_CALUDE_eva_last_when_start_vasya_l1622_162223


namespace NUMINAMATH_CALUDE_sally_seashells_l1622_162233

theorem sally_seashells (total tom jessica : ℕ) (h1 : total = 21) (h2 : tom = 7) (h3 : jessica = 5) :
  total - tom - jessica = 9 := by
  sorry

end NUMINAMATH_CALUDE_sally_seashells_l1622_162233


namespace NUMINAMATH_CALUDE_current_rate_calculation_l1622_162246

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  downstream : ℝ
  upstream : ℝ
  stillWater : ℝ

/-- Calculates the rate of the current given rowing speeds -/
def currentRate (speed : RowingSpeed) : ℝ :=
  speed.downstream - speed.stillWater

theorem current_rate_calculation (speed : RowingSpeed) 
  (h1 : speed.downstream = 24)
  (h2 : speed.upstream = 7)
  (h3 : speed.stillWater = 15.5) :
  currentRate speed = 8.5 := by
  sorry

#eval currentRate { downstream := 24, upstream := 7, stillWater := 15.5 }

end NUMINAMATH_CALUDE_current_rate_calculation_l1622_162246


namespace NUMINAMATH_CALUDE_balloons_bought_at_park_l1622_162284

theorem balloons_bought_at_park (allan_initial : ℕ) (jake_initial : ℕ) (jake_bought : ℕ) :
  allan_initial = 6 →
  jake_initial = 2 →
  allan_initial = jake_initial + jake_bought + 1 →
  jake_bought = 3 := by
sorry

end NUMINAMATH_CALUDE_balloons_bought_at_park_l1622_162284


namespace NUMINAMATH_CALUDE_sequence_sum_bound_l1622_162294

theorem sequence_sum_bound (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n : ℕ, a n > 0) →
  (∀ n : ℕ, S n^2 - (n^2 + n - 1) * S n - (n^2 + n) = 0) →
  (∀ n : ℕ, b n = (n + 1) / ((n + 2)^2 * (a n)^2)) →
  (∀ n : ℕ, T (n + 1) = T n + b (n + 1)) →
  T 0 = 0 →
  ∀ n : ℕ, 0 < n → T n < 5/64 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_bound_l1622_162294


namespace NUMINAMATH_CALUDE_fraction_reduction_l1622_162256

theorem fraction_reduction (n : ℤ) :
  (∃ k : ℤ, n = 7 * k + 1) ↔
  (∃ m : ℤ, 4 * n + 3 = 7 * m) ∧ (∃ l : ℤ, 5 * n + 2 = 7 * l) :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l1622_162256


namespace NUMINAMATH_CALUDE_bubble_radius_l1622_162207

/-- The radius of a sphere with volume equal to the sum of volumes of a hemisphere and a cylinder --/
theorem bubble_radius (hemisphere_radius cylinder_radius cylinder_height : ℝ) 
  (hr : hemisphere_radius = 5)
  (hcr : cylinder_radius = 2)
  (hch : cylinder_height = hemisphere_radius) : 
  ∃ R : ℝ, R^3 = 77.5 ∧ 
  (4/3 * Real.pi * R^3 = 2/3 * Real.pi * hemisphere_radius^3 + Real.pi * cylinder_radius^2 * cylinder_height) :=
by sorry

end NUMINAMATH_CALUDE_bubble_radius_l1622_162207


namespace NUMINAMATH_CALUDE_tv_price_reduction_l1622_162252

/-- Proves that a price reduction resulting in a 75% increase in sales and a 31.25% increase in total sale value implies a 25% price reduction -/
theorem tv_price_reduction (P : ℝ) (N : ℝ) (x : ℝ) 
  (h1 : P > 0) 
  (h2 : N > 0) 
  (h3 : x > 0) 
  (h4 : x < 100) 
  (h5 : P * (1 - x / 100) * (N * 1.75) = P * N * 1.3125) : 
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_tv_price_reduction_l1622_162252


namespace NUMINAMATH_CALUDE_female_students_count_l1622_162211

theorem female_students_count (total_average : ℚ) (male_count : ℕ) (male_average : ℚ) (female_average : ℚ) :
  total_average = 90 →
  male_count = 8 →
  male_average = 87 →
  female_average = 92 →
  ∃ (female_count : ℕ), 
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 12 := by
  sorry

end NUMINAMATH_CALUDE_female_students_count_l1622_162211


namespace NUMINAMATH_CALUDE_pizzas_served_today_l1622_162279

/-- The number of pizzas served during lunch -/
def lunch_pizzas : ℕ := 9

/-- The number of pizzas served during dinner -/
def dinner_pizzas : ℕ := 6

/-- The total number of pizzas served today -/
def total_pizzas : ℕ := lunch_pizzas + dinner_pizzas

theorem pizzas_served_today : total_pizzas = 15 := by
  sorry

end NUMINAMATH_CALUDE_pizzas_served_today_l1622_162279


namespace NUMINAMATH_CALUDE_ball_max_height_l1622_162262

def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 161 :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l1622_162262


namespace NUMINAMATH_CALUDE_problem_solution_l1622_162210

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 3)
  (h_eq2 : y + 1 / x = 31) : 
  z + 1 / y = 9 / 23 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1622_162210


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l1622_162293

/-- A right circular cone with an inscribed right circular cylinder -/
structure ConeWithCylinder where
  -- Cone properties
  cone_diameter : ℝ
  cone_altitude : ℝ
  -- Cylinder properties
  cylinder_radius : ℝ
  -- Conditions
  cone_diameter_positive : 0 < cone_diameter
  cone_altitude_positive : 0 < cone_altitude
  cylinder_radius_positive : 0 < cylinder_radius
  cylinder_inscribed : cylinder_radius ≤ cone_diameter / 2
  cylinder_height_eq_diameter : cylinder_radius * 2 = cylinder_radius * 2
  shared_axis : True

/-- Theorem: The radius of the inscribed cylinder is 24/5 -/
theorem inscribed_cylinder_radius 
  (c : ConeWithCylinder) 
  (h1 : c.cone_diameter = 16) 
  (h2 : c.cone_altitude = 24) : 
  c.cylinder_radius = 24 / 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l1622_162293


namespace NUMINAMATH_CALUDE_circle_and_m_range_l1622_162218

-- Define the circle S
def circle_S (x y : ℝ) := (x - 4)^2 + (y - 4)^2 = 25

-- Define the line that contains the center of S
def center_line (x y : ℝ) := 2*x - y - 4 = 0

-- Define the intersecting line
def intersecting_line (x y m : ℝ) := x + y - m = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (7, 8)
def point_B : ℝ × ℝ := (8, 7)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem circle_and_m_range :
  ∀ (m : ℝ),
  (∃ (C D : ℝ × ℝ), 
    circle_S C.1 C.2 ∧ 
    circle_S D.1 D.2 ∧
    intersecting_line C.1 C.2 m ∧
    intersecting_line D.1 D.2 m ∧
    -- Angle COD is obtuse
    (C.1 * D.1 + C.2 * D.2 < 0)) →
  circle_S point_A.1 point_A.2 ∧
  circle_S point_B.1 point_B.2 ∧
  (∃ (center : ℝ × ℝ), center_line center.1 center.2 ∧ circle_S center.1 center.2) →
  1 < m ∧ m < 7 :=
sorry

end NUMINAMATH_CALUDE_circle_and_m_range_l1622_162218


namespace NUMINAMATH_CALUDE_margarita_vs_ricciana_distance_l1622_162267

/-- Represents the total distance of a long jump, including running and jumping. -/
structure LongJump where
  run : ℕ
  jump : ℕ

/-- Calculates the total distance of a long jump. -/
def total_distance (lj : LongJump) : ℕ := lj.run + lj.jump

theorem margarita_vs_ricciana_distance : 
  let ricciana : LongJump := { run := 20, jump := 4 }
  let margarita : LongJump := { run := 18, jump := 2 * ricciana.jump - 1 }
  total_distance margarita - total_distance ricciana = 1 := by
sorry

end NUMINAMATH_CALUDE_margarita_vs_ricciana_distance_l1622_162267


namespace NUMINAMATH_CALUDE_five_fridays_in_august_l1622_162235

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Given a month and a day of the week, count how many times that day appears -/
def countDayInMonth (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- The next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  sorry

theorem five_fridays_in_august 
  (july : Month)
  (august : Month)
  (h1 : july.days = 31)
  (h2 : august.days = 31)
  (h3 : countDayInMonth july DayOfWeek.Tuesday = 5) :
  countDayInMonth august DayOfWeek.Friday = 5 :=
sorry

end NUMINAMATH_CALUDE_five_fridays_in_august_l1622_162235


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l1622_162208

theorem quadratic_equation_two_distinct_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (2 * x₁^2 - 6 * x₁ = 7) ∧ (2 * x₂^2 - 6 * x₂ = 7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l1622_162208


namespace NUMINAMATH_CALUDE_increasing_prime_sequence_ones_digit_l1622_162250

/-- A sequence of four increasing prime numbers with common difference 4 and first term greater than 3 -/
def IncreasingPrimeSequence (p₁ p₂ p₃ p₄ : ℕ) : Prop :=
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧
  p₁ > 3 ∧
  p₂ = p₁ + 4 ∧
  p₃ = p₂ + 4 ∧
  p₄ = p₃ + 4

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

theorem increasing_prime_sequence_ones_digit
  (p₁ p₂ p₃ p₄ : ℕ) (h : IncreasingPrimeSequence p₁ p₂ p₃ p₄) :
  onesDigit p₁ = 9 := by
  sorry

end NUMINAMATH_CALUDE_increasing_prime_sequence_ones_digit_l1622_162250


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1622_162264

/-- Given a geometric sequence with first term b₁ and common ratio q,
    T_n represents the product of the first n terms. -/
def T (b₁ q : ℝ) (n : ℕ) : ℝ :=
  b₁^n * q^(n * (n - 1) / 2)

/-- Theorem: For a geometric sequence, T_4, T_8/T_4, T_12/T_8, T_16/T_12 form a geometric sequence -/
theorem geometric_sequence_property (b₁ q : ℝ) (b₁_pos : 0 < b₁) (q_pos : 0 < q) :
  ∃ r : ℝ, r ≠ 0 ∧
    (T b₁ q 8 / T b₁ q 4) = (T b₁ q 4) * r ∧
    (T b₁ q 12 / T b₁ q 8) = (T b₁ q 8 / T b₁ q 4) * r ∧
    (T b₁ q 16 / T b₁ q 12) = (T b₁ q 12 / T b₁ q 8) * r :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1622_162264


namespace NUMINAMATH_CALUDE_division_problem_l1622_162214

theorem division_problem (A : ℕ) : A / 3 = 8 ∧ A % 3 = 2 → A = 26 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1622_162214


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l1622_162274

/-- Two vectors are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, if they are collinear, then x = 3 -/
theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, -5)
  let b : ℝ × ℝ := (x - 1, -10)
  collinear a b → x = 3 := by
    sorry


end NUMINAMATH_CALUDE_collinear_vectors_x_value_l1622_162274


namespace NUMINAMATH_CALUDE_min_fourth_integer_l1622_162258

theorem min_fourth_integer (A B C D : ℕ+) : 
  (A + B + C + D : ℚ) / 4 = 16 →
  A = 3 * B →
  B = C - 2 →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  D ≥ 52 :=
by sorry

end NUMINAMATH_CALUDE_min_fourth_integer_l1622_162258


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l1622_162230

theorem rectangle_area_theorem (x : ℝ) : 
  let large_rectangle_area := (2*x + 14) * (2*x + 10)
  let hole_area := (4*x - 6) * (2*x - 4)
  let square_area := (x + 3)^2
  large_rectangle_area - hole_area + square_area = -3*x^2 + 82*x + 125 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l1622_162230


namespace NUMINAMATH_CALUDE_toy_position_l1622_162239

/-- Given a row of toys, this function calculates the position from the left
    based on the total number of toys and the position from the right. -/
def position_from_left (total : ℕ) (position_from_right : ℕ) : ℕ :=
  total - position_from_right + 1

/-- Theorem stating that in a row of 19 toys, 
    if a toy is 8th from the right, it is 12th from the left. -/
theorem toy_position (total : ℕ) (position_from_right : ℕ) 
  (h1 : total = 19) (h2 : position_from_right = 8) : 
  position_from_left total position_from_right = 12 := by
  sorry

end NUMINAMATH_CALUDE_toy_position_l1622_162239


namespace NUMINAMATH_CALUDE_bhanu_house_rent_expenditure_l1622_162234

/-- Calculates Bhanu's expenditure on house rent based on his spending pattern -/
theorem bhanu_house_rent_expenditure (total_income : ℝ) 
  (h1 : 0.30 * total_income = 300) 
  (h2 : total_income > 0) : 
  0.14 * (total_income - 0.30 * total_income) = 98 := by
  sorry

end NUMINAMATH_CALUDE_bhanu_house_rent_expenditure_l1622_162234


namespace NUMINAMATH_CALUDE_marked_price_calculation_l1622_162225

theorem marked_price_calculation (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  cost_price = 100 →
  discount_rate = 0.2 →
  profit_rate = 0.2 →
  ∃ (marked_price : ℝ), 
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) ∧
    marked_price = 150 :=
by sorry

end NUMINAMATH_CALUDE_marked_price_calculation_l1622_162225


namespace NUMINAMATH_CALUDE_star_value_l1622_162295

-- Define the operation *
def star (a b : ℤ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem star_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 12) (prod_eq : a * b = 32) : 
  star a b = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l1622_162295


namespace NUMINAMATH_CALUDE_triangle_area_rational_l1622_162200

theorem triangle_area_rational (x₁ x₂ y₂ : ℤ) :
  ∃ (a b : ℤ), b ≠ 0 ∧ (1/2 : ℚ) * |x₁ + x₂ - x₁*y₂ - x₂*y₂| = a / b :=
sorry

end NUMINAMATH_CALUDE_triangle_area_rational_l1622_162200


namespace NUMINAMATH_CALUDE_ellen_legos_l1622_162248

theorem ellen_legos (initial_legos : ℕ) (lost_legos : ℕ) 
  (h1 : initial_legos = 2080) 
  (h2 : lost_legos = 17) : 
  initial_legos - lost_legos = 2063 := by
sorry

end NUMINAMATH_CALUDE_ellen_legos_l1622_162248


namespace NUMINAMATH_CALUDE_fish_fillet_distribution_l1622_162260

theorem fish_fillet_distribution (total : ℕ) (second_team : ℕ) (third_team : ℕ) 
  (h1 : total = 500)
  (h2 : second_team = 131)
  (h3 : third_team = 180) :
  total - (second_team + third_team) = 189 := by
  sorry

end NUMINAMATH_CALUDE_fish_fillet_distribution_l1622_162260


namespace NUMINAMATH_CALUDE_grocery_spending_fraction_l1622_162289

theorem grocery_spending_fraction (initial_amount : ℝ) (magazine_fraction : ℝ) (final_amount : ℝ) 
  (h1 : initial_amount = 600)
  (h2 : magazine_fraction = 1/4)
  (h3 : final_amount = 360) :
  ∃ F : ℝ, 
    0 ≤ F ∧ F ≤ 1 ∧
    final_amount = (1 - F) * initial_amount * (1 - magazine_fraction) ∧
    F = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_grocery_spending_fraction_l1622_162289


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1622_162203

theorem sqrt_equation_solution (x : ℚ) : 
  (Real.sqrt (4 * x + 7) / Real.sqrt (8 * x + 10) = Real.sqrt 7 / 4) → x = -21/4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1622_162203


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_l1622_162215

theorem complex_reciprocal_sum (z z₁ z₂ : ℂ) : 
  z₁ = 5 + 10*I → z₂ = 3 - 4*I → (1 : ℂ)/z = 1/z₁ + 1/z₂ → z = 5 - (5/2)*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_l1622_162215


namespace NUMINAMATH_CALUDE_years_between_second_and_third_car_l1622_162286

def year_first_car : ℕ := 1970
def years_between_first_and_second : ℕ := 10
def year_third_car : ℕ := 2000

theorem years_between_second_and_third_car : 
  year_third_car - (year_first_car + years_between_first_and_second) = 20 := by
  sorry

end NUMINAMATH_CALUDE_years_between_second_and_third_car_l1622_162286


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l1622_162254

/-- A triangle with side lengths 3, 2x+1, and 10 exists if and only if 3 < x < 6 -/
theorem triangle_existence_condition (x : ℝ) :
  (3 : ℝ) < x ∧ x < 6 ↔ 
  (3 : ℝ) + (2*x + 1) > 10 ∧
  (3 : ℝ) + 10 > 2*x + 1 ∧
  10 + (2*x + 1) > 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l1622_162254


namespace NUMINAMATH_CALUDE_remaining_water_l1622_162283

/-- 
Given an initial amount of water and an amount used, 
calculate the remaining amount of water.
-/
theorem remaining_water (initial : ℚ) (used : ℚ) (remaining : ℚ) :
  initial = 4 →
  used = 9/4 →
  remaining = initial - used →
  remaining = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_water_l1622_162283


namespace NUMINAMATH_CALUDE_quadratic_equation_with_prime_roots_l1622_162227

theorem quadratic_equation_with_prime_roots (a m : ℤ) :
  (∃ x y : ℕ, x ≠ y ∧ Prime x ∧ Prime y ∧ (a * x^2 - m * x + 1996 = 0) ∧ (a * y^2 - m * y + 1996 = 0)) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_prime_roots_l1622_162227


namespace NUMINAMATH_CALUDE_linear_inequality_m_value_l1622_162217

/-- If 3m - 5x^(3+m) > 4 is a linear inequality in x, then m = -2 -/
theorem linear_inequality_m_value (m : ℝ) : 
  (∃ (a b : ℝ), ∀ x, 3*m - 5*x^(3+m) > 4 ↔ a*x + b > 0) → m = -2 :=
sorry

end NUMINAMATH_CALUDE_linear_inequality_m_value_l1622_162217


namespace NUMINAMATH_CALUDE_sin_difference_product_l1622_162247

theorem sin_difference_product (a b : ℝ) : 
  Real.sin (2 * a + b) - Real.sin (2 * a - b) = 2 * Real.cos (2 * a) * Real.sin b := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_product_l1622_162247


namespace NUMINAMATH_CALUDE_charles_speed_l1622_162297

/-- Charles' stroll scenario -/
def charles_stroll (distance : ℝ) (time : ℝ) : Prop :=
  distance = 6 ∧ time = 2 ∧ distance / time = 3

theorem charles_speed : ∃ (distance time : ℝ), charles_stroll distance time :=
  sorry

end NUMINAMATH_CALUDE_charles_speed_l1622_162297


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1622_162212

/-- A quadratic equation (k-1)x^2 + 4x + 1 = 0 has two distinct real roots -/
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  (k - 1 ≠ 0) ∧ (16 - 4 * k + 4 > 0)

/-- The range of k for which the quadratic equation has two distinct real roots -/
theorem quadratic_roots_range :
  ∀ k : ℝ, has_two_distinct_real_roots k ↔ (k < 5 ∧ k ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1622_162212


namespace NUMINAMATH_CALUDE_fibonacci_periodicity_l1622_162280

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_periodicity :
  (∀ n, 10 ∣ (fib (n + 60) - fib n)) ∧
  (∀ k, 1 ≤ k → k < 60 → ∃ n, ¬(10 ∣ (fib (n + k) - fib n))) ∧
  (∀ n, 100 ∣ (fib (n + 300) - fib n)) ∧
  (∀ k, 1 ≤ k → k < 300 → ∃ n, ¬(100 ∣ (fib (n + k) - fib n))) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_periodicity_l1622_162280


namespace NUMINAMATH_CALUDE_max_value_of_s_l1622_162268

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 20) :
  s ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_s_l1622_162268


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1622_162213

/-- Given a rectangular field with one uncovered side of 30 feet and three sides
    requiring 70 feet of fencing, the area of the field is 600 square feet. -/
theorem rectangular_field_area (L W : ℝ) : 
  L = 30 →
  L + 2 * W = 70 →
  L * W = 600 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1622_162213


namespace NUMINAMATH_CALUDE_two_digit_number_digit_difference_l1622_162266

theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 36 → x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_digit_difference_l1622_162266


namespace NUMINAMATH_CALUDE_ending_number_proof_l1622_162226

/-- The ending number for a sequence of even numbers -/
def ending_number : ℕ := 20

/-- The average of the sequence -/
def average : ℕ := 16

/-- The starting point of the sequence -/
def start : ℕ := 11

theorem ending_number_proof :
  ∀ n : ℕ,
  n > start →
  n ≤ ending_number →
  n % 2 = 0 →
  2 * average = 12 + ending_number :=
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l1622_162226


namespace NUMINAMATH_CALUDE_solve_system_1_solve_system_2_l1622_162259

-- First system of equations
theorem solve_system_1 (x y : ℝ) : 
  x - y - 1 = 0 ∧ 4 * (x - y) - y = 0 → x = 5 ∧ y = 4 := by
  sorry

-- Second system of equations
theorem solve_system_2 (x y : ℝ) :
  3 * x - y - 2 = 0 ∧ (6 * x - 2 * y + 1) / 5 + 3 * y = 10 → x = 5 / 3 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_1_solve_system_2_l1622_162259


namespace NUMINAMATH_CALUDE_train_speed_fraction_l1622_162204

theorem train_speed_fraction (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 2 → delay = 1/3 → 
  (2 : ℝ) / (2 + delay) = 6/7 := by sorry

end NUMINAMATH_CALUDE_train_speed_fraction_l1622_162204


namespace NUMINAMATH_CALUDE_oz_words_lost_l1622_162270

/-- The number of letters in the Oz alphabet -/
def alphabet_size : ℕ := 68

/-- The maximum number of letters allowed in a word -/
def max_word_length : ℕ := 2

/-- The position of the forbidden letter in the alphabet -/
def forbidden_letter_position : ℕ := 7

/-- Calculates the number of words lost due to prohibiting a letter -/
def words_lost (alphabet_size : ℕ) (max_word_length : ℕ) (forbidden_letter_position : ℕ) : ℕ :=
  let one_letter_words_lost := 1
  let two_letter_words_lost := 2 * (alphabet_size - 1)
  one_letter_words_lost + two_letter_words_lost

/-- The theorem stating the number of words lost in Oz -/
theorem oz_words_lost :
  words_lost alphabet_size max_word_length forbidden_letter_position = 135 := by
  sorry

end NUMINAMATH_CALUDE_oz_words_lost_l1622_162270


namespace NUMINAMATH_CALUDE_greatest_a_for_inequality_l1622_162245

theorem greatest_a_for_inequality : 
  ∃ (a : ℝ), (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), 
    x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ a * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅)) ∧ 
  (∀ (b : ℝ), (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), 
    x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ b * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅)) → b ≤ a) ∧ 
  a = 2 / Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_greatest_a_for_inequality_l1622_162245


namespace NUMINAMATH_CALUDE_angle_complement_l1622_162244

theorem angle_complement (given_angle : ℝ) (straight_angle : ℝ) :
  given_angle = 13 →
  straight_angle = 180 →
  (straight_angle - (13 * given_angle)) = 11 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_l1622_162244


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1622_162228

theorem arithmetic_calculation : -8 * 4 - (-6 * -3) + (-10 * -5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1622_162228


namespace NUMINAMATH_CALUDE_purchase_cost_l1622_162277

theorem purchase_cost (pretzel_cost : ℝ) (chip_cost_percentage : ℝ) : 
  pretzel_cost = 4 →
  chip_cost_percentage = 175 →
  2 * pretzel_cost + 2 * (pretzel_cost * chip_cost_percentage / 100) = 22 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l1622_162277


namespace NUMINAMATH_CALUDE_joshua_journey_l1622_162276

/-- Proves that given a journey where half the distance is traveled at 12 km/h and 
    the other half at 8 km/h, with a total journey time of 50 minutes, 
    the distance traveled in the second half (jogging) is 4 km. -/
theorem joshua_journey (total_time : ℝ) (speed1 speed2 : ℝ) (h1 : total_time = 50 / 60) 
  (h2 : speed1 = 12) (h3 : speed2 = 8) : 
  let d := (total_time * speed1 * speed2) / (speed1 + speed2)
  d = 4 := by sorry

end NUMINAMATH_CALUDE_joshua_journey_l1622_162276

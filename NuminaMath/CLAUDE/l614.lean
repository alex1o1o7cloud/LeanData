import Mathlib

namespace salmon_migration_result_l614_61429

/-- The total number of salmon in a river after migration -/
def total_salmon (initial : ℕ) (increase_factor : ℕ) : ℕ :=
  initial + initial * increase_factor

/-- Theorem: Given 500 initial salmon and a tenfold increase, the total is 5500 -/
theorem salmon_migration_result :
  total_salmon 500 10 = 5500 := by
  sorry

end salmon_migration_result_l614_61429


namespace R_value_when_S_is_12_l614_61486

-- Define the relationship between R and S
def R (g : ℝ) (S : ℝ) : ℝ := g * S - 6

-- State the theorem
theorem R_value_when_S_is_12 : 
  ∃ g : ℝ, (R g 6 = 12) → (R g 12 = 30) :=
by
  sorry

end R_value_when_S_is_12_l614_61486


namespace margie_change_is_six_l614_61426

/-- The change Margie received after buying apples -/
def margieChange (numApples : ℕ) (costPerApple : ℚ) (amountPaid : ℚ) : ℚ :=
  amountPaid - (numApples : ℚ) * costPerApple

/-- Theorem: Margie's change is $6.00 -/
theorem margie_change_is_six :
  margieChange 5 (80 / 100) 10 = 6 := by
  sorry

end margie_change_is_six_l614_61426


namespace fraction_simplification_and_evaluation_l614_61498

theorem fraction_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 0 →
  (2 / (x^2 - 4)) / (1 / (x^2 - 2*x)) = 2*x / (x + 2) ∧
  (2 * (-1)) / ((-1) + 2) = -2 := by
  sorry

end fraction_simplification_and_evaluation_l614_61498


namespace geometric_sequence_properties_l614_61408

/-- Given a geometric sequence with common ratio q > 0 and T_n as the product of the first n terms,
    if T_7 > T_6 > T_8, then 0 < q < 1 and T_13 > 1 > T_14 -/
theorem geometric_sequence_properties (q : ℝ) (T : ℕ → ℝ) 
  (h_q_pos : q > 0)
  (h_T : ∀ n : ℕ, T n = (T 1) * q^(n * (n - 1) / 2))
  (h_ineq : T 7 > T 6 ∧ T 6 > T 8) :
  (0 < q ∧ q < 1) ∧ (T 13 > 1 ∧ 1 > T 14) := by
  sorry

end geometric_sequence_properties_l614_61408


namespace claire_crafting_time_l614_61462

/-- Represents the system of equations for Claire's time allocation --/
structure ClaireTimeSystem where
  x : ℝ
  y : ℝ
  z : ℝ
  crafting : ℝ
  tailoring : ℝ
  eq1 : (2 * y) + y + (y - 1) + crafting + crafting + 8 = 24
  eq2 : x = 2 * y
  eq3 : z = y - 1
  eq4 : crafting = tailoring
  eq5 : 2 * crafting = 9 - tailoring

/-- Theorem stating that in any valid ClaireTimeSystem, the crafting time is 3 hours --/
theorem claire_crafting_time (s : ClaireTimeSystem) : s.crafting = 3 := by
  sorry

end claire_crafting_time_l614_61462


namespace major_axis_length_for_given_cylinder_l614_61494

/-- The length of the major axis of an ellipse formed by cutting a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem: The length of the major axis of an ellipse formed by cutting a right circular cylinder
    with radius 2 is 5.6, given that the major axis is 40% longer than the minor axis --/
theorem major_axis_length_for_given_cylinder :
  major_axis_length 2 1.4 = 5.6 := by sorry

end major_axis_length_for_given_cylinder_l614_61494


namespace count_solutions_for_a_main_result_l614_61482

theorem count_solutions_for_a (max_a : Nat) : Nat :=
  let count_pairs (a : Nat) : Nat :=
    (Finset.filter (fun p : Nat × Nat =>
      let m := p.1
      let n := p.2
      n * (1 - m) + a * (1 + m) = 0 ∧ 
      m > 0 ∧ n > 0
    ) (Finset.product (Finset.range (max_a + 1)) (Finset.range (max_a + 1)))).card

  (Finset.filter (fun a : Nat =>
    a > 0 ∧ a ≤ max_a ∧ count_pairs a = 6
  ) (Finset.range (max_a + 1))).card

theorem main_result : count_solutions_for_a 50 = 12 := by
  sorry

end count_solutions_for_a_main_result_l614_61482


namespace difference_number_and_three_fifths_l614_61448

theorem difference_number_and_three_fifths (n : ℚ) : n = 160 → n - (3 / 5 * n) = 64 := by
  sorry

end difference_number_and_three_fifths_l614_61448


namespace x_squared_plus_4x_plus_5_range_l614_61433

theorem x_squared_plus_4x_plus_5_range :
  ∀ x : ℝ, x^2 - 7*x + 12 < 0 →
  ∃ y ∈ Set.Ioo 26 37, y = x^2 + 4*x + 5 ∧
  ∀ z, z = x^2 + 4*x + 5 → z ∈ Set.Ioo 26 37 :=
by sorry

end x_squared_plus_4x_plus_5_range_l614_61433


namespace claire_pets_ratio_l614_61463

theorem claire_pets_ratio : 
  ∀ (total_pets gerbils hamsters male_gerbils male_hamsters : ℕ),
    total_pets = 92 →
    gerbils + hamsters = total_pets →
    gerbils = 68 →
    male_hamsters = hamsters / 3 →
    male_gerbils + male_hamsters = 25 →
    male_gerbils * 4 = gerbils :=
by
  sorry

end claire_pets_ratio_l614_61463


namespace floor_paving_cost_l614_61421

/-- The cost of paving a rectangular floor -/
theorem floor_paving_cost 
  (length : ℝ) 
  (width : ℝ) 
  (rate : ℝ) 
  (h1 : length = 5.5) 
  (h2 : width = 3.75) 
  (h3 : rate = 1200) : 
  length * width * rate = 24750 := by
  sorry

end floor_paving_cost_l614_61421


namespace quadratic_solution_absolute_value_l614_61411

theorem quadratic_solution_absolute_value : ∃ (x : ℝ), x^2 + 18*x + 81 = 0 ∧ |x| = 9 := by sorry

end quadratic_solution_absolute_value_l614_61411


namespace cross_section_perimeter_bound_l614_61434

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron where
  a : ℝ
  edge_positive : 0 < a

/-- A triangular cross-section through one vertex of a regular tetrahedron -/
structure TriangularCrossSection (t : RegularTetrahedron) where
  perimeter : ℝ

/-- The perimeter of any triangular cross-section through one vertex of a regular tetrahedron
    is greater than twice the edge length -/
theorem cross_section_perimeter_bound (t : RegularTetrahedron) 
  (s : TriangularCrossSection t) : s.perimeter > 2 * t.a := by
  sorry

end cross_section_perimeter_bound_l614_61434


namespace arithmetic_equality_l614_61466

theorem arithmetic_equality : 4 * 5 + 5 * 4 = 40 := by
  sorry

end arithmetic_equality_l614_61466


namespace dilin_gave_sword_l614_61470

-- Define the types for individuals and gifts
inductive Individual : Type
| Ilse : Individual
| Elsa : Individual
| Bilin : Individual
| Dilin : Individual

inductive Gift : Type
| Sword : Gift
| Necklace : Gift

-- Define the type for statements
inductive Statement : Type
| GiftWasSword : Statement
| IDidNotGive : Statement
| IlseGaveNecklace : Statement
| BilinGaveSword : Statement

-- Define a function to determine if an individual is an elf
def isElf (i : Individual) : Prop :=
  i = Individual.Ilse ∨ i = Individual.Elsa

-- Define a function to determine if an individual is a dwarf
def isDwarf (i : Individual) : Prop :=
  i = Individual.Bilin ∨ i = Individual.Dilin

-- Define the truth value of a statement given who made it and who gave the gift
def isTruthful (speaker : Individual) (giver : Individual) (gift : Gift) (s : Statement) : Prop :=
  match s with
  | Statement.GiftWasSword => gift = Gift.Sword
  | Statement.IDidNotGive => speaker ≠ giver
  | Statement.IlseGaveNecklace => giver = Individual.Ilse ∧ gift = Gift.Necklace
  | Statement.BilinGaveSword => giver = Individual.Bilin ∧ gift = Gift.Sword

-- Define the conditions of truthfulness based on the problem statement
def meetsConditions (speaker : Individual) (giver : Individual) (gift : Gift) (s : Statement) : Prop :=
  (isElf speaker ∧ isDwarf giver → ¬isTruthful speaker giver gift s) ∧
  (isDwarf speaker ∧ (s = Statement.IDidNotGive ∨ s = Statement.IlseGaveNecklace) → ¬isTruthful speaker giver gift s) ∧
  (¬(isElf speaker ∧ isDwarf giver) ∧ ¬(isDwarf speaker ∧ (s = Statement.IDidNotGive ∨ s = Statement.IlseGaveNecklace)) → isTruthful speaker giver gift s)

-- The theorem to be proved
theorem dilin_gave_sword :
  ∃ (speakers : Fin 4 → Individual),
    (∃ (statements : Fin 4 → Statement),
      (∀ i : Fin 4, meetsConditions (speakers i) Individual.Dilin Gift.Sword (statements i)) ∧
      (∃ i : Fin 4, statements i = Statement.GiftWasSword) ∧
      (∃ i : Fin 4, statements i = Statement.IDidNotGive) ∧
      (∃ i : Fin 4, statements i = Statement.IlseGaveNecklace) ∧
      (∃ i : Fin 4, statements i = Statement.BilinGaveSword)) :=
sorry

end dilin_gave_sword_l614_61470


namespace gcd_g_x_l614_61412

def g (x : ℤ) : ℤ := (3*x+5)*(9*x+4)*(11*x+8)*(x+11)

theorem gcd_g_x (x : ℤ) (h : ∃ k : ℤ, x = 34914 * k) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 1760 := by
  sorry

end gcd_g_x_l614_61412


namespace jellybean_probability_l614_61410

def total_jellybeans : ℕ := 15
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 3
def white_jellybeans : ℕ := 4
def green_jellybeans : ℕ := 3
def picked_jellybeans : ℕ := 4

theorem jellybean_probability :
  (Nat.choose red_jellybeans 2 * Nat.choose green_jellybeans 1 * Nat.choose (total_jellybeans - red_jellybeans - green_jellybeans) 1) / 
  Nat.choose total_jellybeans picked_jellybeans = 2 / 13 :=
sorry

end jellybean_probability_l614_61410


namespace polynomial_coefficient_b_l614_61401

theorem polynomial_coefficient_b (a c d : ℝ) : 
  ∃ (p q r s : ℂ),
    (∀ x : ℂ, x^4 + a*x^3 + 49*x^2 + c*x + d = 0 ↔ x = p ∨ x = q ∨ x = r ∨ x = s) ∧
    p + q = 5 + 2*I ∧
    r * s = 10 - I ∧
    p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧
    p.im ≠ 0 ∧ q.im ≠ 0 ∧ r.im ≠ 0 ∧ s.im ≠ 0 := by
  sorry

end polynomial_coefficient_b_l614_61401


namespace f_zero_f_expression_intersection_complement_l614_61437

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Given conditions
axiom f_property : ∀ (x y : ℝ), f (x + y) - f y = x * (x + 2 * y + 1)
axiom f_one : f 1 = 0

-- Define sets A and B
def A : Set ℝ := {a | ∀ x ∈ Set.Ioo 0 (1/2), f x + 3 < 2 * x + a}
def B : Set ℝ := {a | ∀ x ∈ Set.Icc (-2) 2, Monotone (fun x ↦ f x - a * x)}

-- Theorem statements
theorem f_zero : f 0 = -2 := sorry

theorem f_expression : ∀ x : ℝ, f x = x^2 + x - 2 := sorry

theorem intersection_complement : A ∩ (Set.univ \ B) = Set.Icc 1 5 := sorry

end f_zero_f_expression_intersection_complement_l614_61437


namespace marching_band_formation_l614_61478

/-- A function that returns the list of divisors of a natural number -/
def divisors (n : ℕ) : List ℕ := sorry

/-- A function that returns the number of divisors of a natural number within a given range -/
def countDivisorsInRange (n m l : ℕ) : ℕ := sorry

theorem marching_band_formation (total_musicians : ℕ) (min_per_row : ℕ) (num_formations : ℕ) 
  (h1 : total_musicians = 240)
  (h2 : min_per_row = 8)
  (h3 : num_formations = 8) :
  ∃ (max_per_row : ℕ), 
    countDivisorsInRange total_musicians min_per_row max_per_row = num_formations ∧ 
    max_per_row = 80 := by
  sorry

end marching_band_formation_l614_61478


namespace test_point_value_l614_61432

theorem test_point_value
  (total_points : ℕ)
  (total_questions : ℕ)
  (two_point_questions : ℕ)
  (other_type_questions : ℕ)
  (h1 : total_points = 100)
  (h2 : total_questions = 40)
  (h3 : other_type_questions = 10)
  (h4 : two_point_questions + other_type_questions = total_questions)
  (h5 : 2 * two_point_questions + other_type_questions * (total_points - 2 * two_point_questions) / other_type_questions = total_points) :
  (total_points - 2 * two_point_questions) / other_type_questions = 4 :=
by sorry

end test_point_value_l614_61432


namespace subtraction_problem_l614_61474

theorem subtraction_problem (A B : ℕ) : 
  (A ≥ 10 ∧ A ≤ 99) → 
  (B ≥ 10 ∧ B ≤ 99) → 
  A = 23 - 8 → 
  B + 7 = 18 → 
  A - B = 4 := by
sorry

end subtraction_problem_l614_61474


namespace parallelogram_base_length_l614_61467

theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 480) 
  (h2 : height = 15) : 
  area / height = 32 := by
  sorry

end parallelogram_base_length_l614_61467


namespace quadratic_inequality_range_l614_61404

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
by sorry

end quadratic_inequality_range_l614_61404


namespace negative_root_condition_l614_61487

theorem negative_root_condition (p : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ x^4 - 4*p*x^3 + x^2 - 4*p*x + 1 = 0) ↔ p ≥ -3/8 := by sorry

end negative_root_condition_l614_61487


namespace olaf_collection_l614_61403

def total_cars (initial : ℕ) (uncle : ℕ) : ℕ :=
  let grandpa := 2 * uncle
  let dad := 10
  let mum := dad + 5
  let auntie := 6
  let cousin_liam := dad / 2
  let cousin_emma := uncle / 3
  let grandmother := 3 * auntie
  initial + grandpa + dad + mum + auntie + uncle + cousin_liam + cousin_emma + grandmother

theorem olaf_collection (initial : ℕ) (uncle : ℕ) 
  (h1 : initial = 150)
  (h2 : uncle = 5)
  (h3 : auntie = uncle + 1) :
  total_cars initial uncle = 220 := by
  sorry

#eval total_cars 150 5

end olaf_collection_l614_61403


namespace rectangle_width_l614_61469

theorem rectangle_width (w : ℝ) (l : ℝ) (P : ℝ) : 
  l = 2 * w + 6 →  -- length is 6 more than twice the width
  P = 2 * l + 2 * w →  -- perimeter formula
  P = 120 →  -- given perimeter
  w = 18 :=  -- width to prove
by sorry

end rectangle_width_l614_61469


namespace geometric_arithmetic_sequence_ratio_l614_61496

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- {a_n} is a geometric sequence with common ratio q
  (a 6 + a 8 - a 5 = a 7 - (a 6 + a 8)) →  -- a_5, a_6 + a_8, a_7 form an arithmetic sequence
  q = 1 / 2 :=
by sorry

end geometric_arithmetic_sequence_ratio_l614_61496


namespace distance_traveled_l614_61417

-- Define the velocity function
def velocity (t : ℝ) : ℝ := 2 * t + 3

-- Define the theorem
theorem distance_traveled (a b : ℝ) (ha : a = 3) (hb : b = 5) :
  ∫ x in a..b, velocity x = 22 := by
  sorry

end distance_traveled_l614_61417


namespace sum_of_numbers_l614_61485

theorem sum_of_numbers : 3 + 33 + 333 + 3.33 = 372.33 := by
  sorry

end sum_of_numbers_l614_61485


namespace subset_intersection_theorem_l614_61492

theorem subset_intersection_theorem (α : ℝ) (h_pos : α > 0) (h_bound : α < (3 - Real.sqrt 5) / 2) :
  ∃ (n p : ℕ+) (S T : Fin p → Finset (Fin n)),
    p > α * 2^(n : ℝ) ∧
    (∀ i j : Fin p, i ≠ j → S i ≠ S j) ∧
    (∀ i j : Fin p, i ≠ j → T i ≠ T j) ∧
    (∀ i j : Fin p, (S i ∩ T j).Nonempty) :=
by sorry

end subset_intersection_theorem_l614_61492


namespace max_cos_value_l614_61436

theorem max_cos_value (a b : ℝ) (h : Real.cos (a - b) = Real.cos a - Real.cos b) :
  ∀ x : ℝ, Real.cos a ≤ 1 ∧ (Real.cos x ≤ Real.cos a → x = a) :=
by sorry

end max_cos_value_l614_61436


namespace tan_sin_ratio_thirty_degrees_l614_61407

theorem tan_sin_ratio_thirty_degrees :
  let tan_30_sq := sin_30_sq / cos_30_sq
  let sin_30_sq := (1 : ℝ) / 4
  let cos_30_sq := (3 : ℝ) / 4
  (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end tan_sin_ratio_thirty_degrees_l614_61407


namespace cistern_length_l614_61425

/-- Represents a rectangular cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the total wet surface area of a cistern --/
def wetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem: A cistern with given dimensions has a length of 4 meters --/
theorem cistern_length : 
  ∃ (c : Cistern), c.width = 8 ∧ c.depth = 1.25 ∧ wetSurfaceArea c = 62 → c.length = 4 := by
  sorry

end cistern_length_l614_61425


namespace sandy_age_multiple_is_ten_l614_61477

/-- The multiple of Sandy's age that equals her monthly phone bill expense -/
def sandy_age_multiple : ℕ → ℕ → ℕ → ℕ
| kim_age, sandy_future_age, sandy_expense =>
  let sandy_current_age := sandy_future_age - 2
  sandy_expense / sandy_current_age

/-- Theorem stating the multiple of Sandy's age that equals her monthly phone bill expense -/
theorem sandy_age_multiple_is_ten :
  sandy_age_multiple 10 36 340 = 10 := by
  sorry

end sandy_age_multiple_is_ten_l614_61477


namespace arithmetic_sequence_n_equals_10_l614_61406

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 1
  sum_3_5 : a 3 + a 5 = 14
  sum_n : ∃ n : ℕ, n > 0 ∧ (n : ℝ) * (a 1 + a n) / 2 = 100

/-- The theorem stating that n = 10 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_equals_10 (seq : ArithmeticSequence) :
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * (seq.a 1 + seq.a n) / 2 = 100 ∧ n = 10 := by
  sorry

end arithmetic_sequence_n_equals_10_l614_61406


namespace water_consumption_equation_l614_61476

theorem water_consumption_equation (x : ℝ) (h : x > 0) : 
  (80 / x) - (80 * (1 - 0.2) / x) = 5 ↔ 
  (80 / x) - (80 / (x / (1 - 0.2))) = 5 :=
sorry

end water_consumption_equation_l614_61476


namespace coloring_books_distribution_l614_61419

theorem coloring_books_distribution (initial_stock : ℕ) (books_sold : ℕ) (num_shelves : ℕ) 
  (h1 : initial_stock = 27)
  (h2 : books_sold = 6)
  (h3 : num_shelves = 3) :
  (initial_stock - books_sold) / num_shelves = 7 := by
  sorry

end coloring_books_distribution_l614_61419


namespace egg_weight_calculation_l614_61456

/-- Given the total weight of eggs and the number of dozens, 
    calculate the weight of a single egg. -/
theorem egg_weight_calculation 
  (total_weight : ℝ) 
  (dozens : ℕ) 
  (h1 : total_weight = 6) 
  (h2 : dozens = 8) : 
  total_weight / (dozens * 12) = 0.0625 := by
  sorry

end egg_weight_calculation_l614_61456


namespace intersection_point_l614_61459

def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6
def line2 (x y : ℚ) : Prop := -2 * y = 6 * x + 4

theorem intersection_point : 
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (-12/7, 22/7) := by
sorry

end intersection_point_l614_61459


namespace arithmetic_sequence_properties_l614_61446

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℝ := n + 1

-- Define the sum of the first n terms of 2a_n
def S (n : ℕ) : ℝ := 2^(n+2) - 4

-- Theorem statement
theorem arithmetic_sequence_properties :
  (a 1 = 2) ∧ 
  (a 1 + a 2 + a 3 = 9) ∧
  (∀ n : ℕ, a n = n + 1) ∧
  (∀ n : ℕ, S n = 2^(n+2) - 4) :=
by sorry

end arithmetic_sequence_properties_l614_61446


namespace part_one_part_two_l614_61435

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (x : ℝ) :
  let a := 2
  f a x ≥ 7 - |x - 1| ↔ x ∈ Set.Iic (-2) ∪ Set.Ici 5 := by sorry

-- Part II
theorem part_two (m n : ℝ) (h1 : m > 0) (h2 : n > 0) :
  (∀ x, f 1 x ≤ 1 ↔ x ∈ Set.Icc 0 2) →
  m^2 + 2*n^2 = 1 →
  m + 4*n ≤ 3 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ m^2 + 2*n^2 = 1 ∧ m + 4*n = 3 := by sorry

end part_one_part_two_l614_61435


namespace lakeview_academy_teachers_l614_61455

/-- Represents the number of teachers at Lakeview Academy -/
def num_teachers (total_students : ℕ) (classes_per_student : ℕ) (class_size : ℕ) (classes_per_teacher : ℕ) : ℕ :=
  (total_students * classes_per_student * 2) / (class_size * classes_per_teacher)

/-- Theorem stating the number of teachers at Lakeview Academy -/
theorem lakeview_academy_teachers :
  num_teachers 1500 6 25 5 = 144 := by
  sorry

#eval num_teachers 1500 6 25 5

end lakeview_academy_teachers_l614_61455


namespace equations_different_graphs_l614_61495

-- Define the three equations
def eq1 (x y : ℝ) : Prop := y = 2 * x - 3
def eq2 (x y : ℝ) : Prop := y = (2 * x^2 - 18) / (x + 3)
def eq3 (x y : ℝ) : Prop := (x + 3) * y = 2 * x^2 - 18

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, eq1 x y ↔ eq2 x y

-- Theorem statement
theorem equations_different_graphs :
  ¬(same_graph eq1 eq2) ∧ ¬(same_graph eq1 eq3) ∧ ¬(same_graph eq2 eq3) :=
sorry

end equations_different_graphs_l614_61495


namespace total_students_in_schools_l614_61491

theorem total_students_in_schools (capacity1 capacity2 : ℕ) 
  (h1 : capacity1 = 400) 
  (h2 : capacity2 = 340) : 
  2 * capacity1 + 2 * capacity2 = 1480 := by
  sorry

end total_students_in_schools_l614_61491


namespace arithmetic_mean_of_fractions_l614_61457

theorem arithmetic_mean_of_fractions : 
  (1/3 : ℚ) * ((3/4 : ℚ) + (5/6 : ℚ) + (9/10 : ℚ)) = 149/180 := by
  sorry

end arithmetic_mean_of_fractions_l614_61457


namespace overlapping_strips_l614_61481

theorem overlapping_strips (total_length width : ℝ) 
  (left_length right_length : ℝ) 
  (left_area right_area : ℝ) : 
  total_length = 16 →
  left_length = 9 →
  right_length = 7 →
  left_length + right_length = total_length →
  left_area = 27 →
  right_area = 18 →
  (left_area + (left_length * width)) / (right_area + (right_length * width)) = left_length / right_length →
  ∃ overlap_area : ℝ, overlap_area = 13.5 ∧ 
    (left_area + overlap_area) / (right_area + overlap_area) = left_length / right_length :=
by sorry

end overlapping_strips_l614_61481


namespace line_equation_with_given_area_l614_61424

/-- Given a line passing through points (a, 0) and (b, 0) where b > a, 
    cutting a triangular region from the first quadrant with area S,
    prove that the equation of the line is 0 = -2Sx + (b-a)^2y + 2Sa - 2Sb -/
theorem line_equation_with_given_area (a b S : ℝ) (h1 : b > a) (h2 : S > 0) :
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = 0 ↔ -2 * S * x + (b - a)^2 * x + 2 * S * a - 2 * S * b = 0) ∧
    f a = 0 ∧ 
    f b = 0 ∧
    (∃ k, k > 0 ∧ f k > 0 ∧ (k - 0) * (b - a) / 2 = S) :=
by sorry

end line_equation_with_given_area_l614_61424


namespace angle_sum_in_hexagon_with_triangles_l614_61418

/-- Represents a hexagon with two connected triangles -/
structure HexagonWithTriangles where
  /-- Angle A of the hexagon -/
  angle_A : ℝ
  /-- Angle B of the hexagon -/
  angle_B : ℝ
  /-- Angle C of one of the connected triangles -/
  angle_C : ℝ
  /-- An angle x in the figure -/
  x : ℝ
  /-- An angle y in the figure -/
  y : ℝ
  /-- The sum of angles in a hexagon is 720° -/
  hexagon_sum : angle_A + angle_B + (360 - x) + 90 + (114 - y) = 720

/-- Theorem stating that x + y = 50° in the given hexagon with triangles -/
theorem angle_sum_in_hexagon_with_triangles (h : HexagonWithTriangles)
    (h_A : h.angle_A = 30)
    (h_B : h.angle_B = 76)
    (h_C : h.angle_C = 24) :
    h.x + h.y = 50 := by
  sorry

end angle_sum_in_hexagon_with_triangles_l614_61418


namespace distance_negative_five_to_origin_l614_61452

theorem distance_negative_five_to_origin : 
  abs (-5 : ℝ) = 5 := by sorry

end distance_negative_five_to_origin_l614_61452


namespace min_red_cells_for_win_thirteen_red_cells_win_l614_61458

/-- Represents an 8x8 grid where some cells are colored red -/
def Grid := Fin 8 → Fin 8 → Bool

/-- Returns true if the given cell is covered by the selected rows and columns -/
def isCovered (rows columns : Finset (Fin 8)) (i j : Fin 8) : Prop :=
  i ∈ rows ∨ j ∈ columns

/-- Returns the number of red cells in the grid -/
def redCount (g : Grid) : Nat :=
  (Finset.univ.filter (λ i => Finset.univ.filter (λ j => g i j) ≠ ∅)).card

/-- Returns true if there exists an uncovered red cell -/
def hasUncoveredRed (g : Grid) (rows columns : Finset (Fin 8)) : Prop :=
  ∃ i j, g i j ∧ ¬isCovered rows columns i j

theorem min_red_cells_for_win :
  ∀ n : Nat, n < 13 →
    ∃ g : Grid, redCount g = n ∧
      ∃ rows columns : Finset (Fin 8),
        rows.card = 4 ∧ columns.card = 4 ∧ ¬hasUncoveredRed g rows columns :=
by sorry

theorem thirteen_red_cells_win :
  ∃ g : Grid, redCount g = 13 ∧
    ∀ rows columns : Finset (Fin 8),
      rows.card = 4 ∧ columns.card = 4 → hasUncoveredRed g rows columns :=
by sorry

end min_red_cells_for_win_thirteen_red_cells_win_l614_61458


namespace age_calculation_l614_61449

/-- Given Luke's current age and Mr. Bernard's future age relative to Luke's,
    calculate 10 years less than their average current age. -/
theorem age_calculation (luke_age : ℕ) (bernard_future_age_factor : ℕ) (years_in_future : ℕ) : 
  luke_age = 20 →
  years_in_future = 8 →
  bernard_future_age_factor = 3 →
  10 < luke_age →
  (luke_age + (bernard_future_age_factor * luke_age - years_in_future)) / 2 - 10 = 26 := by
sorry

end age_calculation_l614_61449


namespace no_integer_solutions_for_square_difference_150_l614_61472

theorem no_integer_solutions_for_square_difference_150 :
  ∀ m n : ℕ+, m ≥ n → m^2 - n^2 ≠ 150 := by sorry

end no_integer_solutions_for_square_difference_150_l614_61472


namespace coinciding_vertices_l614_61454

/-- A point in the plane -/
structure Point :=
  (x y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- An isosceles right triangle defined by two points of the quadrilateral and a third point -/
structure IsoscelesRightTriangle :=
  (P Q R : Point)

/-- Predicate to check if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Predicate to check if two points coincide -/
def coincide (P Q : Point) : Prop := P.x = Q.x ∧ P.y = Q.y

/-- Theorem: If O₁ and O₃ coincide, then O₂ and O₄ coincide -/
theorem coinciding_vertices 
  (q : Quadrilateral) 
  (t1 : IsoscelesRightTriangle) 
  (t2 : IsoscelesRightTriangle) 
  (t3 : IsoscelesRightTriangle) 
  (t4 : IsoscelesRightTriangle) 
  (h1 : is_convex q)
  (h2 : t1.P = q.A ∧ t1.Q = q.B)
  (h3 : t2.P = q.B ∧ t2.Q = q.C)
  (h4 : t3.P = q.C ∧ t3.Q = q.D)
  (h5 : t4.P = q.D ∧ t4.Q = q.A)
  (h6 : coincide t1.R t3.R) :
  coincide t2.R t4.R := by sorry

end coinciding_vertices_l614_61454


namespace max_notebooks_buyable_l614_61405

def john_money : ℚ := 35.45
def notebook_cost : ℚ := 3.75

theorem max_notebooks_buyable :
  ⌊john_money / notebook_cost⌋ = 9 :=
sorry

end max_notebooks_buyable_l614_61405


namespace kennel_dogs_l614_61490

theorem kennel_dogs (total : ℕ) (long_fur : ℕ) (brown : ℕ) (long_fur_and_brown : ℕ)
  (h_total : total = 45)
  (h_long_fur : long_fur = 29)
  (h_brown : brown = 17)
  (h_long_fur_and_brown : long_fur_and_brown = 9) :
  total - (long_fur + brown - long_fur_and_brown) = 8 :=
by sorry

end kennel_dogs_l614_61490


namespace square_difference_equals_one_l614_61402

theorem square_difference_equals_one : (825 : ℤ) * 825 - 824 * 826 = 1 := by
  sorry

end square_difference_equals_one_l614_61402


namespace food_price_increase_l614_61483

theorem food_price_increase 
  (initial_students : ℝ) 
  (initial_food_price : ℝ) 
  (initial_food_consumption : ℝ) 
  (h_students_positive : initial_students > 0) 
  (h_price_positive : initial_food_price > 0) 
  (h_consumption_positive : initial_food_consumption > 0) :
  let new_students := 0.9 * initial_students
  let new_food_consumption := 0.9259259259259259 * initial_food_consumption
  let new_food_price := x * initial_food_price
  x = 1.2 ↔ 
    new_students * new_food_consumption * new_food_price = 
    initial_students * initial_food_consumption * initial_food_price := by
sorry

end food_price_increase_l614_61483


namespace min_value_expression_l614_61441

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 16) :
  x^2 + 4*x*y + 4*y^2 + z^3 ≥ 73 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 16 ∧ x^2 + 4*x*y + 4*y^2 + z^3 = 73 :=
sorry

end min_value_expression_l614_61441


namespace lg_calculation_l614_61400

-- Define lg as the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_calculation : lg 5 * lg 20 + (lg 2)^2 = 1 := by
  sorry

end lg_calculation_l614_61400


namespace fraction_simplification_l614_61438

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end fraction_simplification_l614_61438


namespace sum_of_odd_integers_between_400_and_700_l614_61431

def first_term : ℕ := 401
def last_term : ℕ := 699
def common_difference : ℕ := 2

def number_of_terms : ℕ := (last_term - first_term) / common_difference + 1

theorem sum_of_odd_integers_between_400_and_700 :
  (number_of_terms : ℝ) / 2 * (first_term + last_term : ℝ) = 82500 := by
  sorry

end sum_of_odd_integers_between_400_and_700_l614_61431


namespace find_A_l614_61451

theorem find_A : ∃ A : ℕ, 
  (1047 % A = 23) ∧ 
  (1047 % (A + 1) = 7) ∧ 
  (A = 64) := by
sorry

end find_A_l614_61451


namespace negation_of_implication_l614_61484

/-- Two lines in a 3D space -/
structure Line3D where
  -- Define necessary properties for a 3D line
  -- This is a simplified representation
  dummy : Unit

/-- Predicate to check if two lines have a common point -/
def have_common_point (l1 l2 : Line3D) : Prop :=
  sorry -- Definition omitted for simplicity

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry -- Definition omitted for simplicity

theorem negation_of_implication (l1 l2 : Line3D) :
  (¬(¬(have_common_point l1 l2) → are_skew l1 l2)) ↔
  (have_common_point l1 l2 → ¬(are_skew l1 l2)) :=
by
  sorry

#check negation_of_implication

end negation_of_implication_l614_61484


namespace cone_angle_theorem_l614_61428

/-- A cone with vertex A -/
structure Cone where
  vertexAngle : ℝ

/-- The configuration of four cones as described in the problem -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  cone4 : Cone
  cone1_eq_cone2 : cone1 = cone2
  cone3_angle : cone3.vertexAngle = π / 3
  cone4_angle : cone4.vertexAngle = 5 * π / 6
  external_tangent : True  -- Represents that cone1, cone2, and cone3 are externally tangent
  internal_tangent : True  -- Represents that cone4 is internally tangent to the other three

theorem cone_angle_theorem (config : ConeConfiguration) :
  config.cone1.vertexAngle = 2 * Real.arctan (Real.sqrt 3 - 1) := by
  sorry

end cone_angle_theorem_l614_61428


namespace unique_solution_absolute_value_equation_l614_61473

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 2| = |x - 4| + |x - 6| := by
sorry

end unique_solution_absolute_value_equation_l614_61473


namespace line_tangent_to_circle_l614_61475

theorem line_tangent_to_circle (m : ℝ) :
  (∀ x y : ℝ, x + y - m = 0 ∧ x^2 + y^2 = 2 → (∀ ε > 0, ∃ x' y' : ℝ, x' + y' - m = 0 ∧ x'^2 + y'^2 < 2 ∧ (x' - x)^2 + (y' - y)^2 < ε)) ↔
  (m > 2 ∨ m < -2) :=
sorry

end line_tangent_to_circle_l614_61475


namespace quadratic_equation_solution_l614_61489

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = 2*x} = {0, 2} := by sorry

end quadratic_equation_solution_l614_61489


namespace cone_volume_l614_61423

/-- The volume of a cone with given slant height and lateral surface area -/
theorem cone_volume (l : ℝ) (lateral_area : ℝ) (h : l = 2) (h' : lateral_area = 2 * Real.pi) :
  ∃ (r : ℝ) (h : ℝ),
    r > 0 ∧ h > 0 ∧
    lateral_area = Real.pi * r * l ∧
    h^2 + r^2 = l^2 ∧
    (1/3 : ℝ) * Real.pi * r^2 * h = (Real.sqrt 3 * Real.pi) / 3 :=
by
  sorry


end cone_volume_l614_61423


namespace polynomial_simplification_l614_61453

theorem polynomial_simplification (x : ℝ) : 
  x^2 * (4*x^3 - 3*x + 1) - 6*(x^3 - 3*x^2 + 4*x - 5) = 
  4*x^5 - 9*x^3 + 19*x^2 - 24*x + 30 := by
  sorry

end polynomial_simplification_l614_61453


namespace count_arrangements_no_adjacent_girls_count_arrangements_AB_adjacent_l614_61461

/-- The number of boys -/
def num_boys : ℕ := 3

/-- The number of girls -/
def num_girls : ℕ := 4

/-- The number of arrangements where no two girls are adjacent -/
def arrangements_no_adjacent_girls : ℕ := 144

/-- The number of arrangements where boys A and B are adjacent -/
def arrangements_AB_adjacent : ℕ := 240

/-- Theorem stating the number of arrangements where no two girls are adjacent -/
theorem count_arrangements_no_adjacent_girls :
  (num_boys.factorial * num_girls.factorial) = arrangements_no_adjacent_girls := by
  sorry

/-- Theorem stating the number of arrangements where boys A and B are adjacent -/
theorem count_arrangements_AB_adjacent :
  ((num_boys + num_girls - 1).factorial * 2) = arrangements_AB_adjacent := by
  sorry

end count_arrangements_no_adjacent_girls_count_arrangements_AB_adjacent_l614_61461


namespace quadratic_roots_sum_to_one_l614_61499

theorem quadratic_roots_sum_to_one (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x + y = 1 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ b = -a :=
sorry

end quadratic_roots_sum_to_one_l614_61499


namespace odd_prime_condition_l614_61468

theorem odd_prime_condition (p : ℕ) : 
  (Prime p ∧ Odd p) →
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ (p - 1) / 2 → Prime (1 + k * (p - 1))) →
  p = 3 ∨ p = 7 := by
sorry

end odd_prime_condition_l614_61468


namespace sqrt_meaningful_iff_geq_two_l614_61416

theorem sqrt_meaningful_iff_geq_two (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by sorry

end sqrt_meaningful_iff_geq_two_l614_61416


namespace f_geq_m_range_l614_61465

/-- The function f(x) = x^2 - 2mx + 2 -/
def f (x m : ℝ) : ℝ := x^2 - 2*m*x + 2

/-- The theorem stating the range of m for which f(x) ≥ m holds for all x ∈ [-1, +∞) -/
theorem f_geq_m_range (m : ℝ) :
  (∀ x : ℝ, x ≥ -1 → f x m ≥ m) ↔ -3 ≤ m ∧ m ≤ 1 :=
sorry

end f_geq_m_range_l614_61465


namespace smallest_four_digit_multiple_of_15_l614_61415

theorem smallest_four_digit_multiple_of_15 : 
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 15 = 0 → n ≥ 1005 :=
by sorry

end smallest_four_digit_multiple_of_15_l614_61415


namespace arithmetic_calculations_l614_61422

theorem arithmetic_calculations :
  (4 + (-7) - (-5) = 2) ∧
  (-1^2023 + 27 * (-1/3)^2 - |(-5)| = -3) := by
  sorry

end arithmetic_calculations_l614_61422


namespace scott_smoothie_sales_l614_61409

/-- Proves that Scott sold 40 cups of smoothies given the conditions of the problem -/
theorem scott_smoothie_sales :
  let smoothie_price : ℕ := 3
  let cake_price : ℕ := 2
  let cakes_sold : ℕ := 18
  let total_money : ℕ := 156
  let smoothies_sold : ℕ := (total_money - cake_price * cakes_sold) / smoothie_price
  smoothies_sold = 40 := by sorry

end scott_smoothie_sales_l614_61409


namespace r_upper_bound_r_7_upper_bound_l614_61480

/-- The maximum number of pieces that can be placed on an n × n chessboard
    without forming a rectangle with sides parallel to grid lines. -/
def r (n : ℕ) : ℕ := sorry

/-- Theorem: Upper bound for r(n) -/
theorem r_upper_bound (n : ℕ) : r n ≤ (n + n * Real.sqrt (4 * n - 3)) / 2 := by sorry

/-- Theorem: Upper bound for r(7) -/
theorem r_7_upper_bound : r 7 ≤ 21 := by sorry

end r_upper_bound_r_7_upper_bound_l614_61480


namespace salary_percent_increase_l614_61445

theorem salary_percent_increase 
  (x y : ℝ) (z : ℝ) 
  (hx : x > 0) 
  (hy : y ≥ 0) 
  (hz : z = (y / x) * 100) : 
  z = (y / x) * 100 := by
sorry

end salary_percent_increase_l614_61445


namespace fathers_full_time_jobs_l614_61440

theorem fathers_full_time_jobs (total_parents : ℝ) 
  (h1 : total_parents > 0) -- Ensure total_parents is positive
  (mothers_ratio : ℝ) 
  (h2 : mothers_ratio = 0.4) -- 40% of parents are mothers
  (mothers_full_time_ratio : ℝ) 
  (h3 : mothers_full_time_ratio = 3/4) -- 3/4 of mothers have full-time jobs
  (not_full_time_ratio : ℝ) 
  (h4 : not_full_time_ratio = 0.16) -- 16% of parents do not have full-time jobs
  : (total_parents * (1 - mothers_ratio) - 
     total_parents * (1 - not_full_time_ratio - mothers_ratio * mothers_full_time_ratio)) / 
    (total_parents * (1 - mothers_ratio)) = 9/10 := by
  sorry


end fathers_full_time_jobs_l614_61440


namespace geometric_sequence_minimum_value_l614_61442

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_third : a 3 = 2 * a 1 + a 2)
  (h_exist : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) :
  (∃ m n : ℕ, 1 / m + 4 / n = 3 / 2) ∧
  (∀ m n : ℕ, 1 / m + 4 / n ≥ 3 / 2) := by
  sorry

end geometric_sequence_minimum_value_l614_61442


namespace project_completion_time_l614_61430

/-- The number of days it takes A to complete the project alone -/
def a_days : ℝ := 10

/-- The number of days it takes B to complete the project alone -/
def b_days : ℝ := 30

/-- The number of days before project completion that A quits -/
def a_quit_days : ℝ := 10

/-- The total number of days to complete the project with A and B working together, with A quitting early -/
def total_days : ℝ := 15

theorem project_completion_time :
  let a_rate : ℝ := 1 / a_days
  let b_rate : ℝ := 1 / b_days
  (total_days - a_quit_days) * a_rate + total_days * b_rate = 1 :=
by sorry

end project_completion_time_l614_61430


namespace trig_equation_proof_l614_61447

theorem trig_equation_proof (x y : Real) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 4 := by
  sorry

end trig_equation_proof_l614_61447


namespace mathematics_arrangements_l614_61444

def word : String := "MATHEMATICS"

def is_vowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

def vowel_count (s : String) : Nat :=
  s.toList.filter is_vowel |>.length

def consonant_count (s : String) : Nat :=
  s.length - vowel_count s

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def multiset_permutations (total : Nat) (duplicates : List Nat) : Nat :=
  factorial total / (duplicates.map factorial |>.prod)

theorem mathematics_arrangements :
  let vowels := vowel_count word
  let consonants := consonant_count word
  let vowel_arrangements := multiset_permutations vowels [2]
  let consonant_arrangements := multiset_permutations consonants [2, 2]
  vowel_arrangements * consonant_arrangements = 15120 := by
  sorry

end mathematics_arrangements_l614_61444


namespace roots_equation_l614_61443

open Real

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := x^2 - 2 * cos θ * x + 1

theorem roots_equation (θ : ℝ) (α : ℝ) 
  (h1 : f θ (sin α) = 1/4 + cos θ) 
  (h2 : f θ (cos α) = 1/4 + cos θ) : 
  (tan α)^2 + 1 / tan α = (16 + 4 * sqrt 11) / 5 := by
  sorry

end roots_equation_l614_61443


namespace binomial_expansion_example_l614_61497

theorem binomial_expansion_example : 100 + 2 * (10 * 3) + 9 = (10 + 3)^2 := by
  sorry

end binomial_expansion_example_l614_61497


namespace no_valid_arrangement_l614_61450

def is_valid_arrangement (perm : List Nat) : Prop :=
  perm.length = 8 ∧
  (∀ n, n ∈ perm → n ∈ [1, 2, 3, 4, 5, 6, 8, 9]) ∧
  (∀ i, i < perm.length - 1 → (10 * perm[i]! + perm[i+1]!) % 7 = 0)

theorem no_valid_arrangement : ¬∃ perm : List Nat, is_valid_arrangement perm := by
  sorry

end no_valid_arrangement_l614_61450


namespace nine_digit_number_bounds_l614_61427

theorem nine_digit_number_bounds (A B : ℕ) : 
  (∃ C b : ℕ, B = 10 * C + b ∧ b < 10 ∧ A = 10^8 * b + C) →
  B > 22222222 →
  Nat.gcd B 18 = 1 →
  A ≥ 122222224 ∧ A ≤ 999999998 :=
by sorry

end nine_digit_number_bounds_l614_61427


namespace two_digit_number_puzzle_l614_61479

theorem two_digit_number_puzzle :
  ∀ x y : ℕ,
  (10 ≤ 10 * x + y) ∧ (10 * x + y < 100) →  -- two-digit number condition
  (x + y) * 3 = 10 * x + y - 2 →             -- puzzle condition
  x = 2 :=                                   -- conclusion
by
  sorry

end two_digit_number_puzzle_l614_61479


namespace number_of_girls_l614_61414

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of possible arrangements -/
def total_arrangements : ℕ := 2880

/-- A function that calculates the number of possible arrangements given the number of boys and girls -/
def calculate_arrangements (boys girls : ℕ) : ℕ :=
  Nat.factorial boys * Nat.factorial girls

/-- Theorem stating that there are 5 girls -/
theorem number_of_girls : ∃ (girls : ℕ), girls = 5 ∧ 
  calculate_arrangements num_boys girls = total_arrangements :=
sorry

end number_of_girls_l614_61414


namespace andreas_living_room_area_l614_61460

theorem andreas_living_room_area :
  ∀ (room_area : ℝ),
  (0.60 * room_area = 4 * 9) →
  room_area = 60 := by
  sorry

end andreas_living_room_area_l614_61460


namespace constant_function_sqrt_l614_61413

/-- Given a function f that is constant 3 for all real inputs, 
    prove that f(√x) + 1 = 4 for all non-negative real x -/
theorem constant_function_sqrt (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = 3) :
  ∀ x : ℝ, x ≥ 0 → f (Real.sqrt x) + 1 = 4 := by
  sorry

end constant_function_sqrt_l614_61413


namespace correct_weighted_mean_l614_61493

def total_values : ℕ := 30
def incorrect_mean : ℝ := 150
def first_error : ℝ := 135 - 165
def second_error : ℝ := 170 - 200
def weight_first_half : ℝ := 2
def weight_second_half : ℝ := 3

theorem correct_weighted_mean :
  let original_sum := incorrect_mean * total_values
  let total_error := first_error + second_error
  let corrected_sum := original_sum - total_error
  let total_weight := weight_first_half * (total_values / 2) + weight_second_half * (total_values / 2)
  corrected_sum / total_weight = 59.2 := by sorry

end correct_weighted_mean_l614_61493


namespace equation_solution_l614_61488

theorem equation_solution :
  ∀ x y : ℝ,
  y = 3 * x →
  (5 * y^2 + y + 10 = 2 * (9 * x^2 + y + 6)) ↔
  (x = 1/3 ∨ x = -2/9) :=
by
  sorry

end equation_solution_l614_61488


namespace centipede_human_ratio_theorem_l614_61420

/-- Represents the population of an island with centipedes, humans, and sheep. -/
structure IslandPopulation where
  centipedes : ℕ
  humans : ℕ
  sheep : ℕ

/-- The ratio of centipedes to humans on the island. -/
def centipede_human_ratio (pop : IslandPopulation) : ℚ :=
  pop.centipedes / pop.humans

/-- Theorem stating the ratio of centipedes to humans given the conditions. -/
theorem centipede_human_ratio_theorem (pop : IslandPopulation) 
  (h1 : pop.centipedes = 100)
  (h2 : pop.sheep = pop.humans / 2) :
  centipede_human_ratio pop = 100 / pop.humans := by
  sorry

end centipede_human_ratio_theorem_l614_61420


namespace area_between_parabola_and_line_l614_61439

-- Define the parabola function
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the line function
def line (x : ℝ) : ℝ := 2 * x

-- Theorem statement
theorem area_between_parabola_and_line :
  ∫ x in (0)..(1), (line x - parabola x) = 1/3 := by
  sorry

end area_between_parabola_and_line_l614_61439


namespace cone_base_radius_l614_61471

theorem cone_base_radius (surface_area : ℝ) (r : ℝ) : 
  surface_area = 12 * Real.pi ∧ 
  (∃ l : ℝ, l = 2 * r ∧ surface_area = Real.pi * r^2 + Real.pi * r * l) → 
  r = 2 := by
  sorry

end cone_base_radius_l614_61471


namespace least_four_digit_multiple_of_six_l614_61464

theorem least_four_digit_multiple_of_six : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧  -- four-digit number
  n % 6 = 0 ∧               -- multiple of 6
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 6 = 0) → n ≤ m) ∧ -- least such number
  n = 1002 := by
sorry

end least_four_digit_multiple_of_six_l614_61464

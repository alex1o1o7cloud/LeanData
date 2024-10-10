import Mathlib

namespace chris_current_age_l1029_102994

-- Define Praveen's current age
def praveen_age : ℝ := sorry

-- Define Chris's current age
def chris_age : ℝ := sorry

-- Condition 1: Praveen's age after 10 years is 3 times his age 3 years back
axiom praveen_age_condition : praveen_age + 10 = 3 * (praveen_age - 3)

-- Condition 2: Chris is 2 years younger than Praveen was 4 years ago
axiom chris_age_condition : chris_age = (praveen_age - 4) - 2

-- Theorem to prove
theorem chris_current_age : chris_age = 3.5 := by sorry

end chris_current_age_l1029_102994


namespace multiplicative_inverse_5_mod_31_l1029_102917

theorem multiplicative_inverse_5_mod_31 : ∃ x : ℤ, (5 * x) % 31 = 1 ∧ x % 31 = 25 := by
  sorry

end multiplicative_inverse_5_mod_31_l1029_102917


namespace degree_to_radian_conversion_l1029_102983

theorem degree_to_radian_conversion (angle_deg : ℝ) (angle_rad : ℝ) : 
  angle_deg = 15 → angle_rad = π / 12 := by
  sorry

end degree_to_radian_conversion_l1029_102983


namespace duck_cow_legs_heads_l1029_102905

theorem duck_cow_legs_heads :
  ∀ (D : ℕ),
  let C : ℕ := 16
  let H : ℕ := D + C
  let L : ℕ := 2 * D + 4 * C
  L - 2 * H = 32 :=
by
  sorry

end duck_cow_legs_heads_l1029_102905


namespace sin_a_less_cos_b_in_obtuse_triangle_l1029_102907

/-- In a triangle ABC where angle C is obtuse, sin A < cos B -/
theorem sin_a_less_cos_b_in_obtuse_triangle (A B C : ℝ) (h_triangle : A + B + C = π) (h_obtuse : C > π/2) : 
  Real.sin A < Real.cos B := by
sorry

end sin_a_less_cos_b_in_obtuse_triangle_l1029_102907


namespace root_magnitude_bound_l1029_102923

theorem root_magnitude_bound (p : ℝ) (r₁ r₂ : ℝ) 
  (h_distinct : r₁ ≠ r₂)
  (h_root₁ : r₁^2 + p*r₁ - 12 = 0)
  (h_root₂ : r₂^2 + p*r₂ - 12 = 0) :
  abs r₁ > 3 ∨ abs r₂ > 3 := by
sorry

end root_magnitude_bound_l1029_102923


namespace equation_to_general_form_l1029_102954

theorem equation_to_general_form :
  ∀ x : ℝ, 5 * x^2 - 2 * x = 3 * (x + 1) ↔ 5 * x^2 - 5 * x - 3 = 0 := by
sorry

end equation_to_general_form_l1029_102954


namespace thirty_seventh_digit_of_1_17_l1029_102962

-- Define the decimal representation of 1/17
def decimal_rep_1_17 : ℕ → ℕ
| 0 => 0
| 1 => 5
| 2 => 8
| 3 => 8
| 4 => 2
| 5 => 3
| 6 => 5
| 7 => 2
| 8 => 9
| 9 => 4
| 10 => 1
| 11 => 1
| 12 => 7
| 13 => 6
| 14 => 4
| 15 => 7
| n + 16 => decimal_rep_1_17 n

-- Define the period of the decimal representation
def period : ℕ := 16

-- Theorem statement
theorem thirty_seventh_digit_of_1_17 :
  decimal_rep_1_17 ((37 - 1) % period) = 8 := by
  sorry

end thirty_seventh_digit_of_1_17_l1029_102962


namespace homework_difference_l1029_102988

/-- The number of pages of reading homework -/
def reading_pages : ℕ := 6

/-- The number of pages of math homework -/
def math_pages : ℕ := 10

/-- The number of pages of science homework -/
def science_pages : ℕ := 3

/-- The number of pages of history homework -/
def history_pages : ℕ := 5

/-- The theorem states that the difference between math homework pages and the sum of reading, science, and history homework pages is -4 -/
theorem homework_difference : 
  (math_pages : ℤ) - (reading_pages + science_pages + history_pages : ℤ) = -4 := by
  sorry

end homework_difference_l1029_102988


namespace fraction_simplification_l1029_102903

theorem fraction_simplification :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 3) = 177 / 182 := by
  sorry

end fraction_simplification_l1029_102903


namespace sector_area_l1029_102939

/-- Given a sector with central angle 135° and arc length 3π cm, its area is 6π cm² -/
theorem sector_area (θ : ℝ) (arc_length : ℝ) (area : ℝ) :
  θ = 135 ∧ arc_length = 3 * Real.pi → area = 6 * Real.pi := by
  sorry

end sector_area_l1029_102939


namespace market_equilibrium_and_subsidy_effect_l1029_102906

/-- Supply function -/
def supply (p : ℝ) : ℝ := 2 + 8 * p

/-- Demand function (to be derived) -/
def demand (p : ℝ) : ℝ := 12 - 2 * p

/-- Equilibrium price -/
def equilibrium_price : ℝ := 1

/-- Equilibrium quantity -/
def equilibrium_quantity : ℝ := 10

/-- Subsidy amount -/
def subsidy : ℝ := 1

/-- New supply function after subsidy -/
def new_supply (p : ℝ) : ℝ := supply (p + subsidy)

/-- New equilibrium price after subsidy -/
def new_equilibrium_price : ℝ := 0.2

/-- New equilibrium quantity after subsidy -/
def new_equilibrium_quantity : ℝ := 11.6

theorem market_equilibrium_and_subsidy_effect :
  (demand 2 = 8) ∧
  (demand 3 = 6) ∧
  (supply equilibrium_price = demand equilibrium_price) ∧
  (supply equilibrium_price = equilibrium_quantity) ∧
  (new_supply new_equilibrium_price = demand new_equilibrium_price) ∧
  (new_equilibrium_quantity - equilibrium_quantity = 1.6) := by
  sorry

end market_equilibrium_and_subsidy_effect_l1029_102906


namespace parabola_coordinate_transform_l1029_102966

/-- Given a parabola y = 2x² in the original coordinate system,
    prove that its equation in a new coordinate system where
    the x-axis is moved up by 2 units and the y-axis is moved
    right by 2 units is y = 2(x+2)² - 2 -/
theorem parabola_coordinate_transform :
  ∀ (x y : ℝ),
  (y = 2 * x^2) →
  (∃ (x' y' : ℝ),
    x' = x + 2 ∧
    y' = y - 2 ∧
    y' = 2 * (x' - 2)^2 - 2) :=
by sorry

end parabola_coordinate_transform_l1029_102966


namespace problem_1_l1029_102981

theorem problem_1 : 2023 * 2023 - 2024 * 2022 = 1 := by
  sorry

end problem_1_l1029_102981


namespace sphere_surface_area_l1029_102902

/-- Given a sphere with volume 72π cubic inches, its surface area is 36π * 2^(2/3) square inches. -/
theorem sphere_surface_area (V : ℝ) (r : ℝ) (S : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  S = 4 * Real.pi * r^2 → 
  S = 36 * Real.pi * 2^(2/3) := by
  sorry

end sphere_surface_area_l1029_102902


namespace prime_equation_solution_l1029_102969

theorem prime_equation_solution :
  ∀ p q : ℕ,
  Prime p → Prime q →
  p^2 - 6*p*q + q^2 + 3*q - 1 = 0 →
  (p = 17 ∧ q = 3) :=
by sorry

end prime_equation_solution_l1029_102969


namespace subtracted_number_l1029_102990

theorem subtracted_number (x : ℕ) : 10000 - x = 9001 → x = 999 := by
  sorry

end subtracted_number_l1029_102990


namespace randolph_age_l1029_102997

/-- Proves that Randolph's age is 55 given the conditions of the problem -/
theorem randolph_age :
  (∀ (sherry sydney randolph : ℕ),
    randolph = sydney + 5 →
    sydney = 2 * sherry →
    sherry = 25 →
    randolph = 55) :=
by sorry

end randolph_age_l1029_102997


namespace simplify_expression_l1029_102957

theorem simplify_expression (m : ℝ) (h : m ≠ 3) : 
  (m^2 / (m-3)) + (9 / (3-m)) = m + 3 := by
  sorry

end simplify_expression_l1029_102957


namespace yellow_marbles_count_l1029_102985

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by
  sorry

end yellow_marbles_count_l1029_102985


namespace joans_remaining_apples_l1029_102978

/-- Given that Joan picked a certain number of apples and gave some away,
    this theorem proves how many apples Joan has left. -/
theorem joans_remaining_apples 
  (apples_picked : ℕ) 
  (apples_given_away : ℕ) 
  (h1 : apples_picked = 43)
  (h2 : apples_given_away = 27) :
  apples_picked - apples_given_away = 16 := by
sorry

end joans_remaining_apples_l1029_102978


namespace isabellas_hair_growth_l1029_102918

/-- Given Isabella's initial and final hair lengths, prove the amount of hair growth. -/
theorem isabellas_hair_growth 
  (initial_length : ℝ) 
  (final_length : ℝ) 
  (h1 : initial_length = 18) 
  (h2 : final_length = 24) : 
  final_length - initial_length = 6 := by
sorry

end isabellas_hair_growth_l1029_102918


namespace complement_A_intersect_B_A_subset_C_implies_a_geq_7_l1029_102901

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the first question
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Theorem for the second question
theorem A_subset_C_implies_a_geq_7 (a : ℝ) :
  A ⊆ C a → a ≥ 7 := by sorry

end complement_A_intersect_B_A_subset_C_implies_a_geq_7_l1029_102901


namespace p_necessary_not_sufficient_l1029_102947

-- Define rational numbers
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define the statements p and q
def p (x : ℝ) : Prop := IsRational (x^2)
def q (x : ℝ) : Prop := IsRational x

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient :
  (∀ x : ℝ, q x → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x) :=
sorry

end p_necessary_not_sufficient_l1029_102947


namespace three_numbers_sequence_l1029_102956

theorem three_numbers_sequence (x y z : ℝ) : 
  (x + y + z = 35 ∧ 
   2 * y = x + z + 1 ∧ 
   y^2 = (x + 3) * z) → 
  ((x = 15 ∧ y = 12 ∧ z = 8) ∨ 
   (x = 5 ∧ y = 12 ∧ z = 18)) := by
sorry

end three_numbers_sequence_l1029_102956


namespace max_infected_population_l1029_102964

/-- A graph representing the CMI infection spread --/
structure InfectionGraph where
  V : Type*  -- Set of vertices (people)
  E : V → V → Prop  -- Edge relation (friendship)
  degree_bound : ∀ v : V, (∃ n : ℕ, n ≤ 3 ∧ (∃ (l : List V), l.length = n ∧ (∀ u ∈ l, E v u)))

/-- The infection state of the graph over time --/
def InfectionState (G : InfectionGraph) := ℕ → G.V → Prop

/-- The initial infection state --/
def initial_infection (G : InfectionGraph) (S : InfectionState G) : Prop :=
  ∃ (infected : Finset G.V), infected.card = 2023 ∧ 
    ∀ v, S 0 v ↔ v ∈ infected

/-- The infection spread rule --/
def infection_rule (G : InfectionGraph) (S : InfectionState G) : Prop :=
  ∀ t v, S (t + 1) v ↔ 
    S t v ∨ (∃ (u₁ u₂ : G.V), u₁ ≠ u₂ ∧ G.E v u₁ ∧ G.E v u₂ ∧ S t u₁ ∧ S t u₂)

/-- Everyone eventually gets infected --/
def all_infected (G : InfectionGraph) (S : InfectionState G) : Prop :=
  ∀ v, ∃ t, S t v

/-- The main theorem --/
theorem max_infected_population (G : InfectionGraph) (S : InfectionState G) 
  (h_initial : initial_infection G S)
  (h_rule : infection_rule G S)
  (h_all : all_infected G S) :
  ∀ n : ℕ, (∃ f : G.V → Fin n, Function.Injective f) → n ≤ 4043 := by
  sorry

end max_infected_population_l1029_102964


namespace total_valid_words_count_l1029_102916

def alphabet_size : ℕ := 25
def max_word_length : ℕ := 5

def valid_words (n : ℕ) : ℕ :=
  if n < 2 then 0
  else if n = 2 then 2
  else Nat.choose n 2 * alphabet_size ^ (n - 2)

def total_valid_words : ℕ :=
  (List.range (max_word_length - 1)).map (fun i => valid_words (i + 2))
    |> List.sum

theorem total_valid_words_count :
  total_valid_words = 160075 := by sorry

end total_valid_words_count_l1029_102916


namespace inequality_count_l1029_102937

theorem inequality_count (x y a b : ℝ) (hx : |x| > a) (hy : |y| > b) :
  ∃! n : ℕ, n = (Bool.toNat (|x + y| > a + b)) +
               (Bool.toNat (|x - y| > |a - b|)) +
               (Bool.toNat (x * y > a * b)) +
               (Bool.toNat (|x / y| > |a / b|)) ∧
  n = 2 :=
sorry

end inequality_count_l1029_102937


namespace cube_volume_from_surface_area_l1029_102958

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 150 → volume = 125 → ∃ (side_length : ℝ), 
    surface_area = 6 * side_length^2 ∧ volume = side_length^3 := by
  sorry

end cube_volume_from_surface_area_l1029_102958


namespace jacob_excess_calories_l1029_102909

def jacob_calorie_problem (calorie_goal : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) : Prop :=
  calorie_goal < 1800 ∧
  breakfast = 400 ∧
  lunch = 900 ∧
  dinner = 1100 ∧
  (breakfast + lunch + dinner) - calorie_goal = 600

theorem jacob_excess_calories :
  ∃ (calorie_goal : ℕ), jacob_calorie_problem calorie_goal 400 900 1100 :=
by
  sorry

end jacob_excess_calories_l1029_102909


namespace parallel_vectors_implies_x_squared_two_l1029_102934

/-- Two vectors in R^2 are parallel if and only if their cross product is zero -/
axiom parallel_iff_cross_product_zero {a b : ℝ × ℝ} :
  (∃ k : ℝ, a = k • b) ↔ a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b in R^2, if they are parallel, then x^2 = 2 -/
theorem parallel_vectors_implies_x_squared_two (x : ℝ) :
  let a : ℝ × ℝ := (x + 2, 1 + x)
  let b : ℝ × ℝ := (x - 2, 1 - x)
  (∃ k : ℝ, a = k • b) → x^2 = 2 := by
  sorry

end parallel_vectors_implies_x_squared_two_l1029_102934


namespace notched_circle_distance_l1029_102972

/-- Given a circle with radius √75 and a point B such that there exist points A and C on the circle
    where AB = 8, BC = 2, and angle ABC is a right angle, prove that the square of the distance
    from B to the center of the circle is 122. -/
theorem notched_circle_distance (O A B C : ℝ × ℝ) : 
  (∀ P : ℝ × ℝ, (P.1 - O.1)^2 + (P.2 - O.2)^2 = 75 → P = A ∨ P = C) →  -- A and C are on the circle
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64 →  -- AB = 8
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4 →   -- BC = 2
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 →  -- Angle ABC is right angle
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = 122 := by
sorry


end notched_circle_distance_l1029_102972


namespace fifth_number_in_tenth_row_is_68_l1029_102970

/-- Represents a lattice with rows of consecutive integers -/
structure IntegerLattice where
  rowLength : ℕ
  rowCount : ℕ

/-- Gets the last number in a given row of the lattice -/
def lastNumberInRow (lattice : IntegerLattice) (row : ℕ) : ℤ :=
  (lattice.rowLength : ℤ) * row

/-- Gets the nth number in a given row of the lattice -/
def nthNumberInRow (lattice : IntegerLattice) (row : ℕ) (n : ℕ) : ℤ :=
  lastNumberInRow lattice row - (lattice.rowLength - n : ℤ)

/-- The theorem to be proved -/
theorem fifth_number_in_tenth_row_is_68 :
  let lattice : IntegerLattice := { rowLength := 7, rowCount := 10 }
  nthNumberInRow lattice 10 5 = 68 := by
  sorry

end fifth_number_in_tenth_row_is_68_l1029_102970


namespace hyperbola_equation_l1029_102965

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, focal length 10, and point (2,1) on its asymptote, 
    prove that its equation is x²/20 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (a^2 + b^2 = 25) → (2 * b / a = 1) → 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 20 - y^2 / 5 = 1) := by
sorry

end hyperbola_equation_l1029_102965


namespace smallest_four_digit_multiple_l1029_102991

theorem smallest_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  5 ∣ n ∧ 6 ∣ n ∧ 2 ∣ n ∧
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) → 5 ∣ m → 6 ∣ m → 2 ∣ m → m ≥ n) ∧
  n = 1020 :=
by sorry

end smallest_four_digit_multiple_l1029_102991


namespace set_operations_l1029_102968

/-- Given a universal set and two of its subsets, prove various set operations -/
theorem set_operations (U A B : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3})
  (hB : B = {2, 5}) :
  (U \ A = {2, 4, 5}) ∧
  (A ∩ B = ∅) ∧
  (A ∪ B = {1, 2, 3, 5}) ∧
  ((U \ A) ∪ (U \ B) = {1, 2, 3, 4, 5}) := by
  sorry

end set_operations_l1029_102968


namespace population_growth_inequality_l1029_102920

theorem population_growth_inequality (m n p : ℝ) 
  (h1 : (1 + p / 100)^2 = (1 + m / 100) * (1 + n / 100)) : 
  p ≤ (m + n) / 2 := by
sorry

end population_growth_inequality_l1029_102920


namespace parts_cost_calculation_l1029_102999

theorem parts_cost_calculation (total_amount : ℝ) (total_parts : ℕ) 
  (expensive_parts : ℕ) (expensive_cost : ℝ) :
  total_amount = 2380 →
  total_parts = 59 →
  expensive_parts = 40 →
  expensive_cost = 50 →
  ∃ (other_cost : ℝ),
    other_cost = 20 ∧
    total_amount = expensive_parts * expensive_cost + (total_parts - expensive_parts) * other_cost :=
by sorry

end parts_cost_calculation_l1029_102999


namespace abs_inequality_equivalence_abs_sum_inequality_equivalence_l1029_102948

-- Question 1
theorem abs_inequality_equivalence (x : ℝ) :
  |x - 2| < |x + 1| ↔ x > 1/2 := by sorry

-- Question 2
theorem abs_sum_inequality_equivalence (x : ℝ) :
  |2*x + 1| + |x - 2| > 4 ↔ x < -1 ∨ x > 1 := by sorry

end abs_inequality_equivalence_abs_sum_inequality_equivalence_l1029_102948


namespace cakes_sold_daily_l1029_102928

def cash_register_cost : ℕ := 1040
def bread_price : ℕ := 2
def bread_quantity : ℕ := 40
def cake_price : ℕ := 12
def daily_rent : ℕ := 20
def daily_electricity : ℕ := 2
def days_to_pay : ℕ := 8

def daily_bread_income : ℕ := bread_price * bread_quantity
def daily_expenses : ℕ := daily_rent + daily_electricity
def daily_profit_from_bread : ℕ := daily_bread_income - daily_expenses

theorem cakes_sold_daily (cakes_sold : ℕ) : 
  cakes_sold = 6 ↔ 
  days_to_pay * (daily_profit_from_bread + cake_price * cakes_sold) = cash_register_cost :=
by sorry

end cakes_sold_daily_l1029_102928


namespace no_rebus_solution_l1029_102910

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_to_nat (d : Fin 10) : ℕ := d.val

def rebus_equation (K U S Y : Fin 10) : Prop :=
  let KUSY := 1000 * (digit_to_nat K) + 100 * (digit_to_nat U) + 10 * (digit_to_nat S) + (digit_to_nat Y)
  let UKSY := 1000 * (digit_to_nat U) + 100 * (digit_to_nat K) + 10 * (digit_to_nat S) + (digit_to_nat Y)
  let UKSUS := 10000 * (digit_to_nat U) + 1000 * (digit_to_nat K) + 100 * (digit_to_nat S) + 10 * (digit_to_nat U) + (digit_to_nat S)
  is_four_digit KUSY ∧ is_four_digit UKSY ∧ is_four_digit UKSUS ∧ KUSY + UKSY = UKSUS

theorem no_rebus_solution :
  ∀ (K U S Y : Fin 10), K ≠ U ∧ K ≠ S ∧ K ≠ Y ∧ U ≠ S ∧ U ≠ Y ∧ S ≠ Y → ¬(rebus_equation K U S Y) :=
sorry

end no_rebus_solution_l1029_102910


namespace reader_group_size_l1029_102986

theorem reader_group_size (S L B : ℕ) (h1 : S = 250) (h2 : L = 230) (h3 : B = 80) :
  S + L - B = 400 := by
  sorry

end reader_group_size_l1029_102986


namespace union_of_M_and_N_l1029_102959

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | Real.log (x - 2) < 1}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | -1 < x ∧ x < 12} := by sorry

end union_of_M_and_N_l1029_102959


namespace geometric_progression_middle_term_l1029_102971

theorem geometric_progression_middle_term :
  ∀ m : ℝ,
  (∃ r : ℝ, (m / (5 + 2 * Real.sqrt 6) = r) ∧ ((5 - 2 * Real.sqrt 6) / m = r)) →
  (m = 1 ∨ m = -1) :=
λ m h => by sorry

end geometric_progression_middle_term_l1029_102971


namespace southAmericanStampsCost_l1029_102943

/-- Represents the number of stamps for a country in a specific decade -/
structure StampCount :=
  (fifties sixties seventies eighties : ℕ)

/-- Represents a country's stamp collection -/
structure Country :=
  (name : String)
  (price : ℚ)
  (counts : StampCount)

def colombia : Country :=
  { name := "Colombia"
  , price := 3 / 100
  , counts := { fifties := 7, sixties := 6, seventies := 12, eighties := 15 } }

def argentina : Country :=
  { name := "Argentina"
  , price := 6 / 100
  , counts := { fifties := 4, sixties := 8, seventies := 10, eighties := 9 } }

def southAmericanCountries : List Country := [colombia, argentina]

def stampsBefore1980s (c : Country) : ℕ :=
  c.counts.fifties + c.counts.sixties + c.counts.seventies

def totalCost (countries : List Country) : ℚ :=
  countries.map (fun c => (stampsBefore1980s c : ℚ) * c.price) |>.sum

theorem southAmericanStampsCost :
  totalCost southAmericanCountries = 207 / 100 := by sorry

end southAmericanStampsCost_l1029_102943


namespace sine_identity_l1029_102938

theorem sine_identity (α : Real) (h : α = π / 7) :
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := by
  sorry

end sine_identity_l1029_102938


namespace blackboard_erasers_l1029_102931

theorem blackboard_erasers (erasers_per_class : ℕ) 
                            (broken_erasers : ℕ) 
                            (remaining_erasers : ℕ) : 
  erasers_per_class = 3 →
  broken_erasers = 12 →
  remaining_erasers = 60 →
  (((remaining_erasers + broken_erasers) / erasers_per_class) / 3) + 12 = 20 := by
  sorry

end blackboard_erasers_l1029_102931


namespace strawberries_problem_l1029_102977

/-- Converts kilograms to grams -/
def kg_to_g (kg : ℕ) : ℕ := kg * 1000

/-- Calculates the remaining strawberries in grams -/
def remaining_strawberries (initial_kg initial_g given_kg given_g : ℕ) : ℕ :=
  (kg_to_g initial_kg + initial_g) - (kg_to_g given_kg + given_g)

theorem strawberries_problem :
  remaining_strawberries 3 300 1 900 = 1400 := by
  sorry

end strawberries_problem_l1029_102977


namespace band_encore_problem_l1029_102979

def band_encore_songs (total_songs : ℕ) (first_set : ℕ) (second_set : ℕ) (avg_third_fourth : ℕ) : ℕ :=
  total_songs - (first_set + second_set + 2 * avg_third_fourth)

theorem band_encore_problem :
  band_encore_songs 30 5 7 8 = 2 := by
  sorry

end band_encore_problem_l1029_102979


namespace intersections_for_12_6_l1029_102924

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  Nat.choose x_points 2 * Nat.choose y_points 2

/-- Theorem stating the maximum number of intersection points for 12 x-axis points and 6 y-axis points -/
theorem intersections_for_12_6 :
  max_intersections 12 6 = 990 := by
  sorry

end intersections_for_12_6_l1029_102924


namespace simplify_expression_l1029_102951

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = a^2 + b^2) :
  a^2 / b + b^2 / a - 1 / (a^2 * b^2) = (a^4 + 2*a*b + b^4 - 1) / (a * b) := by
  sorry

end simplify_expression_l1029_102951


namespace not_sum_of_six_odd_squares_l1029_102989

theorem not_sum_of_six_odd_squares (n : ℕ) : n = 1986 → ¬ ∃ (a b c d e f : ℕ), 
  (∃ (k₁ k₂ k₃ k₄ k₅ k₆ : ℕ), 
    a = 2 * k₁ + 1 ∧ 
    b = 2 * k₂ + 1 ∧ 
    c = 2 * k₃ + 1 ∧ 
    d = 2 * k₄ + 1 ∧ 
    e = 2 * k₅ + 1 ∧ 
    f = 2 * k₆ + 1) ∧ 
  n = a^2 + b^2 + c^2 + d^2 + e^2 + f^2 :=
by sorry

end not_sum_of_six_odd_squares_l1029_102989


namespace cell_phone_call_cost_l1029_102940

/-- Given a constant rate per minute where a 3-minute call costs $0.18, 
    prove that a 10-minute call will cost $0.60. -/
theorem cell_phone_call_cost 
  (rate : ℝ) 
  (h1 : rate * 3 = 0.18) -- Cost of 3-minute call
  : rate * 10 = 0.60 := by 
  sorry

end cell_phone_call_cost_l1029_102940


namespace quadratic_roots_sum_minus_product_l1029_102900

theorem quadratic_roots_sum_minus_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → x₂^2 - 3*x₂ + 2 = 0 → x₁ + x₂ - x₁ * x₂ = 1 := by
  sorry

end quadratic_roots_sum_minus_product_l1029_102900


namespace seeds_in_first_plot_l1029_102922

/-- The number of seeds planted in the first plot -/
def seeds_first_plot : ℕ := sorry

/-- The number of seeds planted in the second plot -/
def seeds_second_plot : ℕ := 200

/-- The percentage of seeds that germinated in the first plot -/
def germination_rate_first : ℚ := 20 / 100

/-- The percentage of seeds that germinated in the second plot -/
def germination_rate_second : ℚ := 35 / 100

/-- The percentage of total seeds that germinated -/
def total_germination_rate : ℚ := 26 / 100

/-- Theorem stating that the number of seeds in the first plot is 300 -/
theorem seeds_in_first_plot :
  (seeds_first_plot : ℚ) * germination_rate_first + 
  (seeds_second_plot : ℚ) * germination_rate_second = 
  total_germination_rate * ((seeds_first_plot : ℚ) + seeds_second_plot) ∧
  seeds_first_plot = 300 := by sorry

end seeds_in_first_plot_l1029_102922


namespace min_value_theorem_l1029_102933

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y - x*y = 0) :
  ∃ (min : ℝ), min = 5 + 2 * Real.sqrt 6 ∧ ∀ (z : ℝ), z = 3*x + 2*y → z ≥ min :=
sorry

end min_value_theorem_l1029_102933


namespace least_four_digit_divisible_by_digits_l1029_102912

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

def divisible_by_nonzero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 → n % d = 0

theorem least_four_digit_divisible_by_digits :
  ∃ (n : ℕ), is_four_digit n ∧
             all_digits_different n ∧
             divisible_by_nonzero_digits n ∧
             (∀ m : ℕ, is_four_digit m ∧
                       all_digits_different m ∧
                       divisible_by_nonzero_digits m →
                       n ≤ m) ∧
             n = 1240 :=
  sorry

end least_four_digit_divisible_by_digits_l1029_102912


namespace gcd_of_2535_5929_11629_l1029_102929

theorem gcd_of_2535_5929_11629 : Nat.gcd 2535 (Nat.gcd 5929 11629) = 1 := by
  sorry

end gcd_of_2535_5929_11629_l1029_102929


namespace flower_path_distance_l1029_102930

/-- Given eight equally spaced flowers along a straight path, 
    where the distance between the first and fifth flower is 80 meters, 
    prove that the distance between the first and last flower is 140 meters. -/
theorem flower_path_distance :
  ∀ (flower_positions : ℕ → ℝ),
    (∀ i j : ℕ, i < j → flower_positions j - flower_positions i = (j - i : ℝ) * (flower_positions 1 - flower_positions 0)) →
    (flower_positions 4 - flower_positions 0 = 80) →
    (flower_positions 7 - flower_positions 0 = 140) :=
by sorry

end flower_path_distance_l1029_102930


namespace primitive_cube_root_expression_l1029_102915

/-- ω is a primitive third root of unity -/
def ω : ℂ :=
  sorry

/-- ω is a primitive third root of unity -/
axiom ω_is_primitive_cube_root : ω^3 = 1 ∧ ω ≠ 1

/-- The value of the expression (1-ω)(1-ω^2)(1-ω^4)(1-ω^8) -/
theorem primitive_cube_root_expression : (1 - ω) * (1 - ω^2) * (1 - ω^4) * (1 - ω^8) = 9 :=
  sorry

end primitive_cube_root_expression_l1029_102915


namespace basketball_shots_improvement_l1029_102946

theorem basketball_shots_improvement (initial_shots : ℕ) (initial_success_rate : ℚ)
  (additional_shots : ℕ) (new_success_rate : ℚ) :
  initial_shots = 30 →
  initial_success_rate = 60 / 100 →
  additional_shots = 10 →
  new_success_rate = 62 / 100 →
  (↑(initial_shots * initial_success_rate.num / initial_success_rate.den +
    (new_success_rate * ↑(initial_shots + additional_shots)).num / (new_success_rate * ↑(initial_shots + additional_shots)).den -
    (initial_success_rate * ↑initial_shots).num / (initial_success_rate * ↑initial_shots).den) : ℚ) = 7 :=
by
  sorry

#check basketball_shots_improvement

end basketball_shots_improvement_l1029_102946


namespace orchestra_members_count_l1029_102950

theorem orchestra_members_count : ∃! n : ℕ, 
  130 < n ∧ n < 260 ∧ 
  n % 6 = 1 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 3 ∧
  n = 241 := by
sorry

end orchestra_members_count_l1029_102950


namespace pqr_product_l1029_102982

theorem pqr_product (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
  (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p)
  (h1 : p + 2 / q = q + 2 / r) (h2 : q + 2 / r = r + 2 / p) :
  |p * q * r| = 2 := by
  sorry

end pqr_product_l1029_102982


namespace complex_polygon_area_theorem_l1029_102908

/-- Represents a square sheet of paper -/
structure Sheet :=
  (side_length : ℝ)

/-- Represents the configuration of three overlapping sheets -/
structure SheetConfiguration :=
  (bottom : Sheet)
  (middle : Sheet)
  (top : Sheet)
  (middle_rotation : ℝ)
  (top_rotation : ℝ)
  (top_shift : ℝ)

/-- Calculates the area of the complex polygon formed by overlapping sheets -/
noncomputable def complex_polygon_area (config : SheetConfiguration) : ℝ :=
  sorry

/-- The main theorem stating the area of the complex polygon -/
theorem complex_polygon_area_theorem (config : SheetConfiguration) :
  config.bottom.side_length = 8 ∧
  config.middle.side_length = 8 ∧
  config.top.side_length = 8 ∧
  config.middle_rotation = 45 ∧
  config.top_rotation = 90 ∧
  config.top_shift = 4 →
  complex_polygon_area config = 144 :=
by sorry

end complex_polygon_area_theorem_l1029_102908


namespace translation_theorem_l1029_102996

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point to the left by a given distance -/
def translateLeft (p : Point) (d : ℝ) : Point :=
  { x := p.x - d, y := p.y }

/-- The theorem stating that translating point M(3, -4) 5 units to the left results in M'(-2, -4) -/
theorem translation_theorem :
  let M : Point := { x := 3, y := -4 }
  let M' : Point := translateLeft M 5
  M'.x = -2 ∧ M'.y = -4 := by sorry

end translation_theorem_l1029_102996


namespace problem_1_problem_2_l1029_102942

theorem problem_1 : |-5| + (3 - Real.sqrt 2) ^ 0 - 2 * Real.tan (π / 4) = 4 := by sorry

theorem problem_2 (a : ℝ) (h1 : a ≠ 3) (h2 : a ≠ -3) : 
  (a / (a^2 - 9)) / (1 + 3 / (a - 3)) = 1 / (a + 3) := by sorry

end problem_1_problem_2_l1029_102942


namespace exist_three_permuted_numbers_l1029_102974

/-- A function that checks if a number is a five-digit number in the decimal system -/
def isFiveDigit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A function that checks if two numbers are permutations of each other -/
def isPermutation (a b : ℕ) : Prop :=
  ∃ (digits_a digits_b : List ℕ),
    digits_a.length = 5 ∧
    digits_b.length = 5 ∧
    digits_a.toFinset = digits_b.toFinset ∧
    a = digits_a.foldl (fun acc d => acc * 10 + d) 0 ∧
    b = digits_b.foldl (fun acc d => acc * 10 + d) 0

/-- Theorem stating that there exist three five-digit numbers that are permutations of each other,
    where the sum of two equals twice the third -/
theorem exist_three_permuted_numbers :
  ∃ (a b c : ℕ),
    isFiveDigit a ∧ isFiveDigit b ∧ isFiveDigit c ∧
    isPermutation a b ∧ isPermutation b c ∧ isPermutation a c ∧
    a + b = 2 * c := by
  sorry

end exist_three_permuted_numbers_l1029_102974


namespace point_in_second_quadrant_l1029_102976

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  second_quadrant (-3) 2 := by sorry

end point_in_second_quadrant_l1029_102976


namespace total_price_is_530_l1029_102904

/-- The total price of hats given the number of hats, their prices, and the number of green hats. -/
def total_price (total_hats : ℕ) (blue_price green_price : ℕ) (green_hats : ℕ) : ℕ :=
  let blue_hats := total_hats - green_hats
  blue_price * blue_hats + green_price * green_hats

/-- Theorem stating that the total price of hats is $530 given the specific conditions. -/
theorem total_price_is_530 :
  total_price 85 6 7 20 = 530 :=
by sorry

end total_price_is_530_l1029_102904


namespace degree_to_radian_conversion_l1029_102961

theorem degree_to_radian_conversion (π : Real) (h : π = Real.pi) :
  120 * (π / 180) = 2 * π / 3 := by sorry

end degree_to_radian_conversion_l1029_102961


namespace intersection_A_B_intersection_A_complement_B_complement_union_A_B_l1029_102944

-- Define the universal set U
def U : Set ℝ := {x | x^2 - 3*x + 2 ≥ 0}

-- Define set A
def A : Set ℝ := {x | |x - 2| > 1}

-- Define set B
def B : Set ℝ := {x | (x - 1) / (x - 2) > 0}

-- Theorem statements
theorem intersection_A_B : A ∩ B = {x | x < 1 ∨ x > 3} := by sorry

theorem intersection_A_complement_B : A ∩ (U \ B) = ∅ := by sorry

theorem complement_union_A_B : U \ (A ∪ B) = {1, 2} := by sorry

end intersection_A_B_intersection_A_complement_B_complement_union_A_B_l1029_102944


namespace complex_equation_solution_l1029_102914

theorem complex_equation_solution (a : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : (1 + a*i)*i = 3 + i) : a = -3 := by
  sorry

end complex_equation_solution_l1029_102914


namespace solve_complex_equation_l1029_102932

theorem solve_complex_equation :
  let z : ℂ := 10 + 180 * Complex.I
  let equation := fun (x : ℂ) ↦ 7 * x - z = 15000
  ∃ (x : ℂ), equation x ∧ x = 2144 + (2 / 7) * Complex.I :=
by sorry

end solve_complex_equation_l1029_102932


namespace three_positions_from_six_people_l1029_102963

/-- The number of ways to choose 3 distinct positions from a group of n people -/
def choose_three_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- The theorem states that choosing 3 distinct positions from 6 people results in 120 ways -/
theorem three_positions_from_six_people :
  choose_three_positions 6 = 120 := by
  sorry

end three_positions_from_six_people_l1029_102963


namespace coin_division_theorem_l1029_102960

/-- Represents a collection of coins with their values -/
structure CoinCollection where
  num_coins : Nat
  coin_values : List Nat
  total_value : Nat

/-- Predicate to check if a coin collection can be divided into three equal groups -/
def can_divide_equally (cc : CoinCollection) : Prop :=
  ∃ (g1 g2 g3 : List Nat),
    g1 ++ g2 ++ g3 = cc.coin_values ∧
    g1.sum = g2.sum ∧ g2.sum = g3.sum

/-- Theorem stating that a specific coin collection can always be divided equally -/
theorem coin_division_theorem (cc : CoinCollection) 
    (h1 : cc.num_coins = 241)
    (h2 : cc.total_value = 360)
    (h3 : ∀ c ∈ cc.coin_values, c > 0)
    (h4 : cc.coin_values.length = cc.num_coins)
    (h5 : cc.coin_values.sum = cc.total_value) :
  can_divide_equally cc :=
sorry

end coin_division_theorem_l1029_102960


namespace intercept_ratio_l1029_102987

/-- Given two lines intersecting the y-axis at different points:
    - Line 1 has y-intercept 2, slope 5, and x-intercept (u, 0)
    - Line 2 has y-intercept 3, slope -7, and x-intercept (v, 0)
    The ratio of u to v is -14/15 -/
theorem intercept_ratio (u v : ℝ) : 
  (2 : ℝ) + 5 * u = 0 →  -- Line 1 equation at x-intercept
  (3 : ℝ) - 7 * v = 0 →  -- Line 2 equation at x-intercept
  u / v = -14 / 15 := by
  sorry

end intercept_ratio_l1029_102987


namespace imaginary_power_sum_l1029_102992

theorem imaginary_power_sum : Complex.I ^ 21 + Complex.I ^ 103 + Complex.I ^ 50 = -1 := by
  sorry

end imaginary_power_sum_l1029_102992


namespace equilateral_triangle_perimeter_l1029_102967

/-- The perimeter of an equilateral triangle with an inscribed circle of radius 2 cm -/
theorem equilateral_triangle_perimeter (r : ℝ) (h : r = 2) :
  let a := 2 * r * Real.sqrt 3
  3 * a = 12 * Real.sqrt 3 := by sorry

end equilateral_triangle_perimeter_l1029_102967


namespace discount_difference_is_187_point_5_l1029_102925

def initial_amount : ℝ := 15000

def single_discount_rate : ℝ := 0.3
def first_successive_discount_rate : ℝ := 0.25
def second_successive_discount_rate : ℝ := 0.05

def single_discount_amount : ℝ := initial_amount * (1 - single_discount_rate)

def successive_discount_amount : ℝ :=
  initial_amount * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)

theorem discount_difference_is_187_point_5 :
  successive_discount_amount - single_discount_amount = 187.5 := by
  sorry

end discount_difference_is_187_point_5_l1029_102925


namespace area_between_concentric_circles_l1029_102995

/-- Given two concentric circles where a chord of length 80 units is tangent to the smaller circle,
    the area between the two circles is equal to 1600π square units. -/
theorem area_between_concentric_circles (O : ℝ × ℝ) (r₁ r₂ : ℝ) (A B : ℝ × ℝ) :
  let circle₁ := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r₁^2}
  let circle₂ := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r₂^2}
  r₁ > r₂ →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 80^2 →
  ∃ P ∈ circle₂, (P.1 - A.1) * (B.1 - A.1) + (P.2 - A.2) * (B.2 - A.2) = 0 →
  π * (r₁^2 - r₂^2) = 1600 * π :=
by sorry

end area_between_concentric_circles_l1029_102995


namespace line_inclination_gt_45_deg_l1029_102998

/-- The angle of inclination of a line ax + (a + 1)y + 2 = 0 is greater than 45° if and only if a < -1/2 or a > 0 -/
theorem line_inclination_gt_45_deg (a : ℝ) :
  let line := {(x, y) : ℝ × ℝ | a * x + (a + 1) * y + 2 = 0}
  let angle_of_inclination := Real.arctan (abs (a / (a + 1)))
  angle_of_inclination > Real.pi / 4 ↔ a < -1/2 ∨ a > 0 :=
sorry

end line_inclination_gt_45_deg_l1029_102998


namespace pen_color_theorem_l1029_102941

-- Define the universe of pens
variable (Pen : Type)

-- Define the property of being in the box
variable (inBox : Pen → Prop)

-- Define the property of being blue
variable (isBlue : Pen → Prop)

-- Theorem statement
theorem pen_color_theorem :
  (¬ ∀ p : Pen, inBox p → isBlue p) →
  ((∃ p : Pen, inBox p ∧ ¬ isBlue p) ∧
   (¬ ∀ p : Pen, inBox p → isBlue p)) :=
by sorry

end pen_color_theorem_l1029_102941


namespace system_solution_product_l1029_102975

theorem system_solution_product : 
  ∃ (a b c d : ℚ),
    (4*a + 2*b + 6*c + 8*d = 48) ∧
    (2*(d+c) = b) ∧
    (4*b + 2*c = a) ∧
    (c + 2 = d) ∧
    (a * b * c * d = -88807680/4879681) := by
  sorry

end system_solution_product_l1029_102975


namespace unique_solution_abc_l1029_102955

theorem unique_solution_abc : 
  ∀ a b c : ℕ+, 
    (3 * a * b * c + 11 * (a + b + c) = 6 * (a * b + b * c + a * c) + 18) → 
    (a = 1 ∧ b = 2 ∧ c = 3) := by
  sorry

end unique_solution_abc_l1029_102955


namespace negation_equivalence_l1029_102913

-- Define the predicate P
def P (k : ℝ) : Prop := ∃ x y : ℝ, y = k * x + 1 ∧ x^2 + y^2 = 2

-- State the theorem
theorem negation_equivalence :
  (¬ ∀ k : ℝ, P k) ↔ (∃ k₀ : ℝ, ¬ P k₀) :=
sorry

end negation_equivalence_l1029_102913


namespace complex_power_magnitude_l1029_102936

theorem complex_power_magnitude : Complex.abs ((2 - 2 * Complex.I) ^ 6) = 512 := by
  sorry

end complex_power_magnitude_l1029_102936


namespace axels_alphabets_l1029_102911

theorem axels_alphabets (total_alphabets : ℕ) (repetitions : ℕ) (different_alphabets : ℕ) : 
  total_alphabets = 10 ∧ repetitions = 2 → different_alphabets = 5 :=
by sorry

end axels_alphabets_l1029_102911


namespace x_squared_greater_than_x_l1029_102984

theorem x_squared_greater_than_x (x : ℝ) :
  (x > 1 → x^2 > x) ∧ ¬(x^2 > x → x > 1) := by sorry

end x_squared_greater_than_x_l1029_102984


namespace max_ratio_inscribed_circumscribed_sphere_radii_l1029_102945

/-- Given a right square pyramid with circumscribed and inscribed spheres, 
    this theorem states the maximum ratio of their radii. -/
theorem max_ratio_inscribed_circumscribed_sphere_radii 
  (R r d : ℝ) 
  (h_positive : R > 0 ∧ r > 0)
  (h_relation : d^2 + (R + r)^2 = 2 * R^2) :
  ∃ (max_ratio : ℝ), max_ratio = Real.sqrt 2 - 1 ∧ 
    r / R ≤ max_ratio ∧ 
    ∃ (r' d' : ℝ), r' / R = max_ratio ∧ 
      d'^2 + (R + r')^2 = 2 * R^2 := by
sorry

end max_ratio_inscribed_circumscribed_sphere_radii_l1029_102945


namespace quadratic_trinomial_equality_l1029_102953

/-- A quadratic trinomial function -/
def quadratic_trinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_trinomial_equality 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x, quadratic_trinomial a b c (3.8 * x - 1) = quadratic_trinomial a b c (-3.8 * x)) →
  (∀ x, quadratic_trinomial a b c x = quadratic_trinomial a a c x) :=
sorry

end quadratic_trinomial_equality_l1029_102953


namespace arithmetic_geometric_mean_ratio_l1029_102927

theorem arithmetic_geometric_mean_ratio 
  (x y : ℝ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h_eq : ((x + y) / 2) + Real.sqrt (x * y) = y - x) : 
  x / y = 1 / 9 := by
sorry

end arithmetic_geometric_mean_ratio_l1029_102927


namespace factorization_equality_l1029_102949

theorem factorization_equality (a : ℝ) : (a + 2) * (a - 2) - 3 * a = (a - 4) * (a + 1) := by
  sorry

end factorization_equality_l1029_102949


namespace trigonometric_simplification_l1029_102919

theorem trigonometric_simplification :
  (Real.sin (15 * π / 180) + Real.sin (45 * π / 180)) /
  (Real.cos (15 * π / 180) + Real.cos (45 * π / 180)) =
  Real.tan (30 * π / 180) := by sorry

end trigonometric_simplification_l1029_102919


namespace multiplication_addition_equality_l1029_102980

theorem multiplication_addition_equality : 42 * 25 + 58 * 42 = 3486 := by
  sorry

end multiplication_addition_equality_l1029_102980


namespace red_peaches_count_l1029_102973

theorem red_peaches_count (yellow_peaches : ℕ) (red_yellow_difference : ℕ) 
  (h1 : yellow_peaches = 11)
  (h2 : red_yellow_difference = 8) :
  yellow_peaches + red_yellow_difference = 19 :=
by sorry

end red_peaches_count_l1029_102973


namespace lottery_winning_probability_l1029_102993

theorem lottery_winning_probability :
  let megaball_count : ℕ := 30
  let winnerball_count : ℕ := 50
  let chosen_winnerball_count : ℕ := 6

  let megaball_prob : ℚ := 1 / megaball_count
  let winnerball_prob : ℚ := 1 / (winnerball_count.choose chosen_winnerball_count)

  megaball_prob * winnerball_prob = 1 / 477621000 := by
  sorry

end lottery_winning_probability_l1029_102993


namespace negation_of_universal_proposition_l1029_102921

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℤ, x^3 < 1)) ↔ (∃ x : ℤ, x^3 ≥ 1) :=
by sorry

end negation_of_universal_proposition_l1029_102921


namespace min_value_n_minus_m_l1029_102935

open Real

noncomputable def f (x : ℝ) : ℝ := log (x / 2) + 1 / 2

noncomputable def g (x : ℝ) : ℝ := exp (x - 2)

theorem min_value_n_minus_m :
  (∀ m : ℝ, ∃ n : ℝ, n > 0 ∧ g m = f n) →
  (∃ m n : ℝ, n > 0 ∧ g m = f n ∧ n - m = log 2) ∧
  (∀ m n : ℝ, n > 0 → g m = f n → n - m ≥ log 2) :=
sorry

end min_value_n_minus_m_l1029_102935


namespace min_value_x_plus_y_l1029_102926

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y = 1/x + 4/y + 8) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + b = 1/a + 4/b + 8 → x + y ≤ a + b :=
by sorry

end min_value_x_plus_y_l1029_102926


namespace quadratic_solution_l1029_102952

/-- The quadratic equation ax^2 + 10x + c = 0 has exactly one solution, a + c = 12, and a < c -/
def quadratic_equation (a c : ℝ) : Prop :=
  ∃! x, a * x^2 + 10 * x + c = 0 ∧ a + c = 12 ∧ a < c

/-- The solution to the quadratic equation is (6-√11, 6+√11) -/
theorem quadratic_solution :
  ∀ a c : ℝ, quadratic_equation a c → a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11 := by
sorry

end quadratic_solution_l1029_102952

import Mathlib

namespace solve_equation_l3669_366913

theorem solve_equation : ∃ x : ℚ, 5 * (x - 9) = 6 * (3 - 3 * x) + 6 ∧ x = 3 := by
  sorry

end solve_equation_l3669_366913


namespace prob_reach_opposite_after_six_moves_l3669_366918

/-- Represents a cube with its vertices and edges. -/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)

/-- Represents a bug's movement on the cube. -/
structure BugMovement (cube : Cube) where
  start_vertex : Fin 8
  num_moves : Nat
  prob_each_edge : ℝ

/-- The probability of the bug reaching the opposite vertex after a specific number of moves. -/
def prob_reach_opposite (cube : Cube) (movement : BugMovement cube) : ℝ :=
  sorry

/-- Theorem stating that the probability of reaching the opposite vertex after six moves is 1/8. -/
theorem prob_reach_opposite_after_six_moves (cube : Cube) (movement : BugMovement cube) :
  movement.num_moves = 6 →
  movement.prob_each_edge = 1/3 →
  prob_reach_opposite cube movement = 1/8 :=
sorry

end prob_reach_opposite_after_six_moves_l3669_366918


namespace triangle_existence_l3669_366902

/-- A triangle with sides x, 10 + x, and 24 can exist if and only if x is a positive integer and x ≥ 34. -/
theorem triangle_existence (x : ℕ) : 
  (∃ (a b c : ℝ), a = x ∧ b = x + 10 ∧ c = 24 ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a) ↔ 
  x ≥ 34 := by
  sorry

#check triangle_existence

end triangle_existence_l3669_366902


namespace equation_solutions_l3669_366914

noncomputable def fourthRoot (x : ℝ) : ℝ := Real.rpow x (1/4)

theorem equation_solutions :
  let f : ℝ → ℝ := λ x => fourthRoot (53 - 3*x) + fourthRoot (29 + x)
  ∀ x : ℝ, f x = 4 ↔ x = 2 ∨ x = 16 :=
by sorry

end equation_solutions_l3669_366914


namespace vegetable_production_equation_l3669_366916

def vegetable_growth_rate (initial_production final_production : ℝ) (years : ℕ) (x : ℝ) : Prop :=
  initial_production * (1 + x) ^ years = final_production

theorem vegetable_production_equation :
  ∃ x : ℝ, vegetable_growth_rate 800 968 2 x :=
sorry

end vegetable_production_equation_l3669_366916


namespace monica_reading_plan_l3669_366950

/-- The number of books Monica read last year -/
def books_last_year : ℕ := 25

/-- The number of books Monica read this year -/
def books_this_year : ℕ := 3 * books_last_year

/-- The number of books Monica will read next year -/
def books_next_year : ℕ := 3 * books_this_year + 7

theorem monica_reading_plan : books_next_year = 232 := by
  sorry

end monica_reading_plan_l3669_366950


namespace divide_by_repeating_decimal_l3669_366934

theorem divide_by_repeating_decimal : 
  let x : ℚ := 142857 / 999999
  7 / x = 49 := by sorry

end divide_by_repeating_decimal_l3669_366934


namespace unique_y_exists_l3669_366998

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y - 3

-- Theorem statement
theorem unique_y_exists : ∃! y : ℝ, star 4 y = 17 := by
  sorry

end unique_y_exists_l3669_366998


namespace age_ratio_seven_years_ago_l3669_366930

-- Define the present ages of Henry and Jill
def henry_present_age : ℕ := 25
def jill_present_age : ℕ := 16

-- Define the sum of their present ages
def sum_present_ages : ℕ := henry_present_age + jill_present_age

-- Define their ages 7 years ago
def henry_past_age : ℕ := henry_present_age - 7
def jill_past_age : ℕ := jill_present_age - 7

-- Define the theorem
theorem age_ratio_seven_years_ago :
  sum_present_ages = 41 →
  ∃ k : ℕ, henry_past_age = k * jill_past_age →
  henry_past_age / jill_past_age = 2 :=
by sorry

end age_ratio_seven_years_ago_l3669_366930


namespace line_parameterization_l3669_366990

/-- Given a line y = 2x + 5 parameterized as (x, y) = (r, -3) + t(5, k),
    prove that r = -4 and k = 10 -/
theorem line_parameterization (r k : ℝ) : 
  (∀ t x y : ℝ, x = r + 5*t ∧ y = -3 + k*t → y = 2*x + 5) →
  r = -4 ∧ k = 10 := by
sorry

end line_parameterization_l3669_366990


namespace bottle_caps_found_l3669_366937

theorem bottle_caps_found (earlier_total current_total : ℕ) 
  (h1 : earlier_total = 25) 
  (h2 : current_total = 32) : 
  current_total - earlier_total = 7 := by
  sorry

end bottle_caps_found_l3669_366937


namespace factorial_sum_equals_35906_l3669_366942

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_sum_equals_35906 : 
  7 * factorial 7 + 5 * factorial 5 + 3 * factorial 3 + 2 * (factorial 2)^2 = 35906 := by
  sorry

end factorial_sum_equals_35906_l3669_366942


namespace same_color_probability_l3669_366941

/-- The probability of drawing two balls of the same color from a bag containing green and white balls. -/
theorem same_color_probability (green white : ℕ) (h : green = 10 ∧ white = 8) :
  let total := green + white
  let prob_green := (green * (green - 1)) / (total * (total - 1))
  let prob_white := (white * (white - 1)) / (total * (total - 1))
  (prob_green + prob_white : ℚ) = 73 / 153 :=
by sorry

end same_color_probability_l3669_366941


namespace camel_inheritance_theorem_l3669_366967

theorem camel_inheritance_theorem :
  let total_camels : ℕ := 17
  let eldest_share : ℚ := 1/2
  let middle_share : ℚ := 1/3
  let youngest_share : ℚ := 1/9
  eldest_share + middle_share + youngest_share = 17/18 := by
  sorry

end camel_inheritance_theorem_l3669_366967


namespace chromatic_number_bound_l3669_366904

/-- A graph G is represented by its vertex set and edge set. -/
structure Graph (V : Type) where
  edges : Set (V × V)

/-- The chromatic number of a graph. -/
def chromaticNumber {V : Type} (G : Graph V) : ℕ :=
  sorry

/-- The number of edges in a graph. -/
def numEdges {V : Type} (G : Graph V) : ℕ :=
  sorry

/-- Theorem: The chromatic number of a graph is bounded by a function of its edge count. -/
theorem chromatic_number_bound {V : Type} (G : Graph V) :
  (chromaticNumber G : ℝ) ≤ 1/2 + Real.sqrt (2 * (numEdges G : ℝ) + 1/4) :=
sorry

end chromatic_number_bound_l3669_366904


namespace boys_without_calculators_l3669_366996

/-- Given a class with boys and girls, and information about calculator possession,
    prove that the number of boys without calculators is 5. -/
theorem boys_without_calculators
  (total_boys : ℕ)
  (total_with_calc : ℕ)
  (girls_with_calc : ℕ)
  (h1 : total_boys = 20)
  (h2 : total_with_calc = 30)
  (h3 : girls_with_calc = 15) :
  total_boys - (total_with_calc - girls_with_calc) = 5 := by
  sorry

end boys_without_calculators_l3669_366996


namespace max_value_x_plus_y_squared_l3669_366938

/-- Given real numbers x and y satisfying 3(x^3 + y^3) = x + y^2,
    the maximum value of x + y^2 is 1/3. -/
theorem max_value_x_plus_y_squared (x y : ℝ) 
  (h : 3 * (x^3 + y^3) = x + y^2) : 
  ∃ (M : ℝ), M = 1/3 ∧ ∀ (a b : ℝ), 3 * (a^3 + b^3) = a + b^2 → a + b^2 ≤ M :=
by sorry

end max_value_x_plus_y_squared_l3669_366938


namespace annie_crayons_l3669_366911

theorem annie_crayons (initial : ℕ) (given : ℕ) (final : ℕ) : 
  given = 36 → final = 40 → initial = 4 := by sorry

end annie_crayons_l3669_366911


namespace consecutive_integers_square_sum_l3669_366906

theorem consecutive_integers_square_sum (e f g h : ℤ) : 
  (e + 1 = f) → (f + 1 = g) → (g + 1 = h) →
  (e < f) → (f < g) → (g < h) →
  (e^2 + h^2 = 3405) →
  (f^2 * g^2 = 2689600) := by
sorry

end consecutive_integers_square_sum_l3669_366906


namespace ruby_count_l3669_366940

theorem ruby_count (total_gems diamonds : ℕ) (h1 : total_gems = 5155) (h2 : diamonds = 45) :
  total_gems - diamonds = 5110 :=
by sorry

end ruby_count_l3669_366940


namespace min_participants_l3669_366931

/-- Represents a participant in the race -/
structure Participant where
  name : String
  position : Nat

/-- Represents the race -/
structure Race where
  participants : List Participant
  /-- No two participants finished simultaneously -/
  no_ties : ∀ p1 p2 : Participant, p1 ∈ participants → p2 ∈ participants → p1 ≠ p2 → p1.position ≠ p2.position

/-- The number of people who finished before a given participant -/
def finished_before (race : Race) (p : Participant) : Nat :=
  (race.participants.filter (fun q => q.position < p.position)).length

/-- The number of people who finished after a given participant -/
def finished_after (race : Race) (p : Participant) : Nat :=
  (race.participants.filter (fun q => q.position > p.position)).length

/-- The theorem stating the minimum number of participants in the race -/
theorem min_participants (race : Race) 
  (andrei dima lenya : Participant)
  (andrei_in : andrei ∈ race.participants)
  (dima_in : dima ∈ race.participants)
  (lenya_in : lenya ∈ race.participants)
  (andrei_cond : finished_before race andrei = (finished_after race andrei) / 2)
  (dima_cond : finished_before race dima = (finished_after race dima) / 3)
  (lenya_cond : finished_before race lenya = (finished_after race lenya) / 4) :
  race.participants.length ≥ 61 := by
  sorry

end min_participants_l3669_366931


namespace e_opposite_x_l3669_366960

/-- Represents the faces of a cube --/
inductive Face : Type
  | X | A | B | C | D | E

/-- Represents the net of a cube --/
structure CubeNet where
  central : Face
  left : Face
  right : Face
  bottom : Face
  connected_to_right : Face
  connected_to_left : Face

/-- Defines the specific cube net given in the problem --/
def given_net : CubeNet :=
  { central := Face.X
  , left := Face.A
  , right := Face.B
  , bottom := Face.D
  , connected_to_right := Face.C
  , connected_to_left := Face.E
  }

/-- Defines the concept of opposite faces in a cube --/
def opposite (f1 f2 : Face) : Prop := sorry

/-- Theorem stating that in the given net, E is opposite to X --/
theorem e_opposite_x (net : CubeNet) : 
  net = given_net → opposite Face.E Face.X :=
sorry

end e_opposite_x_l3669_366960


namespace rectangle_EF_length_l3669_366978

/-- Rectangle ABCD with given properties -/
structure Rectangle where
  AB : ℝ
  BC : ℝ
  DE : ℝ
  DF : ℝ
  EF : ℝ
  h_AB : AB = 4
  h_BC : BC = 10
  h_DE_DF : DE = DF
  h_area : DE * DF / 2 = AB * BC / 4

/-- The length of EF in the given rectangle -/
theorem rectangle_EF_length (r : Rectangle) : r.EF = 2 * Real.sqrt 10 := by
  sorry

end rectangle_EF_length_l3669_366978


namespace constant_sum_zero_l3669_366976

theorem constant_sum_zero (a b : ℝ) : 
  (∀ x y : ℝ, y = a + b / (x + 1)) →
  (2 = a + b / (1 + 1)) →
  (3 = a + b / (3 + 1)) →
  a + b = 0 := by sorry

end constant_sum_zero_l3669_366976


namespace ellipse_and_tangent_line_l3669_366925

/-- Given an ellipse and a line passing through its vertex and focus, 
    prove the standard equation of the ellipse and its tangent line equation. -/
theorem ellipse_and_tangent_line 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (ellipse : ℝ → ℝ → Prop) 
  (line : ℝ → ℝ → Prop) 
  (h_ellipse : ellipse = λ x y => x^2/a^2 + y^2/b^2 = 1)
  (h_line : line = λ x y => Real.sqrt 6 * x + 2 * y - 2 * Real.sqrt 6 = 0)
  (h_vertex_focus : ∃ (E F : ℝ × ℝ), 
    ellipse E.1 E.2 ∧ 
    ellipse F.1 F.2 ∧ 
    line E.1 E.2 ∧ 
    line F.1 F.2 ∧ 
    (E.1 = 0 ∧ E.2 = Real.sqrt 6) ∧ 
    (F.1 = 2 ∧ F.2 = 0)) :
  (∀ x y, ellipse x y ↔ x^2/10 + y^2/6 = 1) ∧
  (∀ x y, (Real.sqrt 5 / 10) * x + (Real.sqrt 3 / 6) * y = 1 →
    (x = Real.sqrt 5 ∧ y = Real.sqrt 3) ∨
    ¬(ellipse x y)) := by
  sorry

end ellipse_and_tangent_line_l3669_366925


namespace charles_pictures_l3669_366971

theorem charles_pictures (total_papers : ℕ) (drawn_today : ℕ) (drawn_yesterday_before : ℕ) (papers_left : ℕ) : 
  total_papers = 20 →
  drawn_today = 6 →
  drawn_yesterday_before = 6 →
  papers_left = 2 →
  total_papers - (drawn_today + drawn_yesterday_before) - papers_left = 6 :=
by sorry

end charles_pictures_l3669_366971


namespace sixteen_triangles_l3669_366943

/-- A right triangle with integer leg lengths and hypotenuse b + 3 --/
structure RightTriangle where
  a : ℕ
  b : ℕ
  hyp_eq : a^2 + b^2 = (b + 3)^2
  b_bound : b < 200

/-- The count of right triangles satisfying the given conditions --/
def count_triangles : ℕ := sorry

/-- Theorem stating that there are exactly 16 right triangles satisfying the conditions --/
theorem sixteen_triangles : count_triangles = 16 := by sorry

end sixteen_triangles_l3669_366943


namespace no_three_integers_divisibility_l3669_366951

theorem no_three_integers_divisibility : ¬∃ (x y z : ℤ), 
  x > 1 ∧ y > 1 ∧ z > 1 ∧
  (y ∣ (x^2 - 1)) ∧ (z ∣ (x^2 - 1)) ∧
  (x ∣ (y^2 - 1)) ∧ (z ∣ (y^2 - 1)) ∧
  (x ∣ (z^2 - 1)) ∧ (y ∣ (z^2 - 1)) :=
by sorry

end no_three_integers_divisibility_l3669_366951


namespace geometric_series_sum_l3669_366949

/-- Sum of a finite geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- First term of the geometric series -/
def a : ℚ := 2

/-- Common ratio of the geometric series -/
def r : ℚ := 2/5

/-- Number of terms in the series -/
def n : ℕ := 5

theorem geometric_series_sum :
  geometric_sum a r n = 2062/375 := by
  sorry

end geometric_series_sum_l3669_366949


namespace angle_300_shares_terminal_side_with_neg_60_l3669_366901

-- Define the concept of angles sharing the same terminal side
def shares_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = k * 360 - α

-- Theorem statement
theorem angle_300_shares_terminal_side_with_neg_60 :
  shares_terminal_side (-60) 300 := by
  sorry

end angle_300_shares_terminal_side_with_neg_60_l3669_366901


namespace carrot_cost_theorem_l3669_366929

/-- Calculates the total cost of carrots for a year given the daily consumption, carrots per bag, and cost per bag. -/
theorem carrot_cost_theorem (carrots_per_day : ℕ) (carrots_per_bag : ℕ) (cost_per_bag : ℚ) :
  carrots_per_day = 1 →
  carrots_per_bag = 5 →
  cost_per_bag = 2 →
  (365 * carrots_per_day / carrots_per_bag : ℚ).ceil * cost_per_bag = 146 := by
  sorry

#eval (365 * 1 / 5 : ℚ).ceil * 2

end carrot_cost_theorem_l3669_366929


namespace infinite_solutions_l3669_366974

/-- Standard prime factorization of a positive integer -/
def prime_factorization (n : ℕ+) : List (ℕ × ℕ) := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ+) : ℕ := 
  let factors := prime_factorization n
  (factors.map (fun (p, α) => α)).prod * 
  (factors.map (fun (p, α) => p^(α - 1))).prod

/-- The set of positive integers n satisfying f(n+1) = f(n) + 1 -/
def S : Set ℕ+ := {n | f (n + 1) = f n + 1}

/-- The main theorem to be proved -/
theorem infinite_solutions : Set.Infinite S := by sorry

end infinite_solutions_l3669_366974


namespace exists_multiple_irreducible_representations_l3669_366936

/-- The set V_n for a given n > 2 -/
def V_n (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ k : ℕ, m = 1 + k * n}

/-- A number is irreducible in V_n if it cannot be expressed as a product of two numbers in V_n -/
def irreducible_in_V_n (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → p * q ≠ m

/-- The main theorem -/
theorem exists_multiple_irreducible_representations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ (irreducibles1 irreducibles2 : List ℕ),
      irreducibles1 ≠ irreducibles2 ∧
      (∀ x ∈ irreducibles1, irreducible_in_V_n n x) ∧
      (∀ x ∈ irreducibles2, irreducible_in_V_n n x) ∧
      (irreducibles1.prod = r) ∧
      (irreducibles2.prod = r) :=
sorry

end exists_multiple_irreducible_representations_l3669_366936


namespace special_arithmetic_sequence_101st_term_l3669_366973

/-- An arithmetic sequence where the square of each term equals the sum of the first 2n-1 terms. -/
def SpecialArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≠ 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  (∀ n, (a n)^2 = (2 * n - 1) * (a 1 + a n) / 2)

theorem special_arithmetic_sequence_101st_term
  (a : ℕ → ℝ) (h : SpecialArithmeticSequence a) : a 101 = 201 := by
  sorry

end special_arithmetic_sequence_101st_term_l3669_366973


namespace total_honey_production_total_honey_is_1060_l3669_366903

/-- Calculates the total honey production for two bee hives with given characteristics -/
theorem total_honey_production
  (hive1_bees : ℕ)
  (hive1_honey : ℝ)
  (hive2_bee_reduction : ℝ)
  (hive2_honey_increase : ℝ)
  (h1 : hive1_bees = 1000)
  (h2 : hive1_honey = 500)
  (h3 : hive2_bee_reduction = 0.2)
  (h4 : hive2_honey_increase = 0.4)
  : ℝ :=
by
  -- The proof goes here
  sorry

#check total_honey_production

/-- The total honey production is 1060 liters -/
theorem total_honey_is_1060 :
  total_honey_production 1000 500 0.2 0.4 rfl rfl rfl rfl = 1060 :=
by
  -- The proof goes here
  sorry

end total_honey_production_total_honey_is_1060_l3669_366903


namespace zoo_visitors_l3669_366961

/-- Given the number of visitors on Friday and the ratio of Saturday visitors to Friday visitors,
    prove that the number of visitors on Saturday is equal to the product of the Friday visitors and the ratio. -/
theorem zoo_visitors (friday_visitors : ℕ) (saturday_ratio : ℕ) :
  let saturday_visitors := friday_visitors * saturday_ratio
  saturday_visitors = friday_visitors * saturday_ratio :=
by sorry

/-- Example with the given values -/
example : 
  let friday_visitors : ℕ := 1250
  let saturday_ratio : ℕ := 3
  let saturday_visitors := friday_visitors * saturday_ratio
  saturday_visitors = 3750 :=
by sorry

end zoo_visitors_l3669_366961


namespace rent_increase_percentage_l3669_366981

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.15 * last_year_earnings
  let this_year_rent := 0.25 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 143.75 := by
sorry

end rent_increase_percentage_l3669_366981


namespace largest_power_of_five_factor_l3669_366955

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials : ℕ := factorial 102 + factorial 103 + factorial 104 + factorial 105

theorem largest_power_of_five_factor : 
  (∀ m : ℕ, 5^(24 + 1) ∣ sum_of_factorials → 5^m ∣ sum_of_factorials) ∧ 
  5^24 ∣ sum_of_factorials :=
sorry

end largest_power_of_five_factor_l3669_366955


namespace algebraic_simplification_l3669_366915

theorem algebraic_simplification (a b : ℝ) : 
  14 * a^8 * b^4 / (7 * a^4 * b^4) - a^3 * a - (2 * a^2)^2 = -3 * a^4 := by
  sorry

end algebraic_simplification_l3669_366915


namespace triangle_problem_l3669_366910

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : 1 + (Real.tan t.A / Real.tan t.B) = 2 * t.c / t.b)
  (h2 : t.a = Real.sqrt 3) :
  t.A = π/3 ∧ 
  (∀ (t' : Triangle), t'.a = Real.sqrt 3 → t'.b * t'.c ≤ t.b * t.c → 
    t.b = t.c ∧ t.b = Real.sqrt 3) :=
by sorry

end triangle_problem_l3669_366910


namespace sum_of_ages_l3669_366995

theorem sum_of_ages (a b c : ℕ) : 
  a = 11 → 
  (a - 3) + (b - 3) + (c - 3) = 6 * (a - 3) → 
  a + b + c = 57 := by
sorry

end sum_of_ages_l3669_366995


namespace negation_of_universal_statement_l3669_366926

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^2 - x < 0) ↔
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - x ≥ 0) :=
by sorry

end negation_of_universal_statement_l3669_366926


namespace expenditure_estimate_l3669_366992

/-- Represents the annual income in billions of yuan -/
def annual_income : ℝ := 15

/-- Represents the relationship between income x and expenditure y -/
def expenditure_function (x : ℝ) : ℝ := 0.8 * x + 0.1

/-- The estimated annual expenditure based on the given income and relationship -/
def estimated_expenditure : ℝ := expenditure_function annual_income

theorem expenditure_estimate : estimated_expenditure = 12.1 := by
  sorry

end expenditure_estimate_l3669_366992


namespace cake_recipe_flour_l3669_366979

/-- The amount of flour required for Mary's cake recipe --/
def flour_required (sugar : ℕ) (flour_sugar_diff : ℕ) (flour_added : ℕ) : ℕ :=
  sugar + flour_sugar_diff

theorem cake_recipe_flour :
  let sugar := 3
  let flour_sugar_diff := 5
  let flour_added := 2
  flour_required sugar flour_sugar_diff flour_added = 8 := by
  sorry

end cake_recipe_flour_l3669_366979


namespace andrey_stamps_l3669_366965

theorem andrey_stamps :
  ∃ (x : ℕ), 
    x % 3 = 1 ∧ 
    x % 5 = 3 ∧ 
    x % 7 = 5 ∧ 
    150 < x ∧ 
    x ≤ 300 ∧ 
    x = 208 := by
  sorry

end andrey_stamps_l3669_366965


namespace equation_solution_l3669_366944

theorem equation_solution : ∃ x : ℝ, 45 - (x - (37 - (15 - 19))) = 58 ∧ x = 28 := by
  sorry

end equation_solution_l3669_366944


namespace red_peaches_count_l3669_366985

/-- The number of baskets of peaches -/
def num_baskets : ℕ := 6

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 16

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 18

/-- The total number of red peaches in all baskets -/
def total_red_peaches : ℕ := num_baskets * red_peaches_per_basket

theorem red_peaches_count : total_red_peaches = 96 := by
  sorry

end red_peaches_count_l3669_366985


namespace fraction_sum_simplification_l3669_366984

theorem fraction_sum_simplification (a b : ℝ) (h : a ≠ b) :
  a^2 / (a - b) + (2*a*b - b^2) / (b - a) = a - b := by
  sorry

end fraction_sum_simplification_l3669_366984


namespace quadratic_equation_roots_l3669_366972

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x - 6 = 0 ∧ x = 3) → 
  (∃ y : ℝ, y^2 - m*y - 6 = 0 ∧ y = -2) ∧ m = 1 := by
sorry

end quadratic_equation_roots_l3669_366972


namespace root_sum_reciprocal_products_l3669_366957

theorem root_sum_reciprocal_products (p q r s : ℂ) : 
  (p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0) →
  (q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0) →
  (r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0) →
  (s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3 := by
sorry

end root_sum_reciprocal_products_l3669_366957


namespace tangent_circle_position_l3669_366905

/-- Represents a trapezoid EFGH with a circle tangent to two sides --/
structure TrapezoidWithTangentCircle where
  -- Lengths of the trapezoid sides
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  -- Q is the center of the circle on EF
  EQ : ℝ
  -- Assumption that EF is parallel to GH is implicit in the structure

/-- The main theorem about the tangent circle in a specific trapezoid --/
theorem tangent_circle_position 
  (t : TrapezoidWithTangentCircle)
  (h1 : t.EF = 86)
  (h2 : t.FG = 60)
  (h3 : t.GH = 26)
  (h4 : t.HE = 80)
  (h5 : t.EQ > 0)
  (h6 : t.EQ < t.EF) :
  t.EQ = 160 / 3 :=
sorry

end tangent_circle_position_l3669_366905


namespace triangle_inequality_l3669_366920

/-- Given two triangles ABC and A₁B₁C₁, where b₁ and c₁ have areas S and S₁ respectively,
    prove the inequality and its equality condition. -/
theorem triangle_inequality (a b c a₁ b₁ c₁ S S₁ : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a₁ > 0 ∧ b₁ > 0 ∧ c₁ > 0 ∧ S > 0 ∧ S₁ > 0 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  a₁ + b₁ > c₁ ∧ b₁ + c₁ > a₁ ∧ c₁ + a₁ > b₁ →
  a₁^2 * (-a^2 + b^2 + c^2) + b₁^2 * (a^2 - b^2 + c^2) + c₁^2 * (a^2 + b^2 - c^2) ≥ 16 * S * S₁ ∧
  (a₁^2 * (-a^2 + b^2 + c^2) + b₁^2 * (a^2 - b^2 + c^2) + c₁^2 * (a^2 + b^2 - c^2) = 16 * S * S₁ ↔
   ∃ k : ℝ, k > 0 ∧ a₁ = k * a ∧ b₁ = k * b ∧ c₁ = k * c) :=
by sorry

end triangle_inequality_l3669_366920


namespace fraction_product_simplification_l3669_366958

theorem fraction_product_simplification :
  (240 : ℚ) / 20 * 6 / 180 * 10 / 4 = 1 := by sorry

end fraction_product_simplification_l3669_366958


namespace minimal_ratio_is_two_thirds_l3669_366924

/-- Represents a point on the square tablecloth -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the square tablecloth with dark spots -/
structure Tablecloth where
  side_length : ℝ
  spots : Set Point

/-- The total area of dark spots on the tablecloth -/
def total_spot_area (t : Tablecloth) : ℝ := sorry

/-- The visible area of spots when folded along a specified line -/
def visible_area_when_folded (t : Tablecloth) (fold_type : Nat) : ℝ := sorry

theorem minimal_ratio_is_two_thirds (t : Tablecloth) :
  let S := total_spot_area t
  let S₁ := visible_area_when_folded t 1  -- Folding along first median or diagonal
  (∀ (i : Nat), i ≤ 3 → visible_area_when_folded t i = S₁) ∧  -- First three folds result in S₁
  (visible_area_when_folded t 4 = S) →  -- Fourth fold (other diagonal) results in S
  ∃ (ratio : ℝ), (∀ (r : ℝ), S₁ / S ≥ r → r ≤ ratio) ∧ ratio = 2/3 := by
  sorry

end minimal_ratio_is_two_thirds_l3669_366924


namespace car_speed_l3669_366993

theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 624 ∧ time = 2 + 2/5 → speed = distance / time → speed = 260 := by
sorry

end car_speed_l3669_366993


namespace complex_conjugate_root_l3669_366966

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- A complex number is a root of a polynomial if the polynomial evaluates to zero at that number -/
def is_root (f : RealPolynomial) (z : ℂ) : Prop := f z.re = 0 ∧ f z.im = 0

theorem complex_conjugate_root (f : RealPolynomial) (a b : ℝ) :
  is_root f (Complex.mk a b) → is_root f (Complex.mk a (-b)) :=
by sorry

end complex_conjugate_root_l3669_366966


namespace average_speed_two_hours_l3669_366975

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) (h1 : speed1 = 140) (h2 : speed2 = 40) :
  (speed1 + speed2) / 2 = 90 := by
  sorry

end average_speed_two_hours_l3669_366975


namespace wooden_block_stacks_height_difference_l3669_366912

/-- The height of wooden block stacks problem -/
theorem wooden_block_stacks_height_difference :
  let first_stack : ℕ := 7
  let second_stack : ℕ := first_stack + 3
  let third_stack : ℕ := second_stack - 6
  let fifth_stack : ℕ := 2 * second_stack
  let total_blocks : ℕ := 55
  let other_stacks_total : ℕ := first_stack + second_stack + third_stack + fifth_stack
  let fourth_stack : ℕ := total_blocks - other_stacks_total
  fourth_stack - third_stack = 10 :=
by sorry

end wooden_block_stacks_height_difference_l3669_366912


namespace chloe_boxes_l3669_366917

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 2

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 6

/-- The total number of pieces of winter clothing Chloe has -/
def total_pieces : ℕ := 32

/-- The number of boxes Chloe found -/
def boxes : ℕ := total_pieces / (scarves_per_box + mittens_per_box)

theorem chloe_boxes : boxes = 4 := by
  sorry

end chloe_boxes_l3669_366917


namespace isosceles_triangle_angles_l3669_366909

-- Define an isosceles triangle with an exterior angle of 140°
structure IsoscelesTriangle where
  angles : Fin 3 → ℝ
  isIsosceles : (angles 0 = angles 1) ∨ (angles 1 = angles 2) ∨ (angles 0 = angles 2)
  sumOfAngles : angles 0 + angles 1 + angles 2 = 180
  exteriorAngle : ℝ
  exteriorAngleValue : exteriorAngle = 140

-- Theorem statement
theorem isosceles_triangle_angles (t : IsoscelesTriangle) :
  (t.angles 0 = 40 ∧ t.angles 1 = 40 ∧ t.angles 2 = 100) ∨
  (t.angles 0 = 70 ∧ t.angles 1 = 70 ∧ t.angles 2 = 40) :=
sorry

end isosceles_triangle_angles_l3669_366909


namespace population_growth_s_curve_l3669_366968

/-- Represents the population size at a given time -/
def PopulationSize := ℝ

/-- Represents time -/
def Time := ℝ

/-- Represents the carrying capacity of the environment -/
def CarryingCapacity := ℝ

/-- Represents the growth rate of the population -/
def GrowthRate := ℝ

/-- A function that models population growth over time -/
def populationGrowthModel (t : Time) (K : CarryingCapacity) (r : GrowthRate) : PopulationSize :=
  sorry

/-- Predicate that checks if a function exhibits an S-curve pattern -/
def isSCurve (f : Time → PopulationSize) : Prop :=
  sorry

/-- Theorem stating that population growth often exhibits an S-curve in nature -/
theorem population_growth_s_curve 
  (limitedEnvironment : CarryingCapacity → Prop)
  (environmentalFactors : (Time → PopulationSize) → Prop) :
  ∃ (K : CarryingCapacity) (r : GrowthRate),
    limitedEnvironment K ∧ 
    environmentalFactors (populationGrowthModel · K r) ∧
    isSCurve (populationGrowthModel · K r) :=
  sorry

end population_growth_s_curve_l3669_366968


namespace complex_modulus_sqrt_two_l3669_366927

theorem complex_modulus_sqrt_two (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) :
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
  sorry

end complex_modulus_sqrt_two_l3669_366927


namespace mike_sold_45_books_l3669_366928

/-- The number of books Mike sold at the garage sale -/
def books_sold (initial_books current_books : ℕ) : ℕ :=
  initial_books - current_books

/-- Proof that Mike sold 45 books -/
theorem mike_sold_45_books (h1 : books_sold 51 6 = 45) : books_sold 51 6 = 45 := by
  sorry

end mike_sold_45_books_l3669_366928


namespace relay_team_arrangements_l3669_366933

theorem relay_team_arrangements (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end relay_team_arrangements_l3669_366933


namespace sector_central_angle_l3669_366919

/-- Given a sector with radius R and a perimeter equal to half the circumference of its circle,
    the central angle of the sector is (π - 2) radians. -/
theorem sector_central_angle (R : ℝ) (h : R > 0) : 
  ∃ θ : ℝ, θ > 0 ∧ θ < 2 * π ∧ 
  (2 * R + R * θ = π * R) → θ = π - 2 := by sorry

end sector_central_angle_l3669_366919


namespace tangent_line_at_origin_monotonically_decreasing_when_m_positive_extremum_values_iff_m_negative_l3669_366945

noncomputable section

variables (m : ℝ) (x : ℝ)

def f (m : ℝ) (x : ℝ) : ℝ := (m^2 * x) / (x^2 - m)

theorem tangent_line_at_origin (h : m = 1) :
  ∃ (k : ℝ), ∀ (x y : ℝ), y = f m x → (x = 0 ∧ y = 0) → x + y = 0 :=
sorry

theorem monotonically_decreasing_when_m_positive (h : m > 0) :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f m x₁ > f m x₂ :=
sorry

theorem extremum_values_iff_m_negative :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f m x₁ = f m x₂ ∧ 
   (∀ (x : ℝ), f m x ≤ f m x₁ ∨ f m x ≤ f m x₂)) ↔ m < 0 :=
sorry

end tangent_line_at_origin_monotonically_decreasing_when_m_positive_extremum_values_iff_m_negative_l3669_366945


namespace missing_number_proof_l3669_366994

theorem missing_number_proof (some_number : ℤ) : 
  some_number = 3 → |9 - 8 * (some_number - 12)| - |5 - 11| = 75 :=
by
  sorry

end missing_number_proof_l3669_366994


namespace ice_cream_cost_l3669_366959

theorem ice_cream_cost (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (quarters : ℕ) 
  (family_members : ℕ) (remaining_cents : ℕ) :
  pennies = 123 →
  nickels = 85 →
  dimes = 35 →
  quarters = 26 →
  family_members = 5 →
  remaining_cents = 48 →
  let total_cents := pennies + nickels * 5 + dimes * 10 + quarters * 25
  let spent_cents := total_cents - remaining_cents
  let cost_per_scoop := spent_cents / family_members
  cost_per_scoop = 300 := by
sorry

end ice_cream_cost_l3669_366959


namespace diameter_circumference_relation_l3669_366900

theorem diameter_circumference_relation (c : ℝ) (d : ℝ) (π : ℝ) : c > 0 → d > 0 → π > 0 → c = π * d → d = (1 / π) * c := by
  sorry

end diameter_circumference_relation_l3669_366900


namespace arithmetic_sequence_convex_condition_l3669_366948

/-- A sequence a is convex if a(n+1) + a(n-1) ≤ 2*a(n) for all n ≥ 2 -/
def IsConvexSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n + 1) + a (n - 1) ≤ 2 * a n

/-- The nth term of an arithmetic sequence with first term b₁ and common difference d -/
def ArithmeticSequence (b₁ d : ℝ) (n : ℕ) : ℝ :=
  b₁ + (n - 1) * d

theorem arithmetic_sequence_convex_condition (d : ℝ) :
  let b := ArithmeticSequence 2 (Real.log d)
  IsConvexSequence (fun n => b n / n) → d ≥ Real.exp 2 := by
  sorry

#check arithmetic_sequence_convex_condition

end arithmetic_sequence_convex_condition_l3669_366948


namespace group_size_problem_l3669_366953

theorem group_size_problem (total_cents : ℕ) (h1 : total_cents = 64736) : ∃ n : ℕ, n * n = total_cents ∧ n = 254 := by
  sorry

end group_size_problem_l3669_366953


namespace geometric_sequence_sum_product_l3669_366989

theorem geometric_sequence_sum_product (a b c : ℝ) : 
  (∃ q : ℝ, b = a * q ∧ c = b * q) →  -- geometric sequence condition
  a + b + c = 14 →                    -- sum condition
  a * b * c = 64 →                    -- product condition
  ((a = 8 ∧ b = 4 ∧ c = 2) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) := by
sorry

end geometric_sequence_sum_product_l3669_366989


namespace cos_double_angle_special_case_l3669_366922

theorem cos_double_angle_special_case (α : Real) 
  (h : Real.sin (π / 2 - α) = 1 / 4) : 
  Real.cos (2 * α) = - 7 / 8 := by
sorry

end cos_double_angle_special_case_l3669_366922


namespace ellipse_and_circle_theorem_l3669_366988

/-- An ellipse with center at the origin and foci on the coordinate axes -/
structure CenteredEllipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The equation of the ellipse -/
def CenteredEllipse.equation (e : CenteredEllipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The ellipse passes through the given points -/
def CenteredEllipse.passes_through (e : CenteredEllipse) : Prop :=
  e.equation 2 (Real.sqrt 2) ∧ e.equation (Real.sqrt 6) 1

/-- The main theorem -/
theorem ellipse_and_circle_theorem (e : CenteredEllipse) 
    (h_passes : e.passes_through) : 
    (e.a^2 = 8 ∧ e.b^2 = 4) ∧
    ∃ (r : ℝ), r^2 = 8/3 ∧
      ∀ (l : ℝ → ℝ → Prop), 
        (∃ (k m : ℝ), ∀ x y, l x y ↔ y = k * x + m) →
        (∃ x y, x^2 + y^2 = r^2 ∧ l x y) →
        ∃ (A B : ℝ × ℝ), 
          A ≠ B ∧
          e.equation A.1 A.2 ∧ 
          e.equation B.1 B.2 ∧
          l A.1 A.2 ∧ 
          l B.1 B.2 ∧
          A.1 * B.1 + A.2 * B.2 = 0 := by
  sorry

end ellipse_and_circle_theorem_l3669_366988


namespace forty_is_twenty_percent_of_two_hundred_l3669_366923

theorem forty_is_twenty_percent_of_two_hundred (x : ℝ) : 40 = (20 / 100) * x → x = 200 := by
  sorry

end forty_is_twenty_percent_of_two_hundred_l3669_366923


namespace population_increase_l3669_366964

/-- The birth rate in people per two seconds -/
def birth_rate : ℚ := 7

/-- The death rate in people per two seconds -/
def death_rate : ℚ := 1

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- The net population increase in one day -/
def net_increase : ℕ := 259200

theorem population_increase :
  (birth_rate - death_rate) / 2 * seconds_per_day = net_increase := by
  sorry

end population_increase_l3669_366964


namespace f_max_min_on_interval_l3669_366963

def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

theorem f_max_min_on_interval :
  let a : ℝ := -3
  let b : ℝ := 3
  ∃ (x_max x_min : ℝ), a ≤ x_max ∧ x_max ≤ b ∧ a ≤ x_min ∧ x_min ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    f x_max = 59 ∧ f x_min = -49 :=
by sorry

end f_max_min_on_interval_l3669_366963


namespace round_trip_ticket_percentage_l3669_366932

theorem round_trip_ticket_percentage (total_passengers : ℝ) :
  let round_trip_with_car := 0.20 * total_passengers
  let round_trip_without_car_ratio := 0.40
  let round_trip_passengers := round_trip_with_car / (1 - round_trip_without_car_ratio)
  round_trip_passengers / total_passengers = 1/3 := by
sorry

end round_trip_ticket_percentage_l3669_366932


namespace gas_needed_is_eighteen_l3669_366947

/-- Calculates the total amount of gas needed to fill both a truck and car tank completely. -/
def total_gas_needed (truck_capacity car_capacity : ℚ) (truck_fullness car_fullness : ℚ) : ℚ :=
  (truck_capacity - truck_capacity * truck_fullness) + (car_capacity - car_capacity * car_fullness)

/-- Proves that the total amount of gas needed to fill both tanks is 18 gallons. -/
theorem gas_needed_is_eighteen :
  total_gas_needed 20 12 (1/2) (1/3) = 18 := by
  sorry

end gas_needed_is_eighteen_l3669_366947


namespace translation_proof_l3669_366921

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector between two points -/
def vector (p q : Point) : Point :=
  ⟨q.x - p.x, q.y - p.y⟩

/-- Translate a point by a vector -/
def translate (p : Point) (v : Point) : Point :=
  ⟨p.x + v.x, p.y + v.y⟩

theorem translation_proof (A C D : Point)
    (h1 : A = ⟨-1, 4⟩)
    (h2 : C = ⟨4, 7⟩)
    (h3 : D = ⟨-4, 1⟩)
    (h4 : ∃ B : Point, vector A B = vector C D) :
    ∃ B : Point, B = ⟨-9, -2⟩ ∧ vector A B = vector C D := by
  sorry

#check translation_proof

end translation_proof_l3669_366921


namespace arithmetic_mean_function_is_constant_l3669_366977

/-- A function from ℤ × ℤ to ℤ⁺ satisfying the arithmetic mean property -/
def ArithmeticMeanFunction (f : ℤ × ℤ → ℤ) : Prop :=
  (∀ i j : ℤ, 0 < f (i, j)) ∧ 
  (∀ i j : ℤ, 4 * f (i, j) = f (i-1, j) + f (i+1, j) + f (i, j-1) + f (i, j+1))

/-- Theorem stating that any function satisfying the arithmetic mean property is constant -/
theorem arithmetic_mean_function_is_constant (f : ℤ × ℤ → ℤ) 
  (h : ArithmeticMeanFunction f) : 
  ∃ c : ℤ, ∀ i j : ℤ, f (i, j) = c :=
sorry

end arithmetic_mean_function_is_constant_l3669_366977


namespace intersection_equals_B_range_l3669_366956

/-- The set A -/
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}

/-- The set B parameterized by m -/
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

/-- The theorem stating the range of m when A ∩ B = B -/
theorem intersection_equals_B_range (m : ℝ) : 
  (A ∩ B m = B m) ↔ (1 ≤ m ∧ m ≤ 3) :=
sorry

end intersection_equals_B_range_l3669_366956


namespace alcohol_quantity_in_mixture_l3669_366908

/-- Given a mixture of alcohol and water, this theorem proves that if the initial ratio
    of alcohol to water is 4:3, and adding 4 liters of water changes the ratio to 4:5,
    then the initial quantity of alcohol in the mixture is 8 liters. -/
theorem alcohol_quantity_in_mixture
  (initial_alcohol : ℝ) (initial_water : ℝ)
  (h1 : initial_alcohol / initial_water = 4 / 3)
  (h2 : initial_alcohol / (initial_water + 4) = 4 / 5) :
  initial_alcohol = 8 := by
  sorry

end alcohol_quantity_in_mixture_l3669_366908


namespace evaluate_expression_l3669_366969

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 5) : y * (y - 3 * x) = -5 := by
  sorry

end evaluate_expression_l3669_366969


namespace squares_below_specific_line_l3669_366935

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of unit squares below a line in the first quadrant --/
def countSquaresBelowLine (l : Line) : ℕ :=
  sorry

/-- The specific line 10x + 210y = 2100 --/
def specificLine : Line := { a := 10, b := 210, c := 2100 }

theorem squares_below_specific_line :
  countSquaresBelowLine specificLine = 941 :=
sorry

end squares_below_specific_line_l3669_366935


namespace gcd_455_299_l3669_366954

theorem gcd_455_299 : Nat.gcd 455 299 = 13 := by
  sorry

end gcd_455_299_l3669_366954


namespace library_books_calculation_l3669_366987

theorem library_books_calculation (initial_books : ℕ) (loaned_books : ℕ) (return_rate : ℚ) : 
  initial_books = 75 → 
  loaned_books = 60 → 
  return_rate = 7/10 → 
  initial_books - loaned_books + (return_rate * loaned_books).floor = 57 := by
sorry

end library_books_calculation_l3669_366987


namespace cookies_in_box_l3669_366982

theorem cookies_in_box (cookies_per_bag : ℕ) (calories_per_cookie : ℕ) (total_calories : ℕ) :
  cookies_per_bag = 20 →
  calories_per_cookie = 20 →
  total_calories = 1600 →
  total_calories / (cookies_per_bag * calories_per_cookie) = 4 :=
by sorry

end cookies_in_box_l3669_366982


namespace friends_weekly_biking_distance_l3669_366999

/-- The total distance biked by two friends in a week -/
def total_weekly_distance (onur_daily_distance : ℕ) (hanil_extra_distance : ℕ) (days_per_week : ℕ) : ℕ :=
  (onur_daily_distance * days_per_week) + ((onur_daily_distance + hanil_extra_distance) * days_per_week)

/-- Theorem: The total distance biked by Onur and Hanil in a week is 2700 kilometers -/
theorem friends_weekly_biking_distance :
  total_weekly_distance 250 40 5 = 2700 := by
  sorry

end friends_weekly_biking_distance_l3669_366999


namespace ordered_pair_solution_l3669_366952

theorem ordered_pair_solution :
  ∀ x y : ℝ,
  (x + y = (6 - x) + (6 - y)) →
  (x - y = (x - 2) + (y - 2)) →
  (x = 2 ∧ y = 4) :=
by
  sorry

end ordered_pair_solution_l3669_366952


namespace triangle_is_acute_l3669_366997

-- Define the triangle and its angles
def Triangle (a1 a2 a3 : ℝ) : Prop :=
  a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a1 + a2 + a3 = 180

-- Define an acute triangle
def AcuteTriangle (a1 a2 a3 : ℝ) : Prop :=
  Triangle a1 a2 a3 ∧ a1 < 90 ∧ a2 < 90 ∧ a3 < 90

-- Theorem statement
theorem triangle_is_acute (a2 : ℝ) :
  Triangle (2 * a2) a2 (1.5 * a2) → AcuteTriangle (2 * a2) a2 (1.5 * a2) :=
by
  sorry

end triangle_is_acute_l3669_366997


namespace complex_fraction_simplification_l3669_366939

theorem complex_fraction_simplification :
  (7 + 16 * Complex.I) / (4 - 5 * Complex.I) = -52/41 + (99/41) * Complex.I :=
by sorry

end complex_fraction_simplification_l3669_366939


namespace hacky_sack_jumping_rope_problem_l3669_366907

theorem hacky_sack_jumping_rope_problem : 
  ∀ (hacky_sack_players jump_rope_players : ℕ),
    hacky_sack_players = 6 →
    jump_rope_players = 6 * hacky_sack_players →
    jump_rope_players ≠ 12 :=
by
  sorry

end hacky_sack_jumping_rope_problem_l3669_366907


namespace expression_simplification_l3669_366962

theorem expression_simplification (x : ℝ) : 2*x + 3*x^2 + 1 - (6 - 2*x - 3*x^2) = 6*x^2 + 4*x - 5 := by
  sorry

end expression_simplification_l3669_366962


namespace total_marks_is_660_l3669_366970

/-- Represents Eva's scores in a semester -/
structure SemesterScores where
  maths : ℕ
  arts : ℕ
  science : ℕ
  history : ℕ

/-- Calculates the total score for a semester -/
def totalScore (scores : SemesterScores) : ℕ :=
  scores.maths + scores.arts + scores.science + scores.history

/-- Eva's scores for the second semester -/
def secondSemester : SemesterScores :=
  { maths := 80, arts := 90, science := 90, history := 85 }

/-- Eva's scores for the first semester -/
def firstSemester : SemesterScores :=
  { maths := secondSemester.maths + 10,
    arts := secondSemester.arts - 15,
    science := secondSemester.science - (secondSemester.science / 3),
    history := secondSemester.history + 5 }

/-- Theorem: The total number of marks in all semesters is 660 -/
theorem total_marks_is_660 :
  totalScore firstSemester + totalScore secondSemester = 660 := by
  sorry


end total_marks_is_660_l3669_366970


namespace base_10_729_to_base_7_l3669_366946

/-- Converts a base-7 number to base 10 --/
def base7ToBase10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7 + d

/-- The base-10 number 729 is equal to 2061 in base 7 --/
theorem base_10_729_to_base_7 : base7ToBase10 2 0 6 1 = 729 := by
  sorry

end base_10_729_to_base_7_l3669_366946


namespace complex_product_ab_l3669_366980

theorem complex_product_ab (z : ℂ) (a b : ℝ) : 
  z = a + b * Complex.I → 
  z = (4 + 3 * Complex.I) * Complex.I → 
  a * b = -12 := by sorry

end complex_product_ab_l3669_366980


namespace parabola_point_position_l3669_366991

theorem parabola_point_position 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_below : 2 < a + b + c) : 
  2 < c + b + a := by sorry

end parabola_point_position_l3669_366991


namespace units_digit_of_product_l3669_366983

theorem units_digit_of_product (n : ℕ) : 
  (2^2010 * 5^2011 * 11^2012) % 10 = 0 :=
by sorry

end units_digit_of_product_l3669_366983


namespace min_value_ab_min_value_is_two_l3669_366986

theorem min_value_ab (a b : ℝ) (h : (a⁻¹ + b⁻¹ : ℝ) = Real.sqrt (a * b)) :
  ∀ x y : ℝ, x > 0 → y > 0 → (x⁻¹ + y⁻¹ : ℝ) = Real.sqrt (x * y) → a * b ≤ x * y :=
by sorry

theorem min_value_is_two (a b : ℝ) (h : (a⁻¹ + b⁻¹ : ℝ) = Real.sqrt (a * b)) :
  a * b = 2 :=
by sorry

end min_value_ab_min_value_is_two_l3669_366986

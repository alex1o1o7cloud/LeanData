import Mathlib

namespace digit_removal_theorem_l4123_412327

theorem digit_removal_theorem :
  (∀ (n : ℕ), n ≥ 2 → 
    (∃! (x : ℕ), x = 625 * 10^(n-2) ∧ 
      (∃ (m : ℕ), x = 6 * 10^n + m ∧ m = x / 25))) ∧
  (¬ ∃ (x : ℕ), ∃ (n : ℕ), ∃ (m : ℕ), 
    x = 6 * 10^n + m ∧ m = x / 35) :=
by sorry

end digit_removal_theorem_l4123_412327


namespace average_after_removal_l4123_412300

theorem average_after_removal (numbers : Finset ℝ) (sum : ℝ) :
  Finset.card numbers = 12 →
  sum / 12 = 90 →
  sum = Finset.sum numbers id →
  68 ∈ numbers →
  75 ∈ numbers →
  82 ∈ numbers →
  (sum - 68 - 75 - 82) / 9 = 95 := by
sorry

end average_after_removal_l4123_412300


namespace davids_biology_marks_l4123_412389

theorem davids_biology_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (chemistry : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (h1 : english = 61) 
  (h2 : mathematics = 65) 
  (h3 : physics = 82) 
  (h4 : chemistry = 67) 
  (h5 : average = 72) 
  (h6 : (english + mathematics + physics + chemistry + biology) / 5 = average) : 
  biology = 85 := by
sorry

end davids_biology_marks_l4123_412389


namespace cookies_eaten_difference_l4123_412326

theorem cookies_eaten_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) :
  initial_sweet = 8 →
  initial_salty = 6 →
  eaten_sweet = 20 →
  eaten_salty = 34 →
  eaten_salty - eaten_sweet = 14 :=
by
  sorry

end cookies_eaten_difference_l4123_412326


namespace min_distance_line_curve_l4123_412347

/-- The line represented by the parametric equations x = t, y = 6 - 2t -/
def line (t : ℝ) : ℝ × ℝ := (t, 6 - 2*t)

/-- The curve represented by the equation (x - 1)² + (y + 2)² = 5 -/
def curve (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

/-- The minimum distance between a point on the line and a point on the curve -/
theorem min_distance_line_curve :
  ∃ (d : ℝ), d = Real.sqrt 5 / 5 ∧
  ∀ (t θ : ℝ),
    let (x₁, y₁) := line t
    let (x₂, y₂) := (1 + Real.sqrt 5 * Real.cos θ, -2 + Real.sqrt 5 * Real.sin θ)
    curve x₂ y₂ →
    d ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) :=
sorry

end min_distance_line_curve_l4123_412347


namespace genetic_material_distribution_l4123_412355

/-- Represents a diploid organism -/
structure DiploidOrganism :=
  (chromosomes : ℕ)
  (is_diploid : chromosomes % 2 = 0)

/-- Represents genetic material in the cytoplasm -/
structure GeneticMaterial :=
  (amount : ℝ)

/-- Represents a cell of a diploid organism -/
structure Cell :=
  (organism : DiploidOrganism)
  (cytoplasm : GeneticMaterial)

/-- Represents the distribution of genetic material during cell division -/
def genetic_distribution (parent : Cell) (daughter1 daughter2 : Cell) : Prop :=
  (daughter1.cytoplasm.amount + daughter2.cytoplasm.amount = parent.cytoplasm.amount) ∧
  (daughter1.cytoplasm.amount ≠ daughter2.cytoplasm.amount)

/-- Theorem stating that genetic material in the cytoplasm is distributed randomly and unequally during cell division -/
theorem genetic_material_distribution 
  (parent : Cell) 
  (daughter1 daughter2 : Cell) :
  genetic_distribution parent daughter1 daughter2 :=
sorry

end genetic_material_distribution_l4123_412355


namespace starship_sales_l4123_412316

theorem starship_sales (starship_price mech_price ultimate_price : ℕ)
                       (total_items total_revenue : ℕ) :
  starship_price = 8 →
  mech_price = 26 →
  ultimate_price = 33 →
  total_items = 31 →
  total_revenue = 370 →
  ∃ (x y : ℕ),
    x + y ≤ total_items ∧
    (total_items - x - y) % 2 = 0 ∧
    x * starship_price + y * mech_price + 
      ((total_items - x - y) / 2) * ultimate_price = total_revenue ∧
    x = 20 := by
  sorry

end starship_sales_l4123_412316


namespace spatial_geometry_theorem_l4123_412312

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- State the theorem
theorem spatial_geometry_theorem 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (perpendicular m β ∧ perpendicular n β → parallel_lines m n) ∧
  (perpendicular m α ∧ perpendicular m β → parallel_planes α β) :=
sorry

end spatial_geometry_theorem_l4123_412312


namespace gain_percentage_proof_l4123_412378

/-- Proves that the gain percentage is 20% when an article is sold for 168 Rs,
    given that it incurs a 15% loss when sold for 119 Rs. -/
theorem gain_percentage_proof (cost_price : ℝ) : 
  (cost_price * 0.85 = 119) →  -- 15% loss when sold for 119
  ((168 - cost_price) / cost_price * 100 = 20) := by
sorry

end gain_percentage_proof_l4123_412378


namespace consecutive_pair_with_17_l4123_412331

theorem consecutive_pair_with_17 (a b : ℤ) : 
  (a = 17 ∨ b = 17) → 
  (abs (a - b) = 1) → 
  (a + b = 35) → 
  (35 % 5 = 0) → 
  ((a = 17 ∧ b = 18) ∨ (a = 18 ∧ b = 17)) := by sorry

end consecutive_pair_with_17_l4123_412331


namespace residue_calculation_l4123_412383

theorem residue_calculation : (222 * 15 - 35 * 9 + 2^3) % 18 = 17 := by
  sorry

end residue_calculation_l4123_412383


namespace max_apartments_l4123_412309

/-- Represents an apartment building with specific properties. -/
structure ApartmentBuilding where
  entrances : Nat
  floors : Nat
  apartments_per_floor : Nat
  two_digit_apartments_in_entrance : Nat

/-- The conditions of the apartment building as described in the problem. -/
def building_conditions (b : ApartmentBuilding) : Prop :=
  b.apartments_per_floor = 4 ∧
  b.two_digit_apartments_in_entrance = 10 * b.entrances ∧
  b.two_digit_apartments_in_entrance ≤ 90

/-- The total number of apartments in the building. -/
def total_apartments (b : ApartmentBuilding) : Nat :=
  b.entrances * b.floors * b.apartments_per_floor

/-- Theorem stating the maximum number of apartments in the building. -/
theorem max_apartments (b : ApartmentBuilding) (h : building_conditions b) :
  total_apartments b ≤ 936 := by
  sorry

#check max_apartments

end max_apartments_l4123_412309


namespace no_square_base_b_l4123_412370

theorem no_square_base_b : ¬ ∃ (b : ℤ), ∃ (n : ℤ), b^2 + 3*b + 1 = n^2 := by sorry

end no_square_base_b_l4123_412370


namespace n_pointed_star_degree_sum_l4123_412304

/-- An n-pointed star formed from a convex polygon -/
structure NPointedStar where
  n : ℕ
  h_n_ge_7 : n ≥ 7

/-- The degree sum of interior angles of an n-pointed star -/
def degree_sum (star : NPointedStar) : ℝ :=
  180 * (star.n - 2)

/-- Theorem: The degree sum of interior angles of an n-pointed star is 180(n-2) -/
theorem n_pointed_star_degree_sum (star : NPointedStar) :
  degree_sum star = 180 * (star.n - 2) := by
  sorry

end n_pointed_star_degree_sum_l4123_412304


namespace triangle_area_l4123_412391

theorem triangle_area (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let angle_C := Real.arccos ((AB^2 + BC^2 - AC^2) / (2 * AB * BC))
  AB = 2 * Real.sqrt 3 ∧ BC = 2 ∧ angle_C = π / 3 →
  (1 / 2) * AB * BC * Real.sin angle_C = 2 * Real.sqrt 3 := by
  sorry

end triangle_area_l4123_412391


namespace percentage_equality_l4123_412384

theorem percentage_equality (x y : ℝ) (h : (18 / 100) * x = (9 / 100) * y) :
  (12 / 100) * x = (6 / 100) * y := by
  sorry

end percentage_equality_l4123_412384


namespace tims_pencils_count_l4123_412364

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 2

/-- The total number of pencils after Tim's action -/
def total_pencils : ℕ := 5

/-- The number of pencils Tim placed in the drawer -/
def tims_pencils : ℕ := total_pencils - initial_pencils

theorem tims_pencils_count : tims_pencils = 3 := by
  sorry

end tims_pencils_count_l4123_412364


namespace complement_intersection_theorem_l4123_412354

-- Define the sets M and N
def M : Set ℝ := {x | x < 0 ∨ x > 2}
def N : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- State the theorem
theorem complement_intersection_theorem :
  (N \ (M ∩ N)) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end complement_intersection_theorem_l4123_412354


namespace range_of_m_l4123_412324

def prop_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*m*x + 7*m - 10 ≠ 0

def prop_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → x^2 - m*x + 4 ≥ 0

theorem range_of_m :
  ∀ m : ℝ, (prop_p m ∨ prop_q m) ∧ (prop_p m ∧ prop_q m) →
  m ∈ Set.Ioo 2 4 ∪ {4} :=
sorry

end range_of_m_l4123_412324


namespace universal_set_equality_l4123_412344

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5, 7}

-- Define set A
def A : Finset Nat := {1, 3, 5, 7}

-- Define set B
def B : Finset Nat := {3, 5}

-- Theorem statement
theorem universal_set_equality : U = A ∪ (U \ B) := by
  sorry

end universal_set_equality_l4123_412344


namespace max_sum_given_constraints_l4123_412388

theorem max_sum_given_constraints (a b : ℝ) 
  (h1 : a^2 + b^2 = 130) 
  (h2 : a * b = 45) : 
  a + b ≤ 2 * Real.sqrt 55 := by
sorry

end max_sum_given_constraints_l4123_412388


namespace meaningful_expression_l4123_412363

theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x = (a + 3) / Real.sqrt (a - 3)) ↔ a > 3 := by
  sorry

end meaningful_expression_l4123_412363


namespace first_player_wins_l4123_412340

/-- Represents a move in the coin game -/
structure Move where
  player : Nat
  coins : Nat

/-- Represents the state of the game -/
structure GameState where
  coins : Nat
  turn : Nat

/-- Checks if a move is valid for a given player -/
def isValidMove (m : Move) (gs : GameState) : Prop :=
  (m.player = gs.turn % 2) ∧
  (if m.player = 0
   then m.coins % 2 = 1 ∧ m.coins ≥ 1 ∧ m.coins ≤ 99
   else m.coins % 2 = 0 ∧ m.coins ≥ 2 ∧ m.coins ≤ 100) ∧
  (m.coins ≤ gs.coins)

/-- Applies a move to a game state -/
def applyMove (m : Move) (gs : GameState) : GameState :=
  { coins := gs.coins - m.coins, turn := gs.turn + 1 }

/-- Defines a winning strategy for the first player -/
def firstPlayerStrategy (gs : GameState) : Move :=
  if gs.turn = 0 then
    { player := 0, coins := 99 }
  else
    { player := 0, coins := 101 - (gs.coins % 101) }

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Move),
    ∀ (gs : GameState),
      gs.coins = 2019 →
      (∀ (m : Move), isValidMove m gs → 
        ∃ (nextMove : Move), 
          isValidMove nextMove (applyMove m gs) ∧
          strategy (applyMove m gs) = nextMove) ∧
      (∀ (sequence : Nat → Move),
        (∀ (i : Nat), isValidMove (sequence i) (applyMove (sequence (i-1)) gs)) →
        ∃ (n : Nat), ¬isValidMove (sequence n) (applyMove (sequence (n-1)) gs)) :=
sorry


end first_player_wins_l4123_412340


namespace simplify_and_solve_for_t_l4123_412329

theorem simplify_and_solve_for_t
  (m Q : ℝ)
  (j : ℝ)
  (h : j ≠ -2)
  (h_pos_m : m > 0)
  (h_pos_Q : Q > 0)
  (h_eq : Q = m / (2 + j) ^ t) :
  t = Real.log (m / Q) / Real.log (2 + j) :=
by sorry

end simplify_and_solve_for_t_l4123_412329


namespace apple_price_correct_l4123_412398

/-- The price of one apple in dollars -/
def apple_price : ℚ := 49/30

/-- The price of one orange in dollars -/
def orange_price : ℚ := 3/4

/-- The number of apples that equal the price of 2 watermelons or 3 pineapples -/
def apple_equiv : ℕ := 6

/-- The number of watermelons that equal the price of 6 apples or 3 pineapples -/
def watermelon_equiv : ℕ := 2

/-- The number of pineapples that equal the price of 6 apples or 2 watermelons -/
def pineapple_equiv : ℕ := 3

/-- The number of oranges bought -/
def oranges_bought : ℕ := 24

/-- The number of apples bought -/
def apples_bought : ℕ := 18

/-- The number of watermelons bought -/
def watermelons_bought : ℕ := 12

/-- The number of pineapples bought -/
def pineapples_bought : ℕ := 18

/-- The total bill in dollars -/
def total_bill : ℚ := 165

theorem apple_price_correct :
  apple_price * apple_equiv = apple_price * watermelon_equiv * 3 ∧
  apple_price * 2 * pineapple_equiv = apple_price * watermelon_equiv * 3 ∧
  orange_price * oranges_bought + apple_price * apples_bought + 
  (apple_price * 3) * watermelons_bought + (apple_price * 2) * pineapples_bought = total_bill :=
by sorry

end apple_price_correct_l4123_412398


namespace fourth_vertex_coordinates_l4123_412335

/-- A regular tetrahedron with integer coordinates -/
structure RegularTetrahedron where
  v1 : ℤ × ℤ × ℤ
  v2 : ℤ × ℤ × ℤ
  v3 : ℤ × ℤ × ℤ
  v4 : ℤ × ℤ × ℤ
  is_regular : True  -- Placeholder for the regularity condition

/-- The fourth vertex of the regular tetrahedron -/
def fourth_vertex (t : RegularTetrahedron) : ℤ × ℤ × ℤ := t.v4

/-- The theorem stating the coordinates of the fourth vertex -/
theorem fourth_vertex_coordinates (t : RegularTetrahedron) 
  (h1 : t.v1 = (0, 1, 2))
  (h2 : t.v2 = (4, 2, 1))
  (h3 : t.v3 = (3, 1, 5)) :
  fourth_vertex t = (3, -2, 2) := by sorry

end fourth_vertex_coordinates_l4123_412335


namespace mikes_shopping_expense_l4123_412350

/-- Calculates the total amount Mike spent given the costs and discounts of items. -/
def total_spent (food_cost wallet_cost shirt_cost shoes_cost belt_cost : ℚ)
  (shirt_discount shoes_discount belt_discount : ℚ) : ℚ :=
  food_cost + wallet_cost +
  shirt_cost * (1 - shirt_discount) +
  shoes_cost * (1 - shoes_discount) +
  belt_cost * (1 - belt_discount)

/-- Theorem stating the total amount Mike spent given the conditions. -/
theorem mikes_shopping_expense :
  let food_cost : ℚ := 30
  let wallet_cost : ℚ := food_cost + 60
  let shirt_cost : ℚ := wallet_cost / 3
  let shoes_cost : ℚ := 2 * wallet_cost
  let belt_cost : ℚ := shoes_cost - 45
  let shirt_discount : ℚ := 20 / 100
  let shoes_discount : ℚ := 15 / 100
  let belt_discount : ℚ := 10 / 100
  total_spent food_cost wallet_cost shirt_cost shoes_cost belt_cost
    shirt_discount shoes_discount belt_discount = 418.5 := by sorry

end mikes_shopping_expense_l4123_412350


namespace distribute_five_gifts_to_three_fans_l4123_412352

/-- The number of ways to distribute n identical gifts to k different fans,
    where each fan receives at least one gift -/
def distribute_gifts (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 5 identical gifts to 3 different fans,
    where each fan receives at least one gift, can be done in 6 ways -/
theorem distribute_five_gifts_to_three_fans :
  distribute_gifts 5 3 = 6 := by sorry

end distribute_five_gifts_to_three_fans_l4123_412352


namespace factorization_equality_l4123_412382

theorem factorization_equality (a b : ℝ) : a^3 - 2*a^2*b + a*b^2 = a*(a-b)^2 := by
  sorry

end factorization_equality_l4123_412382


namespace coin_problem_l4123_412396

theorem coin_problem :
  ∀ (x y : ℕ),
    x + y = 15 →
    2 * x + 5 * y = 51 →
    x = y + 1 :=
by
  sorry

end coin_problem_l4123_412396


namespace inequality_system_solution_l4123_412365

theorem inequality_system_solution (x : ℝ) : 
  (2 + x > 7 - 4 * x) ∧ (x < (4 + x) / 2) → 1 < x ∧ x < 4 := by
  sorry

end inequality_system_solution_l4123_412365


namespace second_fish_length_is_02_l4123_412381

/-- The length of the first fish in feet -/
def first_fish_length : ℝ := 0.3

/-- The difference in length between the first and second fish in feet -/
def length_difference : ℝ := 0.1

/-- The length of the second fish in feet -/
def second_fish_length : ℝ := first_fish_length - length_difference

/-- Theorem stating that the second fish is 0.2 foot long -/
theorem second_fish_length_is_02 : second_fish_length = 0.2 := by
  sorry

end second_fish_length_is_02_l4123_412381


namespace discount_comparison_l4123_412328

def initial_amount : ℝ := 20000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.15, 0.15, 0.05]
def option2_discounts : List ℝ := [0.30, 0.10, 0.02]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

theorem discount_comparison :
  apply_successive_discounts initial_amount option1_discounts -
  apply_successive_discounts initial_amount option2_discounts = 1379.50 :=
sorry

end discount_comparison_l4123_412328


namespace sum_product_difference_l4123_412394

theorem sum_product_difference (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 126) : 
  |x - y| = 11 := by
sorry

end sum_product_difference_l4123_412394


namespace intersection_of_A_and_B_l4123_412373

def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1, 3}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end intersection_of_A_and_B_l4123_412373


namespace phone_cost_calculation_phone_cost_proof_l4123_412321

theorem phone_cost_calculation (current_percentage : Real) (additional_amount : Real) : Real :=
  let total_cost := additional_amount / (1 - current_percentage)
  total_cost

theorem phone_cost_proof :
  phone_cost_calculation 0.4 780 = 1300 := by
  sorry

end phone_cost_calculation_phone_cost_proof_l4123_412321


namespace pyramid_base_side_length_l4123_412314

theorem pyramid_base_side_length (area : ℝ) (slant_height : ℝ) (h1 : area = 120) (h2 : slant_height = 40) :
  ∃ (side_length : ℝ), side_length = 6 ∧ (1/2) * side_length * slant_height = area :=
sorry

end pyramid_base_side_length_l4123_412314


namespace matrix_multiplication_result_l4123_412339

theorem matrix_multiplication_result :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 0; 7, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -1; 0, 2]
  A * B = !![15, -3; 35, -11] := by sorry

end matrix_multiplication_result_l4123_412339


namespace gold_coins_percentage_l4123_412374

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  beads : ℝ
  rings : ℝ
  silver_coins : ℝ
  gold_coins : ℝ

/-- Theorem stating the percentage of gold coins in the urn --/
theorem gold_coins_percentage (u : UrnComposition) 
  (h1 : u.beads = 0.3)
  (h2 : u.rings = 0.1)
  (h3 : u.silver_coins + u.gold_coins = 0.6)
  (h4 : u.silver_coins = 0.35 * (u.silver_coins + u.gold_coins)) :
  u.gold_coins = 0.39 := by
  sorry


end gold_coins_percentage_l4123_412374


namespace nth_equation_proof_l4123_412349

theorem nth_equation_proof (n : ℕ+) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

#check nth_equation_proof

end nth_equation_proof_l4123_412349


namespace fraction_chain_l4123_412318

theorem fraction_chain (a b c d : ℝ) 
  (h1 : a / b = 3)
  (h2 : b / c = 2 / 3)
  (h3 : c / d = 5) :
  d / a = 1 / 10 := by
sorry

end fraction_chain_l4123_412318


namespace job_completion_time_l4123_412361

/-- Represents the time (in hours) it takes for a single machine to complete the job -/
def single_machine_time : ℝ := 216

/-- Represents the number of machines of each type used -/
def machines_per_type : ℕ := 9

/-- Represents the time (in hours) it takes for all machines working together to complete the job -/
def total_job_time : ℝ := 12

theorem job_completion_time :
  (((1 / single_machine_time) * machines_per_type + 
    (1 / single_machine_time) * machines_per_type) * total_job_time = 1) →
  single_machine_time = 216 := by
  sorry

end job_completion_time_l4123_412361


namespace smallest_prime_factor_of_2926_l4123_412332

theorem smallest_prime_factor_of_2926 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2926 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2926 → p ≤ q :=
by sorry

end smallest_prime_factor_of_2926_l4123_412332


namespace second_number_11th_row_l4123_412353

/-- Given a lattice with 11 rows, where each row contains 6 numbers,
    and the last number in each row is n × 6 (where n is the row number),
    prove that the second number in the 11th row is 62. -/
theorem second_number_11th_row (rows : Nat) (numbers_per_row : Nat)
    (last_number : Nat → Nat) :
  rows = 11 →
  numbers_per_row = 6 →
  (∀ n, last_number n = n * numbers_per_row) →
  (last_number 10 + 2 = 62) := by
  sorry

end second_number_11th_row_l4123_412353


namespace quadratic_rewrite_ratio_l4123_412334

/-- Given a quadratic expression 8k^2 - 12k + 20, prove that when rewritten in the form a(k + b)^2 + r, the value of r/b is -47.33 -/
theorem quadratic_rewrite_ratio : 
  ∃ (a b r : ℝ), 
    (∀ k, 8 * k^2 - 12 * k + 20 = a * (k + b)^2 + r) ∧ 
    (r / b = -47.33) := by
  sorry

end quadratic_rewrite_ratio_l4123_412334


namespace one_sofa_in_room_l4123_412346

/-- Represents the number of sofas in the room -/
def num_sofas : ℕ := 1

/-- Represents the total number of legs in the room -/
def total_legs : ℕ := 40

/-- Represents the number of legs on a sofa -/
def legs_per_sofa : ℕ := 4

/-- Represents the number of legs from furniture other than sofas -/
def other_furniture_legs : ℕ := 
  4 * 4 +  -- 4 tables with 4 legs each
  2 * 4 +  -- 2 chairs with 4 legs each
  3 * 3 +  -- 3 tables with 3 legs each
  1 * 1 +  -- 1 table with 1 leg
  1 * 2    -- 1 rocking chair with 2 legs

/-- Theorem stating that there is exactly one sofa in the room -/
theorem one_sofa_in_room : 
  num_sofas * legs_per_sofa + other_furniture_legs = total_legs :=
by sorry

end one_sofa_in_room_l4123_412346


namespace subtraction_decimal_result_l4123_412305

theorem subtraction_decimal_result : 5.3567 - 2.1456 - 1.0211 = 2.1900 := by
  sorry

end subtraction_decimal_result_l4123_412305


namespace lcm_24_36_45_l4123_412323

theorem lcm_24_36_45 : Nat.lcm 24 (Nat.lcm 36 45) = 360 := by
  sorry

end lcm_24_36_45_l4123_412323


namespace polygon_with_30_degree_exterior_angles_has_12_sides_l4123_412319

/-- A polygon with exterior angles each measuring 30° has 12 sides -/
theorem polygon_with_30_degree_exterior_angles_has_12_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 2 →
    exterior_angle = 30 →
    (n : ℝ) * exterior_angle = 360 →
    n = 12 := by
  sorry

end polygon_with_30_degree_exterior_angles_has_12_sides_l4123_412319


namespace pens_in_pack_l4123_412393

/-- The number of pens in each pack -/
def pens_per_pack : ℕ := sorry

/-- The number of packs Kendra has -/
def kendra_packs : ℕ := 4

/-- The number of packs Tony has -/
def tony_packs : ℕ := 2

/-- The number of pens Kendra and Tony keep for themselves -/
def pens_kept : ℕ := 4

/-- The number of friends they give pens to -/
def friends : ℕ := 14

theorem pens_in_pack : 
  (kendra_packs + tony_packs) * pens_per_pack - pens_kept - friends = 0 ∧ 
  pens_per_pack = 3 := by sorry

end pens_in_pack_l4123_412393


namespace stevens_apples_l4123_412311

/-- The number of apples Steven has set aside to meet his seed collection goal. -/
def apples_set_aside : ℕ :=
  let total_seeds_needed : ℕ := 60
  let seeds_per_apple : ℕ := 6
  let seeds_per_pear : ℕ := 2
  let seeds_per_grape : ℕ := 3
  let pears : ℕ := 3
  let grapes : ℕ := 9
  let seeds_short : ℕ := 3

  let seeds_from_pears : ℕ := pears * seeds_per_pear
  let seeds_from_grapes : ℕ := grapes * seeds_per_grape
  let seeds_collected : ℕ := total_seeds_needed - seeds_short
  let seeds_from_apples : ℕ := seeds_collected - seeds_from_pears - seeds_from_grapes

  seeds_from_apples / seeds_per_apple

theorem stevens_apples :
  apples_set_aside = 4 :=
by sorry

end stevens_apples_l4123_412311


namespace quadratic_form_constant_l4123_412387

theorem quadratic_form_constant (a h k : ℚ) : 
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end quadratic_form_constant_l4123_412387


namespace complex_magnitude_example_l4123_412371

theorem complex_magnitude_example : Complex.abs (1 - Complex.I / 2) = Real.sqrt 5 / 2 := by
  sorry

end complex_magnitude_example_l4123_412371


namespace roses_planted_l4123_412377

theorem roses_planted (day1 day2 day3 : ℕ) : 
  day2 = day1 + 20 →
  day3 = 2 * day1 →
  day1 + day2 + day3 = 220 →
  day1 = 50 := by
sorry

end roses_planted_l4123_412377


namespace third_term_is_six_l4123_412392

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_second_fourth : a 2 + a 4 = 10
  fourth_minus_third : a 4 = a 3 + 2

/-- The third term of the arithmetic sequence is 6 -/
theorem third_term_is_six (seq : ArithmeticSequence) : seq.a 3 = 6 := by
  sorry

end third_term_is_six_l4123_412392


namespace factor_expression_l4123_412330

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end factor_expression_l4123_412330


namespace zoo_animal_ratio_l4123_412307

theorem zoo_animal_ratio :
  ∀ (birds non_birds : ℕ),
    birds = 450 →
    birds = non_birds + 360 →
    (birds : ℚ) / non_birds = 5 := by
  sorry

end zoo_animal_ratio_l4123_412307


namespace distance_after_translation_l4123_412372

/-- The distance between two points after translation --/
theorem distance_after_translation (x1 y1 x2 y2 tx ty : ℝ) :
  let p1 := (x1 + tx, y1 + ty)
  let p2 := (x2 + tx, y2 + ty)
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt 73 :=
by
  sorry

#check distance_after_translation 5 3 (-3) 0 3 (-3)

end distance_after_translation_l4123_412372


namespace lisa_savings_l4123_412397

theorem lisa_savings (x : ℚ) : 
  (x + 3/5 * x + 2 * (3/5 * x) = 3760 - 400) → x = 2400 := by
  sorry

end lisa_savings_l4123_412397


namespace fraction_simplification_l4123_412338

variables {a b c x y z : ℝ}

theorem fraction_simplification :
  (cx * (a^2 * x^2 + 3 * a^2 * y^2 + c^2 * z^2) + bz * (a^2 * x^2 + 3 * c^2 * x^2 + c^2 * y^2)) / (cx + bz)
  = a^2 * x^2 + c^2 * y^2 + c^2 * z^2 :=
by sorry

end fraction_simplification_l4123_412338


namespace reflection_creates_symmetry_l4123_412359

/-- Represents a letter in the word --/
inductive Letter
| G | E | O | M | T | R | I | Ya

/-- Represents a position in 2D space --/
structure Position :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a word as a list of letters with their positions --/
def Word := List (Letter × Position)

/-- The original word "ГЕОМЕТРИя" --/
def original_word : Word := sorry

/-- Reflects a position across a vertical axis --/
def reflect_vertical (p : Position) (axis : ℝ) : Position :=
  ⟨2 * axis - p.x, p.y⟩

/-- Reflects a word across a vertical axis --/
def reflect_word_vertical (w : Word) (axis : ℝ) : Word :=
  w.map (fun (l, p) => (l, reflect_vertical p axis))

/-- Checks if a word is symmetrical across a vertical axis --/
def is_symmetrical_vertical (w : Word) (axis : ℝ) : Prop :=
  w = reflect_word_vertical w axis

/-- Theorem: Reflecting the word "ГЕОМЕТРИя" across a vertical axis results in a symmetrical figure --/
theorem reflection_creates_symmetry (axis : ℝ) :
  is_symmetrical_vertical (reflect_word_vertical original_word axis) axis := by
  sorry

end reflection_creates_symmetry_l4123_412359


namespace area_calculation_l4123_412358

/-- The lower boundary function of the region -/
def lower_bound (x : ℝ) : ℝ := |x - 4|

/-- The upper boundary function of the region -/
def upper_bound (x : ℝ) : ℝ := 5 - |x - 2|

/-- The region in the xy-plane -/
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | lower_bound p.1 ≤ p.2 ∧ p.2 ≤ upper_bound p.1}

/-- The area of the region -/
noncomputable def area_of_region : ℝ := sorry

theorem area_calculation : area_of_region = 12.875 := by sorry

end area_calculation_l4123_412358


namespace negation_of_universal_proposition_l4123_412367

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℤ, x^3 < 1) ↔ (∃ x : ℤ, x^3 ≥ 1) := by sorry

end negation_of_universal_proposition_l4123_412367


namespace diego_apple_capacity_l4123_412357

/-- The maximum weight of apples Diego can buy given his carrying capacity and other fruit weights -/
theorem diego_apple_capacity (capacity : ℝ) (watermelon grapes oranges bananas : ℝ) 
  (h_capacity : capacity = 50) 
  (h_watermelon : watermelon = 1.5)
  (h_grapes : grapes = 2.75)
  (h_oranges : oranges = 3.5)
  (h_bananas : bananas = 2.7) :
  capacity - (watermelon + grapes + oranges + bananas) = 39.55 := by
  sorry

#check diego_apple_capacity

end diego_apple_capacity_l4123_412357


namespace system1_solution_system2_solution_l4123_412366

-- System 1
theorem system1_solution :
  ∃ (x y : ℝ), y = x + 1 ∧ x + y = 5 ∧ x = 2 ∧ y = 3 := by sorry

-- System 2
theorem system2_solution :
  ∃ (x y : ℝ), x + 2*y = 9 ∧ 3*x - 2*y = -1 ∧ x = 2 ∧ y = 3.5 := by sorry

end system1_solution_system2_solution_l4123_412366


namespace volume_of_specific_tetrahedron_l4123_412348

/-- The volume of a tetrahedron given its edge lengths -/
def tetrahedron_volume (PQ PR PS QR QS RS : ℝ) : ℝ :=
  -- Define the volume calculation here
  sorry

/-- Theorem: The volume of tetrahedron PQRS with given edge lengths is 15√2 / 2 -/
theorem volume_of_specific_tetrahedron :
  let PQ : ℝ := 6
  let PR : ℝ := 4
  let PS : ℝ := 5
  let QR : ℝ := 5
  let QS : ℝ := 3
  let RS : ℝ := 15 / 4 * Real.sqrt 2
  tetrahedron_volume PQ PR PS QR QS RS = 15 / 2 * Real.sqrt 2 := by
  sorry

end volume_of_specific_tetrahedron_l4123_412348


namespace log_expression_equals_five_l4123_412385

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_five :
  (log10 2)^2 + log10 2 * log10 50 + log10 25 + Real.exp (Real.log 3) = 5 := by sorry

end log_expression_equals_five_l4123_412385


namespace point_in_second_quadrant_implies_a_greater_than_four_l4123_412375

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: If point P(4-a, 2) is in the second quadrant, then a > 4 -/
theorem point_in_second_quadrant_implies_a_greater_than_four (a : ℝ) :
  SecondQuadrant ⟨4 - a, 2⟩ → a > 4 := by
  sorry

end point_in_second_quadrant_implies_a_greater_than_four_l4123_412375


namespace binomial_coefficient_ratio_l4123_412322

theorem binomial_coefficient_ratio (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (3*x - 2)^6 = a₀ + a₁*(2*x - 1) + a₂*(2*x - 1)^2 + a₃*(2*x - 1)^3 + 
                 a₄*(2*x - 1)^4 + a₅*(2*x - 1)^5 + a₆*(2*x - 1)^6 →
  (a₁ + a₃ + a₅) / (a₀ + a₂ + a₄ + a₆) = -63/65 := by
sorry


end binomial_coefficient_ratio_l4123_412322


namespace sqrt_18_minus_sqrt_2_over_sqrt_2_l4123_412390

theorem sqrt_18_minus_sqrt_2_over_sqrt_2 : (Real.sqrt 18 - Real.sqrt 2) / Real.sqrt 2 = 2 := by
  sorry

end sqrt_18_minus_sqrt_2_over_sqrt_2_l4123_412390


namespace triangle_problem_l4123_412399

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  -- Condition 1: Vectors are parallel
  (2 * Real.sin (t.A / 2) * (2 * Real.cos (t.A / 4)^2 - 1) = Real.sqrt 3 * Real.cos t.A) →
  -- Condition 2: a = √7
  (t.a = Real.sqrt 7) →
  -- Condition 3: Area of triangle ABC is 3√3/2
  (1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 3 / 2) →
  -- Conclusion 1: A = π/3
  (t.A = Real.pi / 3) ∧
  -- Conclusion 2: b + c = 5
  (t.b + t.c = 5) := by
  sorry

end triangle_problem_l4123_412399


namespace line_and_circle_properties_l4123_412310

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 2 * k = 0

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define the line l₀
def line_l0 (x y : ℝ) : Prop := x - 2 * y + 2 = 0

theorem line_and_circle_properties :
  (∃ k : ℝ, ∀ x y : ℝ, line_l k x y → line_l0 x y → x = y) ∧
  (∀ k : ℝ, ∃ x y : ℝ, line_l k x y ∧ circle_O x y) :=
sorry

end line_and_circle_properties_l4123_412310


namespace fourth_month_sale_l4123_412369

def sales_4_months : List Int := [6335, 6927, 6855, 6562]
def sale_6th_month : Int := 5091
def average_sale : Int := 6500
def num_months : Int := 6

theorem fourth_month_sale :
  let total_sales := average_sale * num_months
  let sum_known_sales := (sales_4_months.sum + sale_6th_month)
  total_sales - sum_known_sales = 7230 := by sorry

end fourth_month_sale_l4123_412369


namespace similar_triangles_perimeter_l4123_412376

theorem similar_triangles_perimeter (p_small p_large : ℝ) : 
  p_small > 0 → 
  p_large > 0 → 
  p_small / p_large = 2 / 3 → 
  p_small + p_large = 20 → 
  p_small = 8 := by
sorry

end similar_triangles_perimeter_l4123_412376


namespace product_less_than_factor_l4123_412368

theorem product_less_than_factor : ∃ (a b : ℝ), 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ a * b < min a b := by
  sorry

end product_less_than_factor_l4123_412368


namespace interior_angle_sum_regular_polygon_l4123_412343

theorem interior_angle_sum_regular_polygon (n : ℕ) (exterior_angle : ℝ) :
  n > 2 →
  exterior_angle = 45 →
  n * exterior_angle = 360 →
  (n - 2) * 180 = 1080 :=
by sorry

end interior_angle_sum_regular_polygon_l4123_412343


namespace tim_singles_count_l4123_412337

/-- The number of points for a single line -/
def single_points : ℕ := 1000

/-- The number of points for a tetris -/
def tetris_points : ℕ := 8 * single_points

/-- The number of tetrises Tim scored -/
def tim_tetrises : ℕ := 4

/-- The total number of points Tim scored -/
def tim_total_points : ℕ := 38000

/-- The number of singles Tim scored -/
def tim_singles : ℕ := (tim_total_points - tim_tetrises * tetris_points) / single_points

theorem tim_singles_count : tim_singles = 6 := by
  sorry

end tim_singles_count_l4123_412337


namespace polar_to_cartesian_l4123_412313

theorem polar_to_cartesian :
  let ρ : ℝ := 4
  let θ : ℝ := 2 * π / 3
  let x : ℝ := ρ * Real.cos θ
  let y : ℝ := ρ * Real.sin θ
  x = -2 ∧ y = 2 * Real.sqrt 3 := by
  sorry

end polar_to_cartesian_l4123_412313


namespace factorization_expression1_l4123_412362

theorem factorization_expression1 (x y : ℝ) : 2 * x^2 * y - 4 * x * y + 2 * y = 2 * y * (x - 1)^2 := by
  sorry

end factorization_expression1_l4123_412362


namespace jello_bathtub_cost_is_270_l4123_412380

/-- Represents the cost calculation for filling a bathtub with jello. -/
def jello_bathtub_cost (
  jello_mix_per_pound : Real
) (
  bathtub_capacity : Real
) (
  cubic_foot_to_gallon : Real
) (
  gallon_weight : Real
) (
  jello_mix_cost : Real
) : Real :=
  bathtub_capacity * cubic_foot_to_gallon * gallon_weight * jello_mix_per_pound * jello_mix_cost

/-- Theorem stating that the cost to fill the bathtub with jello is $270. -/
theorem jello_bathtub_cost_is_270 :
  jello_bathtub_cost 1.5 6 7.5 8 0.5 = 270 := by
  sorry

#check jello_bathtub_cost_is_270

end jello_bathtub_cost_is_270_l4123_412380


namespace chord_slope_l4123_412317

theorem chord_slope (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + (y-1)^2 = 3 ∧ y = k*x - 1) ∧ 
  (∃ (x1 y1 x2 y2 : ℝ), 
    x1^2 + (y1-1)^2 = 3 ∧ y1 = k*x1 - 1 ∧
    x2^2 + (y2-1)^2 = 3 ∧ y2 = k*x2 - 1 ∧
    (x1-x2)^2 + (y1-y2)^2 = 4) →
  k = 1 ∨ k = -1 :=
by sorry

end chord_slope_l4123_412317


namespace f_at_neg_one_l4123_412315

/-- The function f(x) = x^3 + x^2 - 2x -/
def f (x : ℝ) : ℝ := x^3 + x^2 - 2*x

/-- Theorem: f(-1) = 2 -/
theorem f_at_neg_one : f (-1) = 2 := by
  sorry

end f_at_neg_one_l4123_412315


namespace special_function_sum_l4123_412302

/-- A function satisfying specific properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x^3) = (f x)^3) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ ≠ f x₂)

/-- Theorem stating the sum of f(0), f(1), and f(-1) for a special function -/
theorem special_function_sum (f : ℝ → ℝ) (h : SpecialFunction f) :
  f 0 + f 1 + f (-1) = 0 := by sorry

end special_function_sum_l4123_412302


namespace max_value_of_f_l4123_412342

theorem max_value_of_f (x : ℝ) : 
  ∃ (M : ℝ), M = 2 ∧ ∀ (y : ℝ), min (3 - x^2) (2*x) ≤ M :=
sorry

end max_value_of_f_l4123_412342


namespace system_solution_ratio_l4123_412308

theorem system_solution_ratio (a b x y : ℝ) (h1 : 4 * x - 2 * y = a) 
  (h2 : 9 * y - 18 * x = b) (h3 : b ≠ 0) : a / b = -2 / 9 := by
  sorry

end system_solution_ratio_l4123_412308


namespace candidate_votes_l4123_412395

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) :
  total_votes = 560000 →
  invalid_percent = 15 / 100 →
  candidate_percent = 80 / 100 →
  ∃ (valid_votes : ℕ) (candidate_votes : ℕ),
    valid_votes = (1 - invalid_percent) * total_votes ∧
    candidate_votes = candidate_percent * valid_votes ∧
    candidate_votes = 380800 := by
  sorry

end candidate_votes_l4123_412395


namespace division_problem_l4123_412379

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 167 →
  quotient = 9 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 18 := by
sorry

end division_problem_l4123_412379


namespace parallelogram_area_l4123_412386

/-- Given a parallelogram with height 1 and a right triangle within it with legs 40 and (55 - a),
    where a is the length of the shorter side, prove that its area is 200/3. -/
theorem parallelogram_area (a : ℝ) (h : a > 0) :
  let height : ℝ := 1
  let leg1 : ℝ := 40
  let leg2 : ℝ := 55 - a
  let area : ℝ := a * leg1
  (leg1 ^ 2 + leg2 ^ 2 = (height * area) ^ 2) → area = 200 / 3 :=
by sorry

end parallelogram_area_l4123_412386


namespace ellipse_existence_in_acute_triangle_l4123_412320

/-- Represents an acute triangle -/
structure AcuteTriangle where
  -- Add necessary fields for an acute triangle
  is_acute : Bool

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Orthocenter of a triangle -/
def orthocenter (t : AcuteTriangle) : Point :=
  sorry

/-- Circumcenter of a triangle -/
def circumcenter (t : AcuteTriangle) : Point :=
  sorry

/-- Theorem: For any acute triangle, there exists an ellipse with one focus
    at the orthocenter and the other at the circumcenter of the triangle -/
theorem ellipse_existence_in_acute_triangle (t : AcuteTriangle) :
  ∃ e : Ellipse, e.focus1 = orthocenter t ∧ e.focus2 = circumcenter t :=
by
  sorry

end ellipse_existence_in_acute_triangle_l4123_412320


namespace chairs_to_remove_l4123_412303

theorem chairs_to_remove (initial_chairs : ℕ) (chairs_per_row : ℕ) (expected_students : ℕ)
  (h1 : initial_chairs = 156)
  (h2 : chairs_per_row = 13)
  (h3 : expected_students = 95)
  (h4 : initial_chairs % chairs_per_row = 0) -- All rows are initially completely filled
  : ∃ (removed_chairs : ℕ),
    removed_chairs = 52 ∧
    (initial_chairs - removed_chairs) % chairs_per_row = 0 ∧ -- Remaining rows are completely filled
    (initial_chairs - removed_chairs) ≥ expected_students ∧ -- Can accommodate all students
    ∀ (x : ℕ), x < removed_chairs →
      (initial_chairs - x < expected_students ∨ (initial_chairs - x) % chairs_per_row ≠ 0) -- Minimizes empty seats
    := by sorry

end chairs_to_remove_l4123_412303


namespace subset_implies_t_equals_two_l4123_412360

theorem subset_implies_t_equals_two (t : ℝ) : 
  let A : Set ℝ := {1, t, 2*t}
  let B : Set ℝ := {1, t^2}
  B ⊆ A → t = 2 := by
sorry

end subset_implies_t_equals_two_l4123_412360


namespace min_distance_to_origin_l4123_412306

theorem min_distance_to_origin (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y + 4 = 0) :
  ∃ (min_dist : ℝ), (∀ (a b : ℝ), a^2 + b^2 - 4*a + 6*b + 4 = 0 → 
    min_dist ≤ Real.sqrt (a^2 + b^2)) ∧ min_dist = Real.sqrt 13 - 3 := by
  sorry

end min_distance_to_origin_l4123_412306


namespace ladder_length_l4123_412341

/-- Given a right triangle with an adjacent side of 6.4 meters and an angle of 59.5 degrees
    between the adjacent side and the hypotenuse, the length of the hypotenuse is
    approximately 12.43 meters. -/
theorem ladder_length (adjacent : ℝ) (angle : ℝ) (hypotenuse : ℝ) 
    (h_adjacent : adjacent = 6.4)
    (h_angle : angle = 59.5 * π / 180) -- Convert degrees to radians
    (h_cos : Real.cos angle = adjacent / hypotenuse) :
  abs (hypotenuse - 12.43) < 0.01 := by
  sorry

end ladder_length_l4123_412341


namespace basketball_team_cutoff_l4123_412336

theorem basketball_team_cutoff (girls boys callback : ℕ) 
  (h_girls : girls = 17)
  (h_boys : boys = 32)
  (h_callback : callback = 10) :
  girls + boys - callback = 39 := by
  sorry

end basketball_team_cutoff_l4123_412336


namespace interval_length_implies_difference_l4123_412345

theorem interval_length_implies_difference (r s : ℝ) : 
  (∀ x, r ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ s) → 
  ((s - 4) / 3 - (r - 4) / 3 = 12) → 
  s - r = 36 := by
sorry

end interval_length_implies_difference_l4123_412345


namespace fourth_power_mod_five_l4123_412356

theorem fourth_power_mod_five (a : ℤ) : (a^4) % 5 = 0 ∨ (a^4) % 5 = 1 := by
  sorry

end fourth_power_mod_five_l4123_412356


namespace quadratic_equation_solution_l4123_412301

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 8 * x + 3
  ∃ x₁ x₂ : ℝ, x₁ = 2 + (Real.sqrt 10) / 2 ∧
              x₂ = 2 - (Real.sqrt 10) / 2 ∧
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry


end quadratic_equation_solution_l4123_412301


namespace antonia_pillbox_weeks_l4123_412351

/-- Represents the number of weeks Antonia filled her pillbox -/
def weeks_filled (total_pills : ℕ) (pills_per_week : ℕ) (pills_left : ℕ) : ℕ :=
  (total_pills - pills_left) / pills_per_week

/-- Theorem stating that Antonia filled her pillbox for 2 weeks -/
theorem antonia_pillbox_weeks :
  let num_supplements : ℕ := 5
  let bottles_120 : ℕ := 3
  let bottles_30 : ℕ := 2
  let days_in_week : ℕ := 7
  let pills_left : ℕ := 350

  let total_pills : ℕ := bottles_120 * 120 + bottles_30 * 30
  let pills_per_week : ℕ := num_supplements * days_in_week

  weeks_filled total_pills pills_per_week pills_left = 2 := by
  sorry

#check antonia_pillbox_weeks

end antonia_pillbox_weeks_l4123_412351


namespace queenie_work_days_l4123_412333

/-- Calculates the number of days worked given the daily rate, overtime rate, overtime hours, and total payment -/
def days_worked (daily_rate : ℕ) (overtime_rate : ℕ) (overtime_hours : ℕ) (total_payment : ℕ) : ℕ :=
  (total_payment / (daily_rate + overtime_rate * overtime_hours))

/-- Proves that given the specified conditions, the number of days worked is 4 -/
theorem queenie_work_days : 
  let daily_rate : ℕ := 150
  let overtime_rate : ℕ := 5
  let overtime_hours : ℕ := 4
  let total_payment : ℕ := 770
  days_worked daily_rate overtime_rate overtime_hours total_payment = 4 := by
sorry

#eval days_worked 150 5 4 770

end queenie_work_days_l4123_412333


namespace max_k_minus_m_is_neg_sqrt_two_l4123_412325

/-- A point on a parabola with complementary lines intersecting the parabola -/
structure ParabolaPoint where
  m : ℝ
  k : ℝ
  h1 : m > 0  -- First quadrant condition
  h2 : k = 1 / (-2 * m)  -- Derived from the problem

/-- The maximum value of k - m for a point on the parabola -/
def max_k_minus_m (p : ParabolaPoint) : ℝ := p.k - p.m

/-- Theorem: The maximum value of k - m is -√2 -/
theorem max_k_minus_m_is_neg_sqrt_two :
  ∃ (p : ParabolaPoint), ∀ (q : ParabolaPoint), max_k_minus_m p ≥ max_k_minus_m q ∧ 
  max_k_minus_m p = -Real.sqrt 2 := by
  sorry

end max_k_minus_m_is_neg_sqrt_two_l4123_412325

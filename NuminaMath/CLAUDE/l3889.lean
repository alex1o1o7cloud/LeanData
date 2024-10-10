import Mathlib

namespace square_equation_solution_l3889_388980

theorem square_equation_solution : 
  ∃ x : ℝ, (2010 + x)^2 = 2*x^2 ∧ (x = 4850 ∨ x = -830) :=
by sorry

end square_equation_solution_l3889_388980


namespace odds_against_event_l3889_388985

theorem odds_against_event (odds_in_favor : ℝ) (probability : ℝ) :
  odds_in_favor = 3 →
  probability = 0.375 →
  (odds_in_favor / (odds_in_favor + (odds_against : ℝ)) = probability) →
  odds_against = 5 := by
  sorry

end odds_against_event_l3889_388985


namespace pond_width_proof_l3889_388972

/-- 
Given a rectangular pond with length 20 meters, depth 8 meters, and volume 1600 cubic meters,
prove that its width is 10 meters.
-/
theorem pond_width_proof (length : ℝ) (depth : ℝ) (volume : ℝ) (width : ℝ) 
    (h1 : length = 20)
    (h2 : depth = 8)
    (h3 : volume = 1600)
    (h4 : volume = length * width * depth) : width = 10 := by
  sorry

end pond_width_proof_l3889_388972


namespace root_permutation_l3889_388945

theorem root_permutation (r s t : ℝ) : 
  (r^3 - 21*r + 35 = 0) → 
  (s^3 - 21*s + 35 = 0) → 
  (t^3 - 21*t + 35 = 0) → 
  (r ≠ s) → (s ≠ t) → (t ≠ r) →
  (r^2 + 2*r - 14 = s) ∧ 
  (s^2 + 2*s - 14 = t) ∧ 
  (t^2 + 2*t - 14 = r) := by
sorry

end root_permutation_l3889_388945


namespace circle_radii_ratio_l3889_388937

theorem circle_radii_ratio (A₁ A₂ r₁ r₂ : ℝ) (h_area_ratio : A₁ / A₂ = 98 / 63)
  (h_area_formula₁ : A₁ = π * r₁^2) (h_area_formula₂ : A₂ = π * r₂^2) :
  ∃ (x y z : ℕ), (r₁ / r₂ = x * Real.sqrt y / z) ∧ (x * Real.sqrt y / z = Real.sqrt 14 / 3) ∧ x + y + z = 18 := by
  sorry

end circle_radii_ratio_l3889_388937


namespace customers_per_car_l3889_388998

theorem customers_per_car (num_cars : ℕ) (total_sales : ℕ) 
  (h1 : num_cars = 10) 
  (h2 : total_sales = 50) : 
  ∃ (customers_per_car : ℕ), 
    customers_per_car * num_cars = total_sales ∧ 
    customers_per_car = 5 := by
  sorry

end customers_per_car_l3889_388998


namespace sin_cos_sum_equals_half_l3889_388989

theorem sin_cos_sum_equals_half : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
  Real.sin (70 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end sin_cos_sum_equals_half_l3889_388989


namespace system_solutions_l3889_388923

/-- The system of equations -/
def system (x y z a : ℤ) : Prop :=
  (2*y*z + x - y - z = a) ∧
  (2*x*z - x + y - z = a) ∧
  (2*x*y - x - y + z = a)

/-- Condition for a to have four distinct integer solutions -/
def has_four_solutions (a : ℤ) : Prop :=
  ∃ k : ℤ, k % 2 = 1 ∧ k > 0 ∧ a = (k^2 - 1) / 8

theorem system_solutions (a : ℤ) :
  (¬ ∃ x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃ x₄ y₄ z₄ x₅ y₅ z₅ : ℤ,
    system x₁ y₁ z₁ a ∧ system x₂ y₂ z₂ a ∧ system x₃ y₃ z₃ a ∧
    system x₄ y₄ z₄ a ∧ system x₅ y₅ z₅ a ∧
    (x₁, y₁, z₁) ≠ (x₂, y₂, z₂) ∧ (x₁, y₁, z₁) ≠ (x₃, y₃, z₃) ∧
    (x₁, y₁, z₁) ≠ (x₄, y₄, z₄) ∧ (x₁, y₁, z₁) ≠ (x₅, y₅, z₅) ∧
    (x₂, y₂, z₂) ≠ (x₃, y₃, z₃) ∧ (x₂, y₂, z₂) ≠ (x₄, y₄, z₄) ∧
    (x₂, y₂, z₂) ≠ (x₅, y₅, z₅) ∧ (x₃, y₃, z₃) ≠ (x₄, y₄, z₄) ∧
    (x₃, y₃, z₃) ≠ (x₅, y₅, z₅) ∧ (x₄, y₄, z₄) ≠ (x₅, y₅, z₅)) ∧
  (has_four_solutions a ↔
    ∃ x₁ y₁ z₁ x₂ y₂ z₂ x₃ y₃ z₃ x₄ y₄ z₄ : ℤ,
      system x₁ y₁ z₁ a ∧ system x₂ y₂ z₂ a ∧
      system x₃ y₃ z₃ a ∧ system x₄ y₄ z₄ a ∧
      (x₁, y₁, z₁) ≠ (x₂, y₂, z₂) ∧ (x₁, y₁, z₁) ≠ (x₃, y₃, z₃) ∧
      (x₁, y₁, z₁) ≠ (x₄, y₄, z₄) ∧ (x₂, y₂, z₂) ≠ (x₃, y₃, z₃) ∧
      (x₂, y₂, z₂) ≠ (x₄, y₄, z₄) ∧ (x₃, y₃, z₃) ≠ (x₄, y₄, z₄)) :=
by sorry

end system_solutions_l3889_388923


namespace complement_union_equals_two_five_l3889_388983

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define sets A and B
def A : Set Nat := {1, 4}
def B : Set Nat := {3, 4}

-- Theorem statement
theorem complement_union_equals_two_five :
  (U \ (A ∪ B)) = {2, 5} := by sorry

end complement_union_equals_two_five_l3889_388983


namespace afternoon_sales_l3889_388969

/-- 
Given a salesman who sold pears in the morning and afternoon, 
this theorem proves that if he sold twice as much in the afternoon 
as in the morning, and 420 kilograms in total, then he sold 280 
kilograms in the afternoon.
-/
theorem afternoon_sales 
  (morning_sales : ℕ) 
  (afternoon_sales : ℕ) 
  (h1 : afternoon_sales = 2 * morning_sales) 
  (h2 : morning_sales + afternoon_sales = 420) : 
  afternoon_sales = 280 := by
  sorry

end afternoon_sales_l3889_388969


namespace cube_volume_from_total_edge_length_l3889_388946

/-- The volume of a cube with total edge length of 48 cm is 64 cubic centimeters. -/
theorem cube_volume_from_total_edge_length :
  ∀ (edge_length : ℝ),
  12 * edge_length = 48 →
  edge_length ^ 3 = 64 :=
by
  sorry

end cube_volume_from_total_edge_length_l3889_388946


namespace new_sales_tax_percentage_l3889_388993

theorem new_sales_tax_percentage
  (original_tax : ℝ)
  (market_price : ℝ)
  (savings : ℝ)
  (h1 : original_tax = 3.5)
  (h2 : market_price = 8400)
  (h3 : savings = 14)
  : ∃ (new_tax : ℝ), new_tax = 10/3 ∧ 
    new_tax / 100 * market_price = original_tax / 100 * market_price - savings :=
sorry

end new_sales_tax_percentage_l3889_388993


namespace right_triangle_perimeter_l3889_388968

/-- A right triangle with area 5 and hypotenuse 5 has perimeter 5 + 3√5 -/
theorem right_triangle_perimeter (a b : ℝ) (h_right : a^2 + b^2 = 5^2) 
  (h_area : (1/2) * a * b = 5) : a + b + 5 = 5 + 3 * Real.sqrt 5 := by
  sorry

end right_triangle_perimeter_l3889_388968


namespace largest_negative_integer_congruence_l3889_388979

theorem largest_negative_integer_congruence :
  ∃ x : ℤ, x < 0 ∧ 
    (42 * x + 30) % 24 = 26 % 24 ∧
    x % 12 = (-2) % 12 ∧
    ∀ y : ℤ, y < 0 → (42 * y + 30) % 24 = 26 % 24 → y ≤ x :=
by sorry

end largest_negative_integer_congruence_l3889_388979


namespace coefficient_x4y2_in_expansion_coefficient_equals_60_l3889_388905

/-- The coefficient of x^4y^2 in the expansion of (x-2y)^6 is 60 -/
theorem coefficient_x4y2_in_expansion : ℕ :=
  60

/-- The binomial coefficient "6 choose 2" -/
def binomial_6_2 : ℕ := 15

/-- The expansion of (x-2y)^6 -/
def expansion (x y : ℝ) : ℝ := (x - 2*y)^6

/-- The coefficient of x^4y^2 in the expansion -/
def coefficient (x y : ℝ) : ℝ := binomial_6_2 * (-2)^2

theorem coefficient_equals_60 :
  coefficient = λ _ _ ↦ 60 :=
sorry

end coefficient_x4y2_in_expansion_coefficient_equals_60_l3889_388905


namespace petya_wins_l3889_388943

/-- Represents the game between Petya and Vasya -/
structure CandyGame where
  /-- Total number of candies in both boxes -/
  total_candies : Nat
  /-- Probability of Vasya getting two caramels -/
  prob_two_caramels : ℝ

/-- Petya has a higher chance of winning if his winning probability is greater than 0.5 -/
def petya_has_higher_chance (game : CandyGame) : Prop :=
  1 - (1 - game.prob_two_caramels) > 0.5

/-- Given the conditions of the game, prove that Petya has a higher chance of winning -/
theorem petya_wins (game : CandyGame) 
    (h1 : game.total_candies = 25)
    (h2 : game.prob_two_caramels = 0.54) : 
  petya_has_higher_chance game := by
  sorry

#check petya_wins

end petya_wins_l3889_388943


namespace sage_code_is_8129_l3889_388922

-- Define the mapping of letters to digits
def letter_to_digit : Char → Nat
| 'M' => 0
| 'A' => 1
| 'G' => 2
| 'I' => 3
| 'C' => 4
| 'H' => 5
| 'O' => 6
| 'R' => 7
| 'S' => 8
| 'E' => 9
| _ => 0  -- Default case for other characters

-- Define a function to convert a string to a number
def code_to_number (code : String) : Nat :=
  code.foldl (fun acc c => 10 * acc + letter_to_digit c) 0

-- Theorem statement
theorem sage_code_is_8129 : code_to_number "SAGE" = 8129 := by
  sorry

end sage_code_is_8129_l3889_388922


namespace distance_is_49_l3889_388992

/-- Represents a sign at a kilometer marker -/
structure Sign :=
  (to_yolkino : Nat)
  (to_palkino : Nat)

/-- Calculates the sum of digits of a natural number -/
def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

/-- The distance between Yolkino and Palkino -/
def distance_yolkino_palkino : Nat := sorry

theorem distance_is_49 :
  ∀ (k : Nat), k < distance_yolkino_palkino →
    ∃ (sign : Sign),
      sign.to_yolkino = k ∧
      sign.to_palkino = distance_yolkino_palkino - k ∧
      digit_sum sign.to_yolkino + digit_sum sign.to_palkino = 13 →
  distance_yolkino_palkino = 49 :=
sorry

end distance_is_49_l3889_388992


namespace jellybean_purchase_l3889_388920

theorem jellybean_purchase (n : ℕ) : n ≥ 150 ∧ n % 15 = 14 → n ≥ 164 := by
  sorry

end jellybean_purchase_l3889_388920


namespace davids_english_marks_l3889_388974

theorem davids_english_marks 
  (math_marks : ℕ) 
  (physics_marks : ℕ) 
  (chemistry_marks : ℕ) 
  (biology_marks : ℕ) 
  (average_marks : ℚ) 
  (num_subjects : ℕ) 
  (h1 : math_marks = 35) 
  (h2 : physics_marks = 52) 
  (h3 : chemistry_marks = 47) 
  (h4 : biology_marks = 55) 
  (h5 : average_marks = 46.8) 
  (h6 : num_subjects = 5) : 
  ∃ (english_marks : ℕ), 
    english_marks = 45 ∧ 
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks := by
  sorry

end davids_english_marks_l3889_388974


namespace angle_measure_in_special_quadrilateral_l3889_388964

/-- Given a quadrilateral EFGH with angle relationships ∠E = 2∠F = 4∠G = 5∠H,
    prove that the measure of ∠E is 150°. -/
theorem angle_measure_in_special_quadrilateral (E F G H : ℝ) :
  E = 2 * F ∧ E = 4 * G ∧ E = 5 * H ∧ E + F + G + H = 360 → E = 150 := by
  sorry

end angle_measure_in_special_quadrilateral_l3889_388964


namespace vectors_orthogonality_l3889_388903

/-- Given plane vectors a and b, prove that (a + b) is orthogonal to (a - b) -/
theorem vectors_orthogonality (a b : ℝ × ℝ) 
  (ha : a = (-1/2, Real.sqrt 3/2)) 
  (hb : b = (Real.sqrt 3/2, -1/2)) : 
  (a + b) • (a - b) = 0 := by sorry

end vectors_orthogonality_l3889_388903


namespace tan_value_from_equation_l3889_388939

theorem tan_value_from_equation (x : ℝ) :
  (1 - Real.cos x + Real.sin x) / (1 + Real.cos x + Real.sin x) = -2 →
  Real.tan x = 4 / 3 := by
  sorry

end tan_value_from_equation_l3889_388939


namespace line_perp_parallel_implies_planes_perp_l3889_388906

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (l : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : parallel l β) :
  perpendicular_planes α β :=
sorry

end line_perp_parallel_implies_planes_perp_l3889_388906


namespace unique_number_with_remainders_l3889_388926

theorem unique_number_with_remainders : ∃! n : ℕ, 
  50 < n ∧ n < 70 ∧ n % 5 = 3 ∧ n % 7 = 2 :=
by
  -- The proof would go here
  sorry

end unique_number_with_remainders_l3889_388926


namespace number_with_specific_remainder_l3889_388949

theorem number_with_specific_remainder : ∃ x : ℕ, ∃ k : ℕ, 
  x = 29 * k + 8 ∧ 
  1490 % 29 = 11 ∧ 
  (∀ m : ℕ, m > 29 → (x % m ≠ 8 ∨ 1490 % m ≠ 11)) :=
by sorry

end number_with_specific_remainder_l3889_388949


namespace exists_dry_student_l3889_388952

/-- A student in the water gun game -/
structure Student where
  id : ℕ
  position : ℝ × ℝ

/-- The state of the water gun game -/
structure WaterGunGame where
  n : ℕ
  students : Fin (2 * n + 1) → Student
  distinct_distances : ∀ i j k l : Fin (2 * n + 1), 
    i ≠ j → k ≠ l → (i, j) ≠ (k, l) → 
    dist (students i).position (students j).position ≠ 
    dist (students k).position (students l).position

/-- The shooting function: each student shoots their closest neighbor -/
def shoot (game : WaterGunGame) (shooter : Fin (2 * game.n + 1)) : Fin (2 * game.n + 1) :=
  sorry

/-- The main theorem: there exists a dry student -/
theorem exists_dry_student (game : WaterGunGame) : 
  ∃ s : Fin (2 * game.n + 1), ∀ t : Fin (2 * game.n + 1), shoot game t ≠ s :=
sorry

end exists_dry_student_l3889_388952


namespace optimal_invitation_strategy_l3889_388984

/-- Represents an invitation strategy for a social gathering. -/
structure InvitationStrategy where
  total_acquaintances : Nat
  ladies : Nat
  gentlemen : Nat
  ladies_per_invite : Nat
  gentlemen_per_invite : Nat
  invitations : Nat

/-- Checks if the invitation strategy is valid and optimal. -/
def is_valid_and_optimal (strategy : InvitationStrategy) : Prop :=
  strategy.total_acquaintances = strategy.ladies + strategy.gentlemen
  ∧ strategy.ladies_per_invite + strategy.gentlemen_per_invite < strategy.total_acquaintances
  ∧ strategy.invitations * strategy.ladies_per_invite ≥ strategy.ladies * (strategy.total_acquaintances - 1)
  ∧ strategy.invitations * strategy.gentlemen_per_invite ≥ strategy.gentlemen * (strategy.total_acquaintances - 1)
  ∧ ∀ n : Nat, n < strategy.invitations →
    n * strategy.ladies_per_invite < strategy.ladies * (strategy.total_acquaintances - 1)
    ∨ n * strategy.gentlemen_per_invite < strategy.gentlemen * (strategy.total_acquaintances - 1)

theorem optimal_invitation_strategy :
  ∃ (strategy : InvitationStrategy),
    strategy.total_acquaintances = 20
    ∧ strategy.ladies = 9
    ∧ strategy.gentlemen = 11
    ∧ strategy.ladies_per_invite = 3
    ∧ strategy.gentlemen_per_invite = 2
    ∧ strategy.invitations = 11
    ∧ is_valid_and_optimal strategy
    ∧ (strategy.invitations * strategy.ladies_per_invite) / strategy.ladies = 7
    ∧ (strategy.invitations * strategy.gentlemen_per_invite) / strategy.gentlemen = 2 :=
  sorry

end optimal_invitation_strategy_l3889_388984


namespace sqrt_range_l3889_388921

theorem sqrt_range (x : ℝ) : x ∈ {y : ℝ | ∃ (z : ℝ), z^2 = y - 7} ↔ x ≥ 7 := by
  sorry

end sqrt_range_l3889_388921


namespace xy_value_l3889_388976

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end xy_value_l3889_388976


namespace two_number_difference_l3889_388981

theorem two_number_difference (x y : ℝ) (h1 : x + y = 50) (h2 : 3 * y - 3 * x = 27) : |y - x| = 9 := by
  sorry

end two_number_difference_l3889_388981


namespace perfect_square_condition_l3889_388941

theorem perfect_square_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + 2*(m-3)*x + 25 = y^2) → (m = 8 ∨ m = -2) := by
  sorry

end perfect_square_condition_l3889_388941


namespace problem_statement_l3889_388910

/-- Given points P, Q, and O, and a function f, prove properties about f and a related triangle -/
theorem problem_statement 
  (P : ℝ × ℝ) 
  (Q : ℝ → ℝ × ℝ) 
  (f : ℝ → ℝ) 
  (A : ℝ) 
  (BC : ℝ) 
  (h1 : P = (Real.sqrt 3, 1))
  (h2 : ∀ x, Q x = (Real.cos x, Real.sin x))
  (h3 : ∀ x, f x = P.1 * (Q x).1 + P.2 * (Q x).2 - ((Q x).1 * P.1 + (Q x).2 * P.2))
  (h4 : f A = 4)
  (h5 : BC = 3) :
  (∀ x, f x = -2 * Real.sin (x + π/3) + 4) ∧ 
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) ∧
  (∀ a b c, a + b + c ≤ 3 + 2 * Real.sqrt 3) :=
by sorry

end problem_statement_l3889_388910


namespace investment_principal_l3889_388917

/-- Proves that given an investment with a monthly interest payment of $216 and a simple annual interest rate of 9%, the principal amount of the investment is $28,800. -/
theorem investment_principal (monthly_interest : ℝ) (annual_rate : ℝ) :
  monthly_interest = 216 →
  annual_rate = 0.09 →
  (monthly_interest * 12) / annual_rate = 28800 := by
  sorry

end investment_principal_l3889_388917


namespace expression_equivalence_l3889_388932

theorem expression_equivalence : 
  -44 + 1010 + 66 - 55 = (-44) + 1010 + 66 + (-55) := by
  sorry

end expression_equivalence_l3889_388932


namespace annes_speed_ratio_l3889_388958

/-- Proves that the ratio of Anne's new cleaning rate to her original rate is 2:1 --/
theorem annes_speed_ratio :
  -- Bruce and Anne's original combined rate
  ∀ (B A : ℚ), B + A = 1/4 →
  -- Anne's original rate
  A = 1/12 →
  -- Bruce and Anne's new combined rate (with Anne's changed speed)
  ∀ (A' : ℚ), B + A' = 1/3 →
  -- The ratio of Anne's new rate to her original rate
  A' / A = 2 := by
sorry

end annes_speed_ratio_l3889_388958


namespace quadratic_equation_roots_l3889_388988

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x y : ℝ, x^2 + (k^2 - 4)*x + k - 1 = 0 ∧ 
               y^2 + (k^2 - 4)*y + k - 1 = 0 ∧ 
               x = -y) → 
  k = -2 :=
by sorry

end quadratic_equation_roots_l3889_388988


namespace parallelogram_base_length_l3889_388929

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude_base_relation : ℝ → ℝ → Prop) :
  area = 242 →
  (∀ base height, altitude_base_relation base height → height = 2 * base) →
  ∃ base : ℝ, 
    altitude_base_relation base (2 * base) ∧ 
    area = base * (2 * base) ∧ 
    base = 11 := by
sorry

end parallelogram_base_length_l3889_388929


namespace money_difference_l3889_388908

/-- Given an initial amount of money and an additional amount received, 
    prove that the difference between the final amount and the initial amount 
    is equal to the additional amount received. -/
theorem money_difference (initial additional : ℕ) : 
  (initial + additional) - initial = additional := by
  sorry

end money_difference_l3889_388908


namespace cannot_be_B_l3889_388942

-- Define set A
def A : Set ℝ := {x : ℝ | x ≠ 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x < -1}

-- Theorem statement
theorem cannot_be_B (h : A ∪ B = Set.univ) : False := by
  sorry

end cannot_be_B_l3889_388942


namespace graph_is_single_line_l3889_388957

-- Define the function representing the equation
def f (x y : ℝ) : Prop := (x - 1)^2 * (x + y - 2) = (y - 1)^2 * (x + y - 2)

-- Theorem stating that the graph of f is a single line
theorem graph_is_single_line :
  ∃! (m b : ℝ), ∀ x y : ℝ, f x y ↔ y = m * x + b :=
sorry

end graph_is_single_line_l3889_388957


namespace smallest_k_sum_squares_div_150_l3889_388913

/-- The sum of squares from 1 to n -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- 100 is the smallest positive integer k such that the sum of squares from 1 to k is divisible by 150 -/
theorem smallest_k_sum_squares_div_150 :
  ∀ k : ℕ, k > 0 → k < 100 → ¬(150 ∣ sum_of_squares k) ∧ (150 ∣ sum_of_squares 100) :=
by sorry

end smallest_k_sum_squares_div_150_l3889_388913


namespace circle_diameter_from_area_l3889_388987

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) :
  A = 4 * Real.pi →  -- given area
  A = Real.pi * r^2 →  -- definition of circle area
  d = 2 * r →  -- definition of diameter
  d = 4 := by
  sorry

end circle_diameter_from_area_l3889_388987


namespace min_product_of_three_l3889_388982

def S : Set Int := {-10, -7, -5, 0, 4, 6, 9}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  x * y * z ≤ a * b * c ∧ x * y * z = -540 := by
  sorry

end min_product_of_three_l3889_388982


namespace tangent_parallel_points_l3889_388991

-- Define the function f(x) = x³ + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_parallel_points :
  {p : ℝ × ℝ | p.1 = 1 ∧ p.2 = 0 ∨ p.1 = -1 ∧ p.2 = -4} =
  {p : ℝ × ℝ | f p.1 = p.2 ∧ f' p.1 = 4} :=
by sorry

end tangent_parallel_points_l3889_388991


namespace greatest_divisor_of_28_l3889_388925

theorem greatest_divisor_of_28 : ∃ d : ℕ, d ∣ 28 ∧ ∀ x : ℕ, x ∣ 28 → x ≤ d :=
by sorry

end greatest_divisor_of_28_l3889_388925


namespace cannot_determine_unique_ages_l3889_388953

-- Define variables for Julie and Aaron's ages
variable (J A : ℕ)

-- Define the relationship between their current ages
def current_age_relation : Prop := J = 4 * A

-- Define the relationship between their ages in 10 years
def future_age_relation : Prop := J + 10 = 4 * (A + 10)

-- Theorem stating that unique ages cannot be determined
theorem cannot_determine_unique_ages 
  (h1 : current_age_relation J A) 
  (h2 : future_age_relation J A) :
  ∃ (J' A' : ℕ), J' ≠ J ∧ A' ≠ A ∧ current_age_relation J' A' ∧ future_age_relation J' A' :=
sorry

end cannot_determine_unique_ages_l3889_388953


namespace polynomial_factorization_l3889_388971

theorem polynomial_factorization (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := by
  sorry

end polynomial_factorization_l3889_388971


namespace trigonometric_inequality_l3889_388962

theorem trigonometric_inequality : ∃ (a b c : ℝ),
  a = Real.tan (3 * Real.pi / 4) ∧
  b = Real.cos (2 * Real.pi / 5) ∧
  c = (1 + Real.sin (6 * Real.pi / 5)) ^ 0 ∧
  c > b ∧ b > a := by sorry

end trigonometric_inequality_l3889_388962


namespace function_value_determines_a_l3889_388915

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

-- State the theorem
theorem function_value_determines_a (a : ℝ) : f a (f a 0) = 3*a → a = 4 := by
  sorry

end function_value_determines_a_l3889_388915


namespace solution_sum_l3889_388927

theorem solution_sum (m : ℝ) (x₁ x₂ : ℝ) : 
  (∃ (m : ℝ), 2 * Real.sin (2 * x₁ + π / 6) = m ∧ 
               2 * Real.sin (2 * x₂ + π / 6) = m) →
  x₁ ≠ x₂ →
  x₁ ∈ Set.Icc 0 (π / 2) →
  x₂ ∈ Set.Icc 0 (π / 2) →
  x₁ + x₂ = π / 3 := by
sorry

end solution_sum_l3889_388927


namespace not_all_F_zero_on_C_implies_exists_F_zero_not_on_C_l3889_388955

-- Define the curve C and the function F
variable (C : Set (ℝ × ℝ))
variable (F : ℝ → ℝ → ℝ)

-- Define the set of points satisfying F(x, y) = 0
def F_zero_set (F : ℝ → ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | F p.1 p.2 = 0}

-- State the theorem
theorem not_all_F_zero_on_C_implies_exists_F_zero_not_on_C
  (h : ¬(F_zero_set F ⊆ C)) :
  ∃ p : ℝ × ℝ, p ∉ C ∧ F p.1 p.2 = 0 := by
  sorry

end not_all_F_zero_on_C_implies_exists_F_zero_not_on_C_l3889_388955


namespace cubic_function_properties_l3889_388918

def f (a b m x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + m

def f_deriv (a b x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

theorem cubic_function_properties (a b m : ℝ) :
  (∀ x : ℝ, f_deriv a b x = f_deriv a b (-1 - x)) →
  f_deriv a b 1 = 0 →
  (a = 3 ∧ b = -12) ∧
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    f 3 (-12) m x₁ = 0 ∧ f 3 (-12) m x₂ = 0 ∧ f 3 (-12) m x₃ = 0 ∧
    ∀ x : ℝ, f 3 (-12) m x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  -20 < m ∧ m < 7 :=
by sorry

end cubic_function_properties_l3889_388918


namespace sphere_to_cone_height_l3889_388947

theorem sphere_to_cone_height (R : ℝ) (h : ℝ) (r : ℝ) (l : ℝ) : 
  R > 0 → r > 0 → h > 0 → l > 0 →
  (4 / 3) * Real.pi * R^3 = (1 / 3) * Real.pi * r^2 * h →  -- Volume conservation
  Real.pi * r * l = 3 * Real.pi * r^2 →  -- Lateral surface area condition
  l^2 = r^2 + h^2 →  -- Pythagorean theorem
  h = 4 * R * Real.sqrt 2 :=
by sorry

end sphere_to_cone_height_l3889_388947


namespace water_ratio_is_half_l3889_388900

/-- Represents the water flow problem --/
structure WaterFlow where
  flow_rate_1 : ℚ  -- Flow rate for the first hour (cups per 10 minutes)
  flow_rate_2 : ℚ  -- Flow rate for the second hour (cups per 10 minutes)
  duration_1 : ℚ   -- Duration of first flow rate (hours)
  duration_2 : ℚ   -- Duration of second flow rate (hours)
  water_left : ℚ   -- Amount of water left after dumping (cups)

/-- Calculates the total water collected before dumping --/
def total_water (wf : WaterFlow) : ℚ :=
  wf.flow_rate_1 * 6 * wf.duration_1 + wf.flow_rate_2 * 6 * wf.duration_2

/-- Theorem stating the ratio of water left to total water collected is 1/2 --/
theorem water_ratio_is_half (wf : WaterFlow) 
  (h1 : wf.flow_rate_1 = 2)
  (h2 : wf.flow_rate_2 = 4)
  (h3 : wf.duration_1 = 1)
  (h4 : wf.duration_2 = 1)
  (h5 : wf.water_left = 18) :
  wf.water_left / total_water wf = 1 / 2 := by
  sorry

end water_ratio_is_half_l3889_388900


namespace alcohol_solution_proof_l3889_388948

/-- Proves that adding 1.2 liters of pure alcohol to a 6-liter solution
    that is 40% alcohol results in a 50% alcohol solution -/
theorem alcohol_solution_proof (initial_volume : ℝ) (initial_concentration : ℝ)
    (added_alcohol : ℝ) (final_concentration : ℝ)
    (h1 : initial_volume = 6)
    (h2 : initial_concentration = 0.4)
    (h3 : added_alcohol = 1.2)
    (h4 : final_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) /
  (initial_volume + added_alcohol) = final_concentration :=
by sorry

end alcohol_solution_proof_l3889_388948


namespace no_triangle_solution_l3889_388994

theorem no_triangle_solution (a b c : ℝ) (A B C : ℝ) :
  b = 4 →
  c = 2 →
  C = π / 3 →
  ¬ (∃ (a : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (A > 0 ∧ B > 0 ∧ C > 0) ∧
    (A + B + C = π) ∧
    (a / Real.sin A = b / Real.sin B) ∧
    (b / Real.sin B = c / Real.sin C) ∧
    (c / Real.sin C = a / Real.sin A)) :=
by sorry

end no_triangle_solution_l3889_388994


namespace pencil_sharpening_l3889_388916

/-- The length sharpened off a pencil is the difference between its original length and its length after sharpening. -/
theorem pencil_sharpening (original_length after_sharpening_length : ℝ) 
  (h1 : original_length = 31.25)
  (h2 : after_sharpening_length = 14.75) :
  original_length - after_sharpening_length = 16.5 := by
  sorry

end pencil_sharpening_l3889_388916


namespace wall_bricks_l3889_388963

/-- Represents the number of bricks in the wall -/
def num_bricks : ℕ := 288

/-- Represents the time taken by the first bricklayer to build the wall alone -/
def time_bricklayer1 : ℕ := 8

/-- Represents the time taken by the second bricklayer to build the wall alone -/
def time_bricklayer2 : ℕ := 12

/-- Represents the reduction in combined output when working together -/
def output_reduction : ℕ := 12

/-- Represents the time taken by both bricklayers working together -/
def time_together : ℕ := 6

/-- Theorem stating that the number of bricks in the wall is 288 -/
theorem wall_bricks :
  (time_together : ℚ) * ((num_bricks / time_bricklayer1 : ℚ) + 
  (num_bricks / time_bricklayer2 : ℚ) - output_reduction) = num_bricks := by
  sorry

#eval num_bricks

end wall_bricks_l3889_388963


namespace correct_calculation_l3889_388907

theorem correct_calculation (x : ℤ) : 
  x - 749 = 280 → x + 479 = 1508 := by
  sorry

end correct_calculation_l3889_388907


namespace satisfactory_fraction_is_25_29_l3889_388961

/-- Represents the grade distribution in a school science class -/
structure GradeDistribution :=
  (a : Nat) -- number of A grades
  (b : Nat) -- number of B grades
  (c : Nat) -- number of C grades
  (d : Nat) -- number of D grades
  (f : Nat) -- number of F grades

/-- Calculates the fraction of satisfactory grades -/
def satisfactoryFraction (gd : GradeDistribution) : Rat :=
  let satisfactory := gd.a + gd.b + gd.c + gd.d
  let total := satisfactory + gd.f
  satisfactory / total

/-- The main theorem stating that the fraction of satisfactory grades is 25/29 -/
theorem satisfactory_fraction_is_25_29 :
  let gd : GradeDistribution := ⟨8, 7, 6, 4, 4⟩
  satisfactoryFraction gd = 25 / 29 := by
  sorry

end satisfactory_fraction_is_25_29_l3889_388961


namespace min_payment_bound_l3889_388996

/-- Tea set price in yuan -/
def tea_set_price : ℕ := 200

/-- Tea bowl price in yuan -/
def tea_bowl_price : ℕ := 20

/-- Number of tea sets purchased -/
def num_tea_sets : ℕ := 30

/-- Discount factor for Option 2 -/
def discount_factor : ℚ := 95 / 100

/-- Payment for Option 1 given x tea bowls -/
def payment_option1 (x : ℕ) : ℕ := 20 * x + 5400

/-- Payment for Option 2 given x tea bowls -/
def payment_option2 (x : ℕ) : ℕ := 19 * x + 5700

/-- Theorem: The minimum payment is less than or equal to the minimum of Option 1 and Option 2 -/
theorem min_payment_bound (x : ℕ) (hx : x > 30) :
  ∃ (y : ℕ), y ≤ min (payment_option1 x) (payment_option2 x) ∧
  y = num_tea_sets * tea_set_price + x * tea_bowl_price -
      (min num_tea_sets x) * tea_bowl_price +
      ((x - min num_tea_sets x) * tea_bowl_price * discount_factor).floor :=
sorry

end min_payment_bound_l3889_388996


namespace quadratic_equation_properties_l3889_388999

theorem quadratic_equation_properties (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*(m-1)*x - m*(m+2) = 0 → ∃ y : ℝ, y ≠ x ∧ y^2 - 2*(m-1)*y - m*(m+2) = 0) ∧
  ((-2)^2 - 2*(m-1)*(-2) - m*(m+2) = 0 → 2018 - 3*(m-1)^2 = 2015) := by
  sorry

end quadratic_equation_properties_l3889_388999


namespace decimal_to_binary_38_l3889_388938

theorem decimal_to_binary_38 : 
  (38 : ℕ).digits 2 = [0, 1, 1, 0, 0, 1] :=
by sorry

end decimal_to_binary_38_l3889_388938


namespace race_head_start_l3889_388914

/-- Calculates the head start given in a race between two runners with different speeds -/
def headStart (cristinaSpeed nicky_speed : ℝ) (catchUpTime : ℝ) : ℝ :=
  nicky_speed * catchUpTime

theorem race_head_start :
  let cristinaSpeed : ℝ := 5
  let nickySpeed : ℝ := 3
  let catchUpTime : ℝ := 27
  headStart cristinaSpeed nickySpeed catchUpTime = 81 := by
sorry

end race_head_start_l3889_388914


namespace sum_of_roots_is_eight_l3889_388935

theorem sum_of_roots_is_eight : 
  ∃ (N₁ N₂ : ℝ), N₁ * (N₁ - 8) = 4 ∧ N₂ * (N₂ - 8) = 4 ∧ N₁ + N₂ = 8 := by
  sorry

end sum_of_roots_is_eight_l3889_388935


namespace complex_magnitude_squared_l3889_388944

theorem complex_magnitude_squared (a b : ℝ) (z : ℂ) : 
  z = Complex.mk a b → z + Complex.abs z = 3 + 7*Complex.I → Complex.abs z^2 = 841/9 := by
  sorry

end complex_magnitude_squared_l3889_388944


namespace sara_remaining_marbles_l3889_388966

def initial_black_marbles : ℕ := 792
def marbles_taken : ℕ := 233

theorem sara_remaining_marbles : 
  initial_black_marbles - marbles_taken = 559 := by sorry

end sara_remaining_marbles_l3889_388966


namespace quadratic_equation_roots_l3889_388954

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  x₁^2 - 2023*x₁ - 1 = 0 ∧ x₂^2 - 2023*x₂ - 1 = 0 := by
  sorry

end quadratic_equation_roots_l3889_388954


namespace sufficient_not_necessary_condition_l3889_388967

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 1 → x > 0) ∧ (∃ x, x > 0 ∧ ¬(x > 1)) := by sorry

end sufficient_not_necessary_condition_l3889_388967


namespace binomial_expansion_coefficient_relation_l3889_388909

theorem binomial_expansion_coefficient_relation (n : ℕ) : 
  (n.choose 2 * 3^(n-2) = 5 * n.choose 0 * 3^n) → n = 10 := by
  sorry

end binomial_expansion_coefficient_relation_l3889_388909


namespace negative_three_to_fourth_equals_three_to_fourth_l3889_388959

theorem negative_three_to_fourth_equals_three_to_fourth : (-3) * (-3) * (-3) * (-3) = 3^4 := by
  sorry

end negative_three_to_fourth_equals_three_to_fourth_l3889_388959


namespace fourth_member_income_l3889_388990

def family_size : ℕ := 4
def average_income : ℕ := 10000
def member1_income : ℕ := 8000
def member2_income : ℕ := 15000
def member3_income : ℕ := 6000

theorem fourth_member_income :
  let total_income := family_size * average_income
  let known_members_income := member1_income + member2_income + member3_income
  total_income - known_members_income = 11000 := by
  sorry

end fourth_member_income_l3889_388990


namespace value_of_a_is_one_l3889_388977

/-- Two circles with a common chord of length 2√3 -/
structure TwoCirclesWithCommonChord where
  a : ℝ
  h1 : a > 0
  h2 : ∃ (x y : ℝ), x^2 + y^2 = 4 ∧ x^2 + y^2 + 2*a*y - 6 = 0
  h3 : ∃ (x1 y1 x2 y2 : ℝ),
    (x1^2 + y1^2 = 4 ∧ x1^2 + y1^2 + 2*a*y1 - 6 = 0) ∧
    (x2^2 + y2^2 = 4 ∧ x2^2 + y2^2 + 2*a*y2 - 6 = 0) ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 12

/-- The value of a is 1 for two circles with a common chord of length 2√3 -/
theorem value_of_a_is_one (c : TwoCirclesWithCommonChord) : c.a = 1 :=
  sorry

end value_of_a_is_one_l3889_388977


namespace profit_maximized_at_optimal_price_l3889_388936

/-- Profit function given selling price -/
def profit (x : ℝ) : ℝ := (x - 40) * (1000 - 10 * x)

/-- The selling price that maximizes profit -/
def optimal_price : ℝ := 70

theorem profit_maximized_at_optimal_price :
  ∀ x : ℝ, profit x ≤ profit optimal_price :=
sorry

end profit_maximized_at_optimal_price_l3889_388936


namespace regular_polygon_sides_l3889_388933

/-- Theorem: A regular polygon with exterior angles of 40° has 9 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 40 → n * exterior_angle = 360 → n = 9 := by
  sorry

end regular_polygon_sides_l3889_388933


namespace stratified_sampling_result_l3889_388951

/-- Represents the total number of teachers -/
def total_teachers : ℕ := 150

/-- Represents the number of senior teachers -/
def senior_teachers : ℕ := 15

/-- Represents the number of intermediate teachers -/
def intermediate_teachers : ℕ := 90

/-- Represents the number of teachers sampled -/
def sampled_teachers : ℕ := 30

/-- Represents the number of junior teachers -/
def junior_teachers : ℕ := total_teachers - senior_teachers - intermediate_teachers

/-- Theorem stating the correct numbers of teachers selected in each category -/
theorem stratified_sampling_result :
  (senior_teachers * sampled_teachers / total_teachers = 3) ∧
  (intermediate_teachers * sampled_teachers / total_teachers = 18) ∧
  (junior_teachers * sampled_teachers / total_teachers = 9) :=
sorry

end stratified_sampling_result_l3889_388951


namespace surface_area_ratio_l3889_388950

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  /-- The edge length of the tetrahedron -/
  edge_length : ℝ
  /-- Assumption that the edge length is positive -/
  edge_positive : edge_length > 0

/-- The surface area of a regular tetrahedron -/
def surface_area_tetrahedron (t : RegularTetrahedron) : ℝ := sorry

/-- The surface area of the inscribed sphere of a regular tetrahedron -/
def surface_area_inscribed_sphere (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the ratio of the surface areas -/
theorem surface_area_ratio (t : RegularTetrahedron) :
  surface_area_tetrahedron t / surface_area_inscribed_sphere t = 6 * Real.sqrt 3 / Real.pi := by
  sorry

end surface_area_ratio_l3889_388950


namespace max_value_of_expression_l3889_388975

theorem max_value_of_expression (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  (a * b + b * c + c * d + d * e) / (2 * a^2 + b^2 + 2 * c^2 + d^2 + 2 * e^2) ≤ Real.sqrt (3 / 8) ∧
  ∃ a b c d e : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧
    (a * b + b * c + c * d + d * e) / (2 * a^2 + b^2 + 2 * c^2 + d^2 + 2 * e^2) = Real.sqrt (3 / 8) :=
by sorry

end max_value_of_expression_l3889_388975


namespace multiples_properties_l3889_388970

theorem multiples_properties (x y : ℤ) 
  (hx : ∃ k : ℤ, x = 5 * k) 
  (hy : ∃ m : ℤ, y = 10 * m) : 
  (∃ n : ℤ, y = 5 * n) ∧ 
  (∃ p : ℤ, x - y = 5 * p) ∧ 
  (∃ q : ℤ, y - x = 5 * q) := by
sorry

end multiples_properties_l3889_388970


namespace square_position_after_2007_transformations_l3889_388919

/-- Represents the vertices of a square in clockwise order -/
inductive SquarePosition
  | ABCD
  | DABC
  | CBAD
  | DCBA

/-- Applies one full cycle of transformations to a square -/
def applyTransformationCycle (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.ABCD
  | SquarePosition.DABC => SquarePosition.DABC
  | SquarePosition.CBAD => SquarePosition.CBAD
  | SquarePosition.DCBA => SquarePosition.DCBA

/-- Applies n cycles of transformations to a square -/
def applyNCycles (pos : SquarePosition) (n : Nat) : SquarePosition :=
  match n with
  | 0 => pos
  | n + 1 => applyNCycles (applyTransformationCycle pos) n

/-- Applies a specific number of individual transformations to a square -/
def applyTransformations (pos : SquarePosition) (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => pos
  | 1 => match pos with
         | SquarePosition.ABCD => SquarePosition.DABC
         | SquarePosition.DABC => SquarePosition.CBAD
         | SquarePosition.CBAD => SquarePosition.DCBA
         | SquarePosition.DCBA => SquarePosition.ABCD
  | 2 => match pos with
         | SquarePosition.ABCD => SquarePosition.CBAD
         | SquarePosition.DABC => SquarePosition.DCBA
         | SquarePosition.CBAD => SquarePosition.ABCD
         | SquarePosition.DCBA => SquarePosition.DABC
  | 3 => match pos with
         | SquarePosition.ABCD => SquarePosition.DCBA
         | SquarePosition.DABC => SquarePosition.ABCD
         | SquarePosition.CBAD => SquarePosition.DABC
         | SquarePosition.DCBA => SquarePosition.CBAD
  | _ => pos  -- This case should never occur due to % 4

theorem square_position_after_2007_transformations :
  applyTransformations SquarePosition.ABCD 2007 = SquarePosition.DCBA :=
by sorry

end square_position_after_2007_transformations_l3889_388919


namespace red_jellybean_count_l3889_388911

/-- Given a jar of jellybeans with specific counts for different colors, 
    prove that the number of red jellybeans is 120. -/
theorem red_jellybean_count (total : ℕ) (blue purple orange : ℕ) 
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_purple : purple = 26)
  (h_orange : orange = 40) :
  total - (blue + purple + orange) = 120 := by
  sorry

end red_jellybean_count_l3889_388911


namespace georgie_guacamole_servings_l3889_388934

/-- The number of servings of guacamole Georgie can make -/
def guacamole_servings (avocados_per_serving : ℕ) (initial_avocados : ℕ) (bought_avocados : ℕ) : ℕ :=
  (initial_avocados + bought_avocados) / avocados_per_serving

/-- Theorem: Georgie can make 3 servings of guacamole -/
theorem georgie_guacamole_servings :
  guacamole_servings 3 5 4 = 3 := by
  sorry

end georgie_guacamole_servings_l3889_388934


namespace sum_inequality_l3889_388995

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  (1 / (b * c + a + 1 / a) + 1 / (a * c + b + 1 / b) + 1 / (a * b + c + 1 / c)) ≤ 27 / 31 := by
  sorry

end sum_inequality_l3889_388995


namespace arithmetic_sequence_sum_l3889_388960

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) :
  (∀ k, a (k + 1) - a k = a 1 - a 0) →  -- arithmetic sequence condition
  a 0 = 3 →                            -- first term is 3
  a n = 39 →                           -- last term is 39
  n ≥ 2 →                              -- ensure at least 3 terms
  a (n - 1) + a (n - 2) = 72 :=         -- sum of last two terms before 39
by
  sorry

end arithmetic_sequence_sum_l3889_388960


namespace prize_problem_l3889_388930

-- Define the types of prizes
inductive PrizeType
| A
| B

-- Define the unit prices
def unit_price : PrizeType → ℕ
| PrizeType.A => 10
| PrizeType.B => 15

-- Define the total number of prizes
def total_prizes : ℕ := 100

-- Define the maximum total cost
def max_total_cost : ℕ := 1160

-- Define the condition for the quantity of type A prizes
def type_a_condition (a b : ℕ) : Prop := a ≤ 3 * b

-- Define the cost function
def cost (a b : ℕ) : ℕ := a * unit_price PrizeType.A + b * unit_price PrizeType.B

-- Define the valid purchasing plan
def valid_plan (a b : ℕ) : Prop :=
  a + b = total_prizes ∧
  cost a b ≤ max_total_cost ∧
  type_a_condition a b

-- Theorem statement
theorem prize_problem :
  (3 * unit_price PrizeType.A + 2 * unit_price PrizeType.B = 60) ∧
  (unit_price PrizeType.A + 3 * unit_price PrizeType.B = 55) ∧
  (∃ (plans : List (ℕ × ℕ)), 
    plans.length = 8 ∧
    (∀ p ∈ plans, valid_plan p.1 p.2) ∧
    (∃ (a b : ℕ), (a, b) ∈ plans ∧ cost a b = 1125 ∧ 
      ∀ (x y : ℕ), valid_plan x y → cost x y ≥ 1125)) :=
by sorry

end prize_problem_l3889_388930


namespace wire_length_l3889_388901

/-- Given a wire cut into three pieces in the ratio of 7:3:2, where the shortest piece is 16 cm long,
    the total length of the wire before it was cut is 96 cm. -/
theorem wire_length (ratio_long ratio_medium ratio_short : ℕ) 
  (shortest_piece : ℝ) (h1 : ratio_long = 7) (h2 : ratio_medium = 3) 
  (h3 : ratio_short = 2) (h4 : shortest_piece = 16) : 
  (ratio_long + ratio_medium + ratio_short) * (shortest_piece / ratio_short) = 96 := by
  sorry

end wire_length_l3889_388901


namespace wilson_oldest_child_age_wilson_oldest_child_age_proof_l3889_388902

/-- The age of the oldest Wilson child given the average age and the ages of the two younger children -/
theorem wilson_oldest_child_age 
  (average_age : ℝ) 
  (younger_child1_age : ℕ) 
  (younger_child2_age : ℕ) 
  (h1 : average_age = 8) 
  (h2 : younger_child1_age = 5) 
  (h3 : younger_child2_age = 8) : 
  ℕ := 
  11

theorem wilson_oldest_child_age_proof :
  let oldest_child_age := wilson_oldest_child_age 8 5 8 rfl rfl rfl
  (8 : ℝ) = (5 + 8 + oldest_child_age) / 3 := by
  sorry

end wilson_oldest_child_age_wilson_oldest_child_age_proof_l3889_388902


namespace cosine_sine_relation_l3889_388912

open Real

theorem cosine_sine_relation (α β : ℝ) (x y : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : cos (α + β) = -4/5)
  (h4 : sin β = x)
  (h5 : cos α = y)
  (h6 : 4/5 < x ∧ x < 1) :
  y = -4/5 * sqrt (1 - x^2) + 3/5 * x := by
sorry

end cosine_sine_relation_l3889_388912


namespace evaluate_expression_l3889_388997

theorem evaluate_expression : 49^2 - 25^2 + 10^2 = 1876 := by
  sorry

end evaluate_expression_l3889_388997


namespace a_2017_equals_2_l3889_388956

def S (n : ℕ+) : ℕ := 2 * n - 1

def a (n : ℕ+) : ℕ := S n - S (n - 1)

theorem a_2017_equals_2 : a 2017 = 2 := by sorry

end a_2017_equals_2_l3889_388956


namespace parabola_point_coordinate_l3889_388931

/-- The x-coordinate of a point on the parabola y^2 = 6x that is twice as far from the focus as from the y-axis -/
theorem parabola_point_coordinate :
  ∀ (x y : ℝ),
  y^2 = 6*x →
  (x + 3/2)^2 + y^2 = 4 * x^2 →
  x = 3/2 := by
sorry

end parabola_point_coordinate_l3889_388931


namespace work_completion_time_l3889_388924

/-- The number of days it takes A to complete the work -/
def days_A : ℝ := 4

/-- The number of days it takes C to complete the work -/
def days_C : ℝ := 8

/-- The number of days it takes A, B, and C together to complete the work -/
def days_ABC : ℝ := 2

/-- The number of days it takes B to complete the work -/
def days_B : ℝ := 8

theorem work_completion_time :
  (1 / days_A + 1 / days_B + 1 / days_C = 1 / days_ABC) :=
by sorry

end work_completion_time_l3889_388924


namespace debit_card_advantage_l3889_388986

/-- Represents the benefit of using a credit card for N days -/
def credit_card_benefit (N : ℕ) : ℚ :=
  20 * N + 120

/-- Represents the benefit of using a debit card -/
def debit_card_benefit : ℚ := 240

/-- The maximum number of days for which using the debit card is more advantageous -/
def max_days_debit_advantageous : ℕ := 6

theorem debit_card_advantage :
  ∀ N : ℕ, N ≤ max_days_debit_advantageous ↔ debit_card_benefit ≥ credit_card_benefit N :=
by sorry

#check debit_card_advantage

end debit_card_advantage_l3889_388986


namespace drone_velocity_at_3_seconds_l3889_388940

-- Define the displacement function
def h (t : ℝ) : ℝ := 15 * t - t^2

-- Define the velocity function as the derivative of the displacement function
def v (t : ℝ) : ℝ := 15 - 2 * t

-- Theorem statement
theorem drone_velocity_at_3_seconds :
  v 3 = 9 := by sorry

end drone_velocity_at_3_seconds_l3889_388940


namespace max_value_implies_a_l3889_388978

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * (x - 2)^2

theorem max_value_implies_a (a : ℝ) (h1 : a ≠ 0) :
  (∃ (M : ℝ), M = 16/9 ∧ ∀ (x : ℝ), f a x ≤ M) → a = 3/2 := by
  sorry

end max_value_implies_a_l3889_388978


namespace women_who_left_l3889_388965

theorem women_who_left (initial_men initial_women : ℕ) 
  (h1 : initial_men * 5 = initial_women * 4)
  (h2 : initial_men + 2 = 14)
  (h3 : 2 * (initial_women - 3) = 24) : 
  3 = initial_women - (24 / 2) :=
by sorry

end women_who_left_l3889_388965


namespace sixth_term_geometric_sequence_l3889_388928

/-- The sixth term of a geometric sequence with first term 5 and second term 1.25 is 5/1024 -/
theorem sixth_term_geometric_sequence (a₁ a₂ : ℝ) (h₁ : a₁ = 5) (h₂ : a₂ = 1.25) :
  let r := a₂ / a₁
  let a₆ := a₁ * r^5
  a₆ = 5 / 1024 := by
sorry

end sixth_term_geometric_sequence_l3889_388928


namespace miriam_pushups_l3889_388904

/-- Miriam's push-up challenge over a week --/
theorem miriam_pushups (monday tuesday wednesday thursday friday : ℕ) : 
  monday = 5 ∧ 
  wednesday = 2 * tuesday ∧
  thursday = (monday + tuesday + wednesday) / 2 ∧
  friday = monday + tuesday + wednesday + thursday ∧
  friday = 39 →
  tuesday = 7 := by
  sorry

end miriam_pushups_l3889_388904


namespace probability_four_twos_correct_l3889_388973

def num_dice : ℕ := 12
def num_sides : ℕ := 6
def num_success : ℕ := 4

def probability_exactly_four_twos : ℚ :=
  (Nat.choose num_dice num_success) *
  (1 / num_sides) ^ num_success *
  ((num_sides - 1) / num_sides) ^ (num_dice - num_success)

theorem probability_four_twos_correct :
  probability_exactly_four_twos = 
    (Nat.choose num_dice num_success) *
    (1 / num_sides) ^ num_success *
    ((num_sides - 1) / num_sides) ^ (num_dice - num_success) :=
by sorry

end probability_four_twos_correct_l3889_388973

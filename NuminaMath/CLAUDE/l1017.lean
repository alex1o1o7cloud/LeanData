import Mathlib

namespace NUMINAMATH_CALUDE_cindy_walking_speed_l1017_101742

/-- Cindy's running speed in miles per hour -/
def running_speed : ℝ := 3

/-- Distance Cindy runs in miles -/
def run_distance : ℝ := 0.5

/-- Distance Cindy walks in miles -/
def walk_distance : ℝ := 0.5

/-- Total time for the journey in minutes -/
def total_time : ℝ := 40

/-- Cindy's walking speed in miles per hour -/
def walking_speed : ℝ := 1

theorem cindy_walking_speed :
  running_speed = 3 ∧
  run_distance = 0.5 ∧
  walk_distance = 0.5 ∧
  total_time = 40 →
  walking_speed = 1 := by sorry

end NUMINAMATH_CALUDE_cindy_walking_speed_l1017_101742


namespace NUMINAMATH_CALUDE_min_folds_to_exceed_target_l1017_101791

def paper_thickness : ℝ := 0.1
def target_thickness : ℝ := 12

def thickness_after_folds (n : ℕ) : ℝ :=
  paper_thickness * (2 ^ n)

theorem min_folds_to_exceed_target : 
  ∀ n : ℕ, (thickness_after_folds n > target_thickness) ↔ (n ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_min_folds_to_exceed_target_l1017_101791


namespace NUMINAMATH_CALUDE_interest_rate_proof_l1017_101744

/-- 
Given a principal sum and an annual interest rate,
if the simple interest for 4 years is one-fifth of the principal,
then the annual interest rate is 5%.
-/
theorem interest_rate_proof (P R : ℝ) (P_pos : P > 0) : 
  (P * R * 4) / 100 = P / 5 → R = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l1017_101744


namespace NUMINAMATH_CALUDE_count_bases_with_final_digit_one_l1017_101735

/-- The number of bases between 2 and 12 (inclusive) where 625 in base 10 has a final digit of 1 -/
def count_bases : ℕ := 7

/-- The set of bases between 2 and 12 (inclusive) where 625 in base 10 has a final digit of 1 -/
def valid_bases : Finset ℕ := {2, 3, 4, 6, 8, 9, 12}

theorem count_bases_with_final_digit_one :
  (Finset.range 11).filter (fun b => 625 % (b + 2) = 1) = valid_bases ∧
  valid_bases.card = count_bases :=
sorry

end NUMINAMATH_CALUDE_count_bases_with_final_digit_one_l1017_101735


namespace NUMINAMATH_CALUDE_apple_pies_count_l1017_101770

def total_apple_weight : ℕ := 120
def applesauce_fraction : ℚ := 1/2
def pounds_per_pie : ℕ := 4

theorem apple_pies_count :
  (total_apple_weight * (1 - applesauce_fraction) / pounds_per_pie : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_pies_count_l1017_101770


namespace NUMINAMATH_CALUDE_wall_volume_l1017_101728

/-- Proves that the volume of a rectangular wall with given dimensions is 6804 cubic units -/
theorem wall_volume : 
  ∀ (width height length : ℕ),
  width = 3 →
  height = 6 * width →
  length = 7 * height →
  width * height * length = 6804 := by
  sorry

end NUMINAMATH_CALUDE_wall_volume_l1017_101728


namespace NUMINAMATH_CALUDE_cafe_order_combinations_l1017_101710

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- Theorem: The number of ways two people can each choose one item from a set of 15 items,
    where order matters and repetition is allowed, is equal to 225. -/
theorem cafe_order_combinations :
  menu_items ^ num_people = 225 := by sorry

end NUMINAMATH_CALUDE_cafe_order_combinations_l1017_101710


namespace NUMINAMATH_CALUDE_ace_spade_probability_l1017_101773

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Probability of drawing an Ace as the first card and a spade as the second card -/
def prob_ace_then_spade : ℚ :=
  (NumAces / StandardDeck) * (NumSpades / (StandardDeck - 1))

theorem ace_spade_probability :
  prob_ace_then_spade = 3 / 127 := by
  sorry

end NUMINAMATH_CALUDE_ace_spade_probability_l1017_101773


namespace NUMINAMATH_CALUDE_orange_basket_problem_l1017_101717

/-- 
Given:
- When 2 oranges are put in each basket, 4 oranges are left over.
- When 5 oranges are put in each basket, 1 basket is left over.

Prove that the number of baskets is 3 and the number of oranges is 10.
-/
theorem orange_basket_problem (b o : ℕ) 
  (h1 : 2 * b + 4 = o) 
  (h2 : 5 * (b - 1) = o) : 
  b = 3 ∧ o = 10 := by
  sorry


end NUMINAMATH_CALUDE_orange_basket_problem_l1017_101717


namespace NUMINAMATH_CALUDE_andrey_gifts_l1017_101725

theorem andrey_gifts :
  ∃ (n : ℕ) (a : ℕ),
    n > 2 ∧
    n * (n - 2) = a * (n - 1) + 16 ∧
    n = 18 :=
by sorry

end NUMINAMATH_CALUDE_andrey_gifts_l1017_101725


namespace NUMINAMATH_CALUDE_solve_equation_l1017_101701

theorem solve_equation (x : ℚ) : 5 * (2 * x - 3) = 3 * (3 - 4 * x) + 15 → x = 39 / 22 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1017_101701


namespace NUMINAMATH_CALUDE_sequence_ratio_l1017_101715

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that (a₂ - a₁) / b₂ = 1/2 -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (-2 : ℝ) - a₁ = a₁ - a₂ ∧ 
  a₂ - a₁ = a₁ - (-8 : ℝ) ∧
  (-2 : ℝ) * b₁ = b₁ * b₂ ∧
  b₁ * b₂ = b₂ * b₃ ∧
  b₂ * b₃ = b₃ * (-8 : ℝ) →
  (a₂ - a₁) / b₂ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_ratio_l1017_101715


namespace NUMINAMATH_CALUDE_hotel_rooms_for_couples_l1017_101789

theorem hotel_rooms_for_couples :
  let single_rooms : ℕ := 14
  let bubble_bath_per_bath : ℕ := 10
  let total_bubble_bath : ℕ := 400
  let baths_per_single_room : ℕ := 1
  let baths_per_couple_room : ℕ := 2
  ∃ couple_rooms : ℕ,
    couple_rooms = 13 ∧
    total_bubble_bath = bubble_bath_per_bath * (single_rooms * baths_per_single_room + couple_rooms * baths_per_couple_room) :=
by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_for_couples_l1017_101789


namespace NUMINAMATH_CALUDE_volunteer_arrangement_l1017_101750

theorem volunteer_arrangement (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → 
  (Nat.choose n k) * (Nat.choose (n - k) k) = 140 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_arrangement_l1017_101750


namespace NUMINAMATH_CALUDE_quadratic_root_factorization_l1017_101708

theorem quadratic_root_factorization 
  (a₀ a₁ a₂ x r s : ℝ) 
  (h₁ : a₂ ≠ 0) 
  (h₂ : a₀ ≠ 0) 
  (h₃ : a₀ + a₁ * r + a₂ * r^2 = 0) 
  (h₄ : a₀ + a₁ * s + a₂ * s^2 = 0) :
  a₀ + a₁ * x + a₂ * x^2 = a₀ * (1 - x / r) * (1 - x / s) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_factorization_l1017_101708


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_half_l1017_101793

theorem arctan_sum_equals_pi_half (a b : ℝ) (h1 : a = 1/3) (h2 : (a+1)*(b+1) = 3) :
  Real.arctan a + Real.arctan b = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_half_l1017_101793


namespace NUMINAMATH_CALUDE_joan_football_games_l1017_101769

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The total number of football games Joan went to this year and last year -/
def total_games : ℕ := 9

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := total_games - games_this_year

theorem joan_football_games : games_last_year = 5 := by
  sorry

end NUMINAMATH_CALUDE_joan_football_games_l1017_101769


namespace NUMINAMATH_CALUDE_negation_of_no_red_cards_negation_equivalent_to_some_red_cards_l1017_101740

-- Define the universe of cards
variable (U : Type)

-- Define the property of being a red card
variable (red : U → Prop)

-- Define the property of being in the deck
variable (in_deck : U → Prop)

-- Statement to be proven
theorem negation_of_no_red_cards (h : ¬∃ x, red x ∧ in_deck x) :
  ¬∀ x, red x → ¬in_deck x :=
sorry

-- Proof that the negation is equivalent to "Some red cards are in this deck"
theorem negation_equivalent_to_some_red_cards :
  (¬∀ x, red x → ¬in_deck x) ↔ (∃ x, red x ∧ in_deck x) :=
sorry

end NUMINAMATH_CALUDE_negation_of_no_red_cards_negation_equivalent_to_some_red_cards_l1017_101740


namespace NUMINAMATH_CALUDE_anna_age_proof_l1017_101799

/-- Anna's current age -/
def anna_age : ℕ := 54

/-- Clara's current age -/
def clara_age : ℕ := 80

/-- Years in the past -/
def years_ago : ℕ := 41

theorem anna_age_proof :
  anna_age = 54 ∧
  clara_age = 80 ∧
  clara_age - years_ago = 3 * (anna_age - years_ago) :=
sorry

end NUMINAMATH_CALUDE_anna_age_proof_l1017_101799


namespace NUMINAMATH_CALUDE_min_dot_product_on_hyperbola_l1017_101757

/-- The minimum dot product of two points on the hyperbola x² - y² = 2 -/
theorem min_dot_product_on_hyperbola :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  x₁ > 0 → x₂ > 0 →
  x₁^2 - y₁^2 = 2 →
  x₂^2 - y₂^2 = 2 →
  x₁ * x₂ + y₁ * y₂ ≥ 2 ∧
  ∃ (x₁' y₁' x₂' y₂' : ℝ),
    x₁' > 0 ∧ x₂' > 0 ∧
    x₁'^2 - y₁'^2 = 2 ∧
    x₂'^2 - y₂'^2 = 2 ∧
    x₁' * x₂' + y₁' * y₂' = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_on_hyperbola_l1017_101757


namespace NUMINAMATH_CALUDE_arrangements_count_l1017_101781

/-- The number of ways to arrange 5 distinct objects in a row, 
    where two specific objects are not allowed to be adjacent -/
def arrangements_with_restriction : ℕ := 72

/-- Theorem stating that the number of arrangements with the given restriction is 72 -/
theorem arrangements_count : arrangements_with_restriction = 72 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l1017_101781


namespace NUMINAMATH_CALUDE_xy_value_l1017_101729

theorem xy_value (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : (x - y)^2 = 9) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1017_101729


namespace NUMINAMATH_CALUDE_five_integer_chords_l1017_101724

/-- A circle with a point P inside --/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- The number of chords with integer lengths passing through P --/
def numIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The theorem statement --/
theorem five_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 17) 
  (h2 : c.distanceFromCenter = 8) : 
  numIntegerChords c = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_integer_chords_l1017_101724


namespace NUMINAMATH_CALUDE_range_of_m_l1017_101772

/-- The piecewise function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < m then x else x^2 + 4*x

/-- The property that for all p < m, there exists q ≥ m such that f(p) + f(q) = 0 -/
def property (m : ℝ) : Prop :=
  ∀ p < m, ∃ q ≥ m, f m p + f m q = 0

/-- The theorem stating the range of m -/
theorem range_of_m : ∀ m : ℝ, property m ↔ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1017_101772


namespace NUMINAMATH_CALUDE_tomatoes_left_l1017_101749

/-- Given 21 initial tomatoes and birds eating one-third of them, prove that 14 tomatoes are left -/
theorem tomatoes_left (initial : ℕ) (eaten_fraction : ℚ) (h1 : initial = 21) (h2 : eaten_fraction = 1/3) :
  initial - (initial * eaten_fraction).floor = 14 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_l1017_101749


namespace NUMINAMATH_CALUDE_xiaoming_mother_retirement_year_l1017_101721

/-- Calculates the retirement year based on the given retirement plan --/
def calculate_retirement_year (birth_year : ℕ) : ℕ :=
  let original_retirement_year := birth_year + 55
  if original_retirement_year ≥ 2018 ∧ original_retirement_year < 2021
  then original_retirement_year + 1
  else original_retirement_year

/-- Theorem stating that Xiaoming's mother's retirement year is 2020 --/
theorem xiaoming_mother_retirement_year :
  calculate_retirement_year 1964 = 2020 :=
by sorry

end NUMINAMATH_CALUDE_xiaoming_mother_retirement_year_l1017_101721


namespace NUMINAMATH_CALUDE_club_membership_l1017_101754

theorem club_membership (total_members : ℕ) (attendance : ℕ) (men : ℕ) (women : ℕ) : 
  total_members = 30 →
  attendance = 18 →
  total_members = men + women →
  attendance = men + (women / 3) →
  men = 12 := by
sorry

end NUMINAMATH_CALUDE_club_membership_l1017_101754


namespace NUMINAMATH_CALUDE_bushes_needed_for_zucchinis_l1017_101741

/-- Represents the number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 10

/-- Represents the number of containers of blueberries that can be traded for zucchinis -/
def containers_traded : ℕ := 6

/-- Represents the number of zucchinis received in trade for containers_traded -/
def zucchinis_received : ℕ := 3

/-- Represents the target number of zucchinis -/
def target_zucchinis : ℕ := 60

/-- Theorem stating that 12 bushes are needed to obtain 60 zucchinis -/
theorem bushes_needed_for_zucchinis : 
  (target_zucchinis * containers_traded) / (zucchinis_received * containers_per_bush) = 12 :=
sorry

end NUMINAMATH_CALUDE_bushes_needed_for_zucchinis_l1017_101741


namespace NUMINAMATH_CALUDE_construction_materials_sum_l1017_101777

theorem construction_materials_sum : 
  0.17 + 0.237 + 0.646 + 0.5 + 1.73 + 0.894 = 4.177 := by
  sorry

end NUMINAMATH_CALUDE_construction_materials_sum_l1017_101777


namespace NUMINAMATH_CALUDE_second_crew_tractors_second_crew_is_seven_l1017_101705

/-- Calculates the number of tractors in the second crew given the farming conditions --/
theorem second_crew_tractors (total_acres : ℕ) (total_days : ℕ) (first_crew_tractors : ℕ) 
  (first_crew_days : ℕ) (second_crew_days : ℕ) (acres_per_day : ℕ) : ℕ :=
  let first_crew_acres := first_crew_tractors * first_crew_days * acres_per_day
  let remaining_acres := total_acres - first_crew_acres
  let acres_per_tractor := second_crew_days * acres_per_day
  remaining_acres / acres_per_tractor

/-- Proves that the number of tractors in the second crew is 7 --/
theorem second_crew_is_seven : 
  second_crew_tractors 1700 5 2 2 3 68 = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_crew_tractors_second_crew_is_seven_l1017_101705


namespace NUMINAMATH_CALUDE_min_sum_proof_l1017_101763

/-- The minimum sum of m and n satisfying the conditions -/
def min_sum : ℕ := 106

/-- The value of m in the minimal solution -/
def m_min : ℕ := 3

/-- The value of n in the minimal solution -/
def n_min : ℕ := 103

/-- Checks if two numbers are congruent modulo 1000 -/
def congruent_mod_1000 (a b : ℕ) : Prop :=
  a % 1000 = b % 1000

theorem min_sum_proof :
  ∀ m n : ℕ,
    n > m →
    m ≥ 1 →
    congruent_mod_1000 (1978^n) (1978^m) →
    m + n ≥ min_sum ∧
    (m + n = min_sum → m = m_min ∧ n = n_min) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_proof_l1017_101763


namespace NUMINAMATH_CALUDE_ben_sandwich_options_l1017_101719

/-- Represents the number of different types for each sandwich component -/
structure SandwichOptions where
  bread : Nat
  meat : Nat
  cheese : Nat

/-- Represents specific sandwich combinations that are not allowed -/
structure ForbiddenCombinations where
  beef_swiss : Nat
  rye_turkey : Nat
  turkey_swiss : Nat

/-- Calculates the number of sandwich options given the available choices and forbidden combinations -/
def calculate_sandwich_options (options : SandwichOptions) (forbidden : ForbiddenCombinations) : Nat :=
  options.bread * options.meat * options.cheese - (forbidden.beef_swiss + forbidden.rye_turkey + forbidden.turkey_swiss)

/-- The main theorem stating the number of different sandwiches Ben could order -/
theorem ben_sandwich_options :
  let options : SandwichOptions := { bread := 5, meat := 7, cheese := 6 }
  let forbidden : ForbiddenCombinations := { beef_swiss := 5, rye_turkey := 6, turkey_swiss := 5 }
  calculate_sandwich_options options forbidden = 194 := by
  sorry

end NUMINAMATH_CALUDE_ben_sandwich_options_l1017_101719


namespace NUMINAMATH_CALUDE_raj_ate_ten_bananas_l1017_101768

/-- The number of bananas Raj ate -/
def bananas_eaten (initial_bananas : ℕ) (remaining_bananas : ℕ) : ℕ :=
  initial_bananas - remaining_bananas - 2 * remaining_bananas

/-- Theorem stating that Raj ate 10 bananas -/
theorem raj_ate_ten_bananas :
  bananas_eaten 310 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_raj_ate_ten_bananas_l1017_101768


namespace NUMINAMATH_CALUDE_expression_simplification_l1017_101790

theorem expression_simplification (y : ℝ) : 
  4 * y - 3 * y^3 + 6 - (1 - 4 * y + 3 * y^3) = -6 * y^3 + 8 * y + 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1017_101790


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1017_101718

theorem perfect_square_condition (a b : ℕ+) (h_b_odd : Odd b.val) 
  (h_int : ∃ k : ℤ, ((a.val + b.val)^2 + 4*a.val : ℤ) = k * (a.val * b.val)) :
  ∃ n : ℕ+, a = n^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1017_101718


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l1017_101767

/-- For a geometric progression with first term b₁, common ratio q, n-th term bₙ, and sum of first n terms Sₙ, 
    the ratio (Sₙ - bₙ) / (Sₙ - b₁) is equal to 1/q for all q -/
theorem geometric_progression_ratio (n : ℕ) (b₁ q : ℝ) : 
  let bₙ := b₁ * q^(n - 1)
  let Sₙ := if q ≠ 1 then b₁ * (q^n - 1) / (q - 1) else n * b₁
  (Sₙ - bₙ) / (Sₙ - b₁) = 1 / q :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l1017_101767


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l1017_101737

theorem dining_bill_calculation (people : ℕ) (tip_percentage : ℚ) (individual_share : ℚ) :
  people = 8 →
  tip_percentage = 1/10 →
  individual_share = 191125/10000 →
  ∃ (original_bill : ℚ), 
    (original_bill * (1 + tip_percentage)) / people = individual_share ∧
    original_bill = 139 :=
by sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l1017_101737


namespace NUMINAMATH_CALUDE_perimeter_of_C_l1017_101743

-- Define squares A, B, and C
def square_A : Real → Real := λ s ↦ 4 * s
def square_B : Real → Real := λ s ↦ 4 * s
def square_C : Real → Real := λ s ↦ 4 * s

-- Define the conditions
def perimeter_A : Real := 20
def perimeter_B : Real := 36

-- Define the side length of C as the difference between side lengths of A and B
def side_C (a b : Real) : Real := b - a

-- Theorem statement
theorem perimeter_of_C : 
  ∀ (a b : Real),
  square_A a = perimeter_A →
  square_B b = perimeter_B →
  square_C (side_C a b) = 16 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_C_l1017_101743


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1017_101795

theorem no_solution_for_equation : ¬∃ (x : ℝ), (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1017_101795


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l1017_101758

theorem triangle_angle_sum (a b c : ℝ) (h1 : a + b + c = 180) 
                           (h2 : a = 85) (h3 : b = 35) : c = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l1017_101758


namespace NUMINAMATH_CALUDE_pizza_area_increase_l1017_101738

/-- The radius of the larger pizza in inches -/
def r1 : ℝ := 5

/-- The radius of the smaller pizza in inches -/
def r2 : ℝ := 2

/-- The percentage increase in area from the smaller pizza to the larger pizza -/
def M : ℝ := 525

theorem pizza_area_increase :
  (π * r1^2 - π * r2^2) / (π * r2^2) * 100 = M :=
sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l1017_101738


namespace NUMINAMATH_CALUDE_line_equation_two_points_l1017_101714

/-- The equation of a line passing through two points -/
theorem line_equation_two_points (x₁ y₁ x₂ y₂ x y : ℝ) :
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁) ↔ 
  (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ 
  ∃ t : ℝ, x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁) :=
sorry

end NUMINAMATH_CALUDE_line_equation_two_points_l1017_101714


namespace NUMINAMATH_CALUDE_solution_characterization_l1017_101727

def SolutionSet : Set (ℕ × ℕ × ℕ) := {(1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 3, 2), (3, 5, 4), (2, 1, 1), (2, 1, 3), (4, 3, 5), (5, 4, 3), (3, 2, 1)}

def DivisibilityCondition (x y z : ℕ) : Prop :=
  (x ∣ y + 1) ∧ (y ∣ z + 1) ∧ (z ∣ x + 1)

theorem solution_characterization :
  ∀ x y z : ℕ, (x > 0 ∧ y > 0 ∧ z > 0) →
    (DivisibilityCondition x y z ↔ (x, y, z) ∈ SolutionSet) := by
  sorry

end NUMINAMATH_CALUDE_solution_characterization_l1017_101727


namespace NUMINAMATH_CALUDE_cafeteria_apples_l1017_101760

/-- Calculates the number of apples bought by the cafeteria -/
def apples_bought (initial : ℕ) (used : ℕ) (final : ℕ) : ℕ :=
  final - (initial - used)

/-- Proves that the cafeteria bought 6 apples -/
theorem cafeteria_apples : apples_bought 23 20 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l1017_101760


namespace NUMINAMATH_CALUDE_unsold_books_l1017_101792

theorem unsold_books (total : ℕ) (sold : ℕ) (price : ℕ) (revenue : ℕ) : 
  (2 : ℕ) * sold = 3 * total ∧ 
  price = 4 ∧ 
  revenue = 288 ∧ 
  sold * price = revenue → 
  total - sold = 36 := by
  sorry

end NUMINAMATH_CALUDE_unsold_books_l1017_101792


namespace NUMINAMATH_CALUDE_sum_of_abc_l1017_101794

theorem sum_of_abc (a b c : ℕ+) (h1 : a * b + c = 31)
                   (h2 : b * c + a = 31) (h3 : a * c + b = 31) :
  (a : ℕ) + b + c = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l1017_101794


namespace NUMINAMATH_CALUDE_one_element_set_l1017_101786

def A (k : ℝ) : Set ℝ := {x | k * x^2 - 4 * x + 2 = 0}

theorem one_element_set (k : ℝ) :
  (∃! x, x ∈ A k) → (k = 0 ∧ A k = {1/2}) ∨ (k = 2 ∧ A k = {1}) := by
  sorry

end NUMINAMATH_CALUDE_one_element_set_l1017_101786


namespace NUMINAMATH_CALUDE_preimage_of_four_l1017_101739

def f (x : ℝ) : ℝ := x^2

theorem preimage_of_four (x : ℝ) : f x = 4 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_four_l1017_101739


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l1017_101722

theorem smallest_constant_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x / (y + 2 * z)) + Real.sqrt (y / (2 * x + z)) + Real.sqrt (z / (x + 2 * y)) > Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l1017_101722


namespace NUMINAMATH_CALUDE_maria_oatmeal_cookies_l1017_101766

/-- The number of oatmeal cookies Maria had -/
def num_oatmeal_cookies (cookies_per_bag : ℕ) (num_chocolate_chip : ℕ) (num_baggies : ℕ) : ℕ :=
  num_baggies * cookies_per_bag - num_chocolate_chip

/-- Theorem stating that Maria had 2 oatmeal cookies -/
theorem maria_oatmeal_cookies :
  num_oatmeal_cookies 5 33 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_maria_oatmeal_cookies_l1017_101766


namespace NUMINAMATH_CALUDE_arithmetic_sequence_contains_2017_l1017_101764

/-- An arithmetic sequence containing 25, 41, and 65 also contains 2017 -/
theorem arithmetic_sequence_contains_2017 (a₁ d : ℤ) (k n m : ℕ) 
  (h_pos : d > 0)
  (h_25 : 25 = a₁ + k * d)
  (h_41 : 41 = a₁ + n * d)
  (h_65 : 65 = a₁ + m * d) :
  ∃ l : ℕ, 2017 = a₁ + l * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_contains_2017_l1017_101764


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1017_101703

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 2 * I / (1 + I)
  Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1017_101703


namespace NUMINAMATH_CALUDE_sequence_problem_l1017_101765

-- Define the arithmetic sequence
def is_arithmetic_sequence (x y z w : ℝ) : Prop :=
  y - x = z - y ∧ z - y = w - z

-- Define the geometric sequence
def is_geometric_sequence (x y z w v : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r ∧ w = z * r ∧ v = w * r

theorem sequence_problem (a b c d e : ℝ) :
  is_arithmetic_sequence (-1) a b (-4) →
  is_geometric_sequence (-1) c d e (-4) →
  c = -1 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_problem_l1017_101765


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1017_101720

theorem quadratic_roots_relation (p q : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) ∧ 
  (x₂^2 + p*x₂ + q = 0) ∧ 
  ((x₁ + 1)^2 + q*(x₁ + 1) + p = 0) ∧ 
  ((x₂ + 1)^2 + q*(x₂ + 1) + p = 0) →
  p = -1 ∧ q = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1017_101720


namespace NUMINAMATH_CALUDE_range_of_a_l1017_101706

/-- Definition of the circle D -/
def circle_D (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1

/-- Definition of point B -/
def point_B : ℝ × ℝ := (-1, 0)

/-- Definition of point C -/
def point_C (a : ℝ) : ℝ × ℝ := (a, 0)

/-- Theorem stating the range of a -/
theorem range_of_a (A B C : ℝ × ℝ) (a : ℝ) :
  (∃ x y, A = (x, y) ∧ circle_D x y) →  -- A lies on circle D
  B = point_B →                         -- B is at (-1, 0)
  C = point_C a →                       -- C is at (a, 0)
  (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0 →  -- Right angle at A
  14/5 ≤ a ∧ a ≤ 16/3 :=                -- Range of a
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1017_101706


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1017_101776

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (2 * x < 3 * x - 10) → x ≥ 11 ∧ 2 * 11 < 3 * 11 - 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1017_101776


namespace NUMINAMATH_CALUDE_linear_equation_condition_l1017_101782

/-- If (a+1)x + 3y^|a| = 1 is a linear equation in x and y, then a = 1 -/
theorem linear_equation_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ k m : ℝ, (a + 1) * x + 3 * y^(|a|) = k * x + m * y + 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l1017_101782


namespace NUMINAMATH_CALUDE_field_planted_fraction_l1017_101774

theorem field_planted_fraction (a b d : ℝ) (ha : a > 0) (hb : b > 0) (hd : d > 0) :
  let c := (a^2 + b^2).sqrt
  let x := (a * b * d) / (a^2 + b^2)
  let triangle_area := a * b / 2
  let square_area := x^2
  let planted_area := triangle_area - square_area
  a = 5 → b = 12 → d = 3 →
  planted_area / triangle_area = 52761 / 857430 := by
  sorry

end NUMINAMATH_CALUDE_field_planted_fraction_l1017_101774


namespace NUMINAMATH_CALUDE_rachels_winter_clothing_l1017_101751

theorem rachels_winter_clothing (boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) 
  (h1 : boxes = 7)
  (h2 : scarves_per_box = 3)
  (h3 : mittens_per_box = 4) :
  boxes * (scarves_per_box + mittens_per_box) = 49 :=
by sorry

end NUMINAMATH_CALUDE_rachels_winter_clothing_l1017_101751


namespace NUMINAMATH_CALUDE_part_a_part_b_l1017_101747

-- Define the set M of functions satisfying the given conditions
def M : Set (ℤ → ℝ) :=
  {f | f 0 ≠ 0 ∧ ∀ n m : ℤ, f n * f m = f (n + m) + f (n - m)}

-- Theorem for part (a)
theorem part_a (f : ℤ → ℝ) (hf : f ∈ M) (h1 : f 1 = 5/2) :
  ∀ n : ℤ, f n = 2^n + 2^(-n) := by sorry

-- Theorem for part (b)
theorem part_b (f : ℤ → ℝ) (hf : f ∈ M) (h1 : f 1 = Real.sqrt 3) :
  ∀ n : ℤ, f n = 2 * Real.cos (π * n / 6) := by sorry

end NUMINAMATH_CALUDE_part_a_part_b_l1017_101747


namespace NUMINAMATH_CALUDE_infinite_sum_n_over_n4_plus_1_l1017_101733

/-- The infinite sum of n / (n^4 + 1) from n = 1 to infinity equals 1. -/
theorem infinite_sum_n_over_n4_plus_1 : 
  ∑' n : ℕ+, (n : ℝ) / ((n : ℝ)^4 + 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_infinite_sum_n_over_n4_plus_1_l1017_101733


namespace NUMINAMATH_CALUDE_trisha_money_theorem_l1017_101771

/-- The amount of money Trisha spent on meat -/
def meat_cost : ℕ := 17

/-- The amount of money Trisha spent on chicken -/
def chicken_cost : ℕ := 22

/-- The amount of money Trisha spent on veggies -/
def veggies_cost : ℕ := 43

/-- The amount of money Trisha spent on eggs -/
def eggs_cost : ℕ := 5

/-- The amount of money Trisha spent on dog's food -/
def dog_food_cost : ℕ := 45

/-- The amount of money Trisha had left after shopping -/
def money_left : ℕ := 35

/-- The total amount of money Trisha brought at the beginning -/
def total_money : ℕ := meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost + money_left

theorem trisha_money_theorem : total_money = 167 := by
  sorry

end NUMINAMATH_CALUDE_trisha_money_theorem_l1017_101771


namespace NUMINAMATH_CALUDE_jackies_activities_exceed_day_l1017_101783

/-- Represents the duration of Jackie's daily activities in hours -/
structure DailyActivities where
  working : ℝ
  exercising : ℝ
  sleeping : ℝ
  commuting : ℝ
  meals : ℝ
  language_classes : ℝ
  phone_calls : ℝ
  reading : ℝ

/-- Theorem stating that Jackie's daily activities exceed 24 hours -/
theorem jackies_activities_exceed_day (activities : DailyActivities) 
  (h1 : activities.working = 8)
  (h2 : activities.exercising = 3)
  (h3 : activities.sleeping = 8)
  (h4 : activities.commuting = 1)
  (h5 : activities.meals = 2)
  (h6 : activities.language_classes = 1.5)
  (h7 : activities.phone_calls = 0.5)
  (h8 : activities.reading = 40 / 60) :
  activities.working + activities.exercising + activities.sleeping + 
  activities.commuting + activities.meals + activities.language_classes + 
  activities.phone_calls + activities.reading > 24 := by
  sorry

#check jackies_activities_exceed_day

end NUMINAMATH_CALUDE_jackies_activities_exceed_day_l1017_101783


namespace NUMINAMATH_CALUDE_consecutive_integers_permutation_divisibility_l1017_101796

theorem consecutive_integers_permutation_divisibility
  (p : ℕ) (h_prime : Nat.Prime p)
  (m : ℕ → ℕ) (h_consecutive : ∀ i ∈ Finset.range p, m (i + 1) = m i + 1)
  (σ : Fin p → Fin p) (h_perm : Function.Bijective σ) :
  ∃ (k l : Fin p), k ≠ l ∧ p ∣ (m k * m (σ k) - m l * m (σ l)) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_permutation_divisibility_l1017_101796


namespace NUMINAMATH_CALUDE_farm_distance_problem_l1017_101745

/-- Represents the distances between three farms -/
structure FarmDistances where
  x : ℝ  -- Distance between first and second farms
  y : ℝ  -- Distance between second and third farms
  z : ℝ  -- Distance between first and third farms

/-- Theorem stating the conditions and results for the farm distance problem -/
theorem farm_distance_problem (a : ℝ) : 
  ∃ (d : FarmDistances), 
    d.x + d.y = 4 * d.z ∧                   -- Condition 1
    d.z + d.y = d.x + a ∧                   -- Condition 2
    d.x + d.z = 85 ∧                        -- Condition 3
    0 < a ∧ a < 85 ∧                        -- Interval for a
    d.x = (340 - a) / 6 ∧                   -- Distance x
    d.y = (2 * a + 85) / 3 ∧                -- Distance y
    d.z = (170 + a) / 6 ∧                   -- Distance z
    d.x + d.y > d.z ∧ d.y + d.z > d.x ∧ d.z + d.x > d.y -- Triangle inequality
    := by sorry

end NUMINAMATH_CALUDE_farm_distance_problem_l1017_101745


namespace NUMINAMATH_CALUDE_cost_price_percentage_l1017_101734

theorem cost_price_percentage (cost_price selling_price : ℝ) (profit_percent : ℝ) :
  profit_percent = 150 →
  selling_price = cost_price + (profit_percent / 100) * cost_price →
  (cost_price / selling_price) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l1017_101734


namespace NUMINAMATH_CALUDE_roger_bike_distance_l1017_101726

def morning_distance : ℝ := 2
def evening_multiplier : ℝ := 5

theorem roger_bike_distance : 
  morning_distance + evening_multiplier * morning_distance = 12 := by
  sorry

end NUMINAMATH_CALUDE_roger_bike_distance_l1017_101726


namespace NUMINAMATH_CALUDE_triangle_equilateral_l1017_101709

theorem triangle_equilateral (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_condition : a^2 + b^2 + c^2 - a*b - b*c - a*c = 0) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l1017_101709


namespace NUMINAMATH_CALUDE_tournament_three_cycle_l1017_101787

/-- Represents a tournament with n contestants. -/
structure Tournament (n : ℕ) where
  -- n ≥ 3
  contestants_count : n ≥ 3
  -- Represents the result of matches between contestants
  defeats : Fin n → Fin n → Prop
  -- Each pair of contestants plays exactly one match
  one_match (i j : Fin n) : i ≠ j → (defeats i j ∨ defeats j i) ∧ ¬(defeats i j ∧ defeats j i)
  -- No contestant wins all their matches
  no_perfect_winner (i : Fin n) : ∃ j : Fin n, j ≠ i ∧ defeats j i

/-- 
There exist three contestants A, B, and C such that A defeats B, B defeats C, and C defeats A.
-/
theorem tournament_three_cycle {n : ℕ} (t : Tournament n) :
  ∃ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    t.defeats a b ∧ t.defeats b c ∧ t.defeats c a :=
sorry

end NUMINAMATH_CALUDE_tournament_three_cycle_l1017_101787


namespace NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l1017_101752

/-- The height of a right circular cylinder inscribed in a hemisphere -/
theorem cylinder_height_in_hemisphere (r_cylinder : ℝ) (r_hemisphere : ℝ) 
  (h_cylinder : r_cylinder = 3)
  (h_hemisphere : r_hemisphere = 7) :
  Real.sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_in_hemisphere_l1017_101752


namespace NUMINAMATH_CALUDE_base2_digit_difference_l1017_101797

-- Function to calculate the number of digits in base-2 representation
def base2Digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

-- Theorem statement
theorem base2_digit_difference : base2Digits 1800 - base2Digits 500 = 2 := by
  sorry

end NUMINAMATH_CALUDE_base2_digit_difference_l1017_101797


namespace NUMINAMATH_CALUDE_total_amount_shared_l1017_101775

theorem total_amount_shared (T : ℝ) : 
  (0.4 * T = 0.3 * T + 5) → T = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_shared_l1017_101775


namespace NUMINAMATH_CALUDE_polynomial_intercept_nonzero_coeff_l1017_101746

theorem polynomial_intercept_nonzero_coeff 
  (a b c d e f : ℝ) 
  (Q : ℝ → ℝ) 
  (h_Q : ∀ x, Q x = x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) 
  (h_roots : ∃ p q r s t : ℝ, p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ t ≠ 0 ∧ 
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
    q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
    r ≠ s ∧ r ≠ t ∧ 
    s ≠ t ∧
    Q p = 0 ∧ Q q = 0 ∧ Q r = 0 ∧ Q s = 0 ∧ Q t = 0)
  (h_zero_root : Q 0 = 0) :
  d ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_intercept_nonzero_coeff_l1017_101746


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l1017_101732

theorem factorial_difference_quotient : (Nat.factorial 12 - Nat.factorial 11) / Nat.factorial 10 = 121 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l1017_101732


namespace NUMINAMATH_CALUDE_point_movement_l1017_101755

def Point := ℝ × ℝ

def moveUp (p : Point) (units : ℝ) : Point :=
  (p.1, p.2 + units)

def moveRight (p : Point) (units : ℝ) : Point :=
  (p.1 + units, p.2)

theorem point_movement :
  let original : Point := (-2, 3)
  (moveUp original 2 = (-2, 5)) ∧
  (moveRight original 2 = (0, 3)) := by
  sorry

end NUMINAMATH_CALUDE_point_movement_l1017_101755


namespace NUMINAMATH_CALUDE_sequence_roots_theorem_l1017_101785

theorem sequence_roots_theorem (b c : ℕ → ℝ) : 
  (∀ n : ℕ, n ≥ 1 → b n ≤ c n) → 
  (∀ n : ℕ, n ≥ 1 → (b (n + 1))^2 + (b n) * (b (n + 1)) + (c n) = 0 ∧ 
                     (c (n + 1))^2 + (b n) * (c (n + 1)) + (c n) = 0) →
  (∀ n : ℕ, n ≥ 1 → b n = 0 ∧ c n = 0) :=
by sorry

end NUMINAMATH_CALUDE_sequence_roots_theorem_l1017_101785


namespace NUMINAMATH_CALUDE_refrigerator_cost_proof_l1017_101730

/-- The cost of the refrigerator that satisfies the given conditions --/
def refrigerator_cost : ℝ := 15000

/-- The cost of the mobile phone --/
def mobile_phone_cost : ℝ := 8000

/-- The selling price of the refrigerator --/
def refrigerator_selling_price : ℝ := 0.96 * refrigerator_cost

/-- The selling price of the mobile phone --/
def mobile_phone_selling_price : ℝ := 1.09 * mobile_phone_cost

/-- The total profit --/
def total_profit : ℝ := 120

theorem refrigerator_cost_proof :
  refrigerator_selling_price + mobile_phone_selling_price = 
  refrigerator_cost + mobile_phone_cost + total_profit :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_cost_proof_l1017_101730


namespace NUMINAMATH_CALUDE_fraction_simplification_l1017_101712

theorem fraction_simplification : (1952^2 - 1940^2) / (1959^2 - 1933^2) = 6/13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1017_101712


namespace NUMINAMATH_CALUDE_expression_value_l1017_101716

theorem expression_value (a b : ℝ) (h : a * b > 0) :
  (a / abs a) + (b / abs b) + ((a * b) / abs (a * b)) = 3 ∨
  (a / abs a) + (b / abs b) + ((a * b) / abs (a * b)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1017_101716


namespace NUMINAMATH_CALUDE_correct_regression_conclusions_l1017_101713

/-- Represents a conclusion about regression analysis -/
structure RegressionConclusion where
  statement : String
  is_correct : Bool

/-- Counts the number of correct conclusions in a list -/
def count_correct (conclusions : List RegressionConclusion) : Nat :=
  conclusions.filter (·.is_correct) |>.length

/-- The main theorem about the number of correct regression conclusions -/
theorem correct_regression_conclusions :
  let conclusions : List RegressionConclusion := [
    { statement := "R² larger implies better fit", is_correct := true },
    { statement := "Larger sum of squared residuals implies better fit", is_correct := false },
    { statement := "Larger r implies better fit", is_correct := true },
    { statement := "Residual plot can judge model fit", is_correct := false }
  ]
  count_correct conclusions = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_regression_conclusions_l1017_101713


namespace NUMINAMATH_CALUDE_function_properties_l1017_101704

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x^2 * x + 2 * b - a^3

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x ∈ Set.Ioo (-2 : ℝ) 6, f a b x > 0) →
  (∀ x ∈ Set.Iic (-2 : ℝ) ∪ Set.Ici 6, f a b x < 0) →
  f a b (-2) = 0 →
  f a b 6 = 0 →
  (∃ c d : ℝ, c = -4 ∧ d = -48 ∧ ∀ x, f a b x = c * x^2 + 2 * c * x + d) ∧
  (∀ x ∈ Set.Icc 1 10, f a b x ≤ -20) ∧
  (∀ x ∈ Set.Icc 1 10, f a b x ≥ -192) ∧
  (∃ x ∈ Set.Icc 1 10, f a b x = -20) ∧
  (∃ x ∈ Set.Icc 1 10, f a b x = -192) :=
by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1017_101704


namespace NUMINAMATH_CALUDE_blue_pens_count_l1017_101731

theorem blue_pens_count (red_price blue_price total_cost total_pens : ℕ) 
  (h1 : red_price = 5)
  (h2 : blue_price = 7)
  (h3 : total_cost = 102)
  (h4 : total_pens = 16) :
  ∃ (red_count blue_count : ℕ), 
    red_count + blue_count = total_pens ∧
    red_count * red_price + blue_count * blue_price = total_cost ∧
    blue_count = 11 := by
  sorry

end NUMINAMATH_CALUDE_blue_pens_count_l1017_101731


namespace NUMINAMATH_CALUDE_first_expression_value_l1017_101702

theorem first_expression_value (E a : ℝ) (h1 : (E + (3 * a - 8)) / 2 = 89) (h2 : a = 34) : E = 84 := by
  sorry

end NUMINAMATH_CALUDE_first_expression_value_l1017_101702


namespace NUMINAMATH_CALUDE_smallest_divisor_with_remainder_l1017_101756

theorem smallest_divisor_with_remainder (x y z : ℕ) : 
  x > 0 ∧ y > 0 ∧ z > 0 →
  x % 9 = 2 →
  x % 7 = 4 →
  y % 13 = 12 →
  y - x = 14 →
  y % z = 3 →
  (∀ w : ℕ, w > 0 ∧ w < z ∧ y % w = 3 → False) →
  z = 22 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisor_with_remainder_l1017_101756


namespace NUMINAMATH_CALUDE_coin_collection_dimes_l1017_101779

def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25
def half_dollar : ℕ := 50

theorem coin_collection_dimes :
  ∀ (p n d q h : ℕ),
    p ≥ 1 → n ≥ 1 → d ≥ 1 → q ≥ 1 → h ≥ 1 →
    p + n + d + q + h = 12 →
    p * penny + n * nickel + d * dime + q * quarter + h * half_dollar = 163 →
    d = 5 := by
  sorry

end NUMINAMATH_CALUDE_coin_collection_dimes_l1017_101779


namespace NUMINAMATH_CALUDE_two_by_two_paper_covers_nine_vertices_l1017_101762

/-- Represents a square paper on a grid -/
structure SquarePaper where
  side_length : ℕ
  min_vertices_covered : ℕ

/-- Counts the number of vertices covered by a square paper on a grid -/
def count_vertices_covered (paper : SquarePaper) : ℕ :=
  (paper.side_length + 1) ^ 2

/-- Theorem: A 2x2 square paper covering at least 7 vertices covers exactly 9 vertices -/
theorem two_by_two_paper_covers_nine_vertices (paper : SquarePaper)
  (h1 : paper.side_length = 2)
  (h2 : paper.min_vertices_covered ≥ 7) :
  count_vertices_covered paper = 9 := by
  sorry

end NUMINAMATH_CALUDE_two_by_two_paper_covers_nine_vertices_l1017_101762


namespace NUMINAMATH_CALUDE_expression_equality_l1017_101788

theorem expression_equality : 4⁻¹ - Real.sqrt (1/16) + (3 - Real.sqrt 2)^0 = 1 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l1017_101788


namespace NUMINAMATH_CALUDE_susy_initial_followers_l1017_101761

/-- Represents the number of followers gained by a student over three weeks -/
structure FollowerGain where
  week1 : ℕ
  week2 : ℕ
  week3 : ℕ

/-- Represents a student with their school size and follower information -/
structure Student where
  schoolSize : ℕ
  initialFollowers : ℕ
  followerGain : FollowerGain

def totalFollowersAfterThreeWeeks (student : Student) : ℕ :=
  student.initialFollowers + student.followerGain.week1 + student.followerGain.week2 + student.followerGain.week3

theorem susy_initial_followers
  (susy : Student)
  (sarah : Student)
  (h1 : susy.schoolSize = 800)
  (h2 : sarah.schoolSize = 300)
  (h3 : susy.followerGain.week1 = 40)
  (h4 : susy.followerGain.week2 = susy.followerGain.week1 / 2)
  (h5 : susy.followerGain.week3 = susy.followerGain.week2 / 2)
  (h6 : sarah.initialFollowers = 50)
  (h7 : max (totalFollowersAfterThreeWeeks susy) (totalFollowersAfterThreeWeeks sarah) = 180) :
  susy.initialFollowers = 110 := by
  sorry

end NUMINAMATH_CALUDE_susy_initial_followers_l1017_101761


namespace NUMINAMATH_CALUDE_pascals_cycling_trip_l1017_101736

theorem pascals_cycling_trip (current_speed : ℝ) (speed_reduction : ℝ) (time_difference : ℝ) :
  current_speed = 8 →
  speed_reduction = 4 →
  time_difference = 16 →
  let reduced_speed := current_speed - speed_reduction
  let increased_speed := current_speed * 1.5
  ∃ (distance : ℝ), distance = 96 ∧
    distance / reduced_speed = distance / increased_speed + time_difference :=
by sorry

end NUMINAMATH_CALUDE_pascals_cycling_trip_l1017_101736


namespace NUMINAMATH_CALUDE_grandmas_age_l1017_101780

theorem grandmas_age :
  ∀ x : ℕ, (x : ℝ) - (x : ℝ) / 7 = 84 → x = 98 := by
  sorry

end NUMINAMATH_CALUDE_grandmas_age_l1017_101780


namespace NUMINAMATH_CALUDE_valid_three_digit_numbers_l1017_101723

def is_valid_number (abc : ℕ) : Prop :=
  let a := abc / 100
  let b := (abc / 10) % 10
  let c := abc % 10
  let cab := c * 100 + a * 10 + b
  let bca := b * 100 + c * 10 + a
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  abc ≥ 100 ∧ abc < 1000 ∧
  2 * b = a + c ∧
  (cab * abc : ℚ) = bca * bca

theorem valid_three_digit_numbers :
  ∀ abc : ℕ, is_valid_number abc → abc = 432 ∨ abc = 864 :=
sorry

end NUMINAMATH_CALUDE_valid_three_digit_numbers_l1017_101723


namespace NUMINAMATH_CALUDE_curve_C_properties_l1017_101759

-- Define the curve C
def C (m n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m * p.1^2 + n * p.2^2 = 1}

-- Define what it means for a curve to be an ellipse with foci on the y-axis
def is_ellipse_with_foci_on_y_axis (S : Set (ℝ × ℝ)) : Prop :=
  sorry

-- Define what it means for a curve to be a hyperbola with given asymptotes
def is_hyperbola_with_asymptotes (S : Set (ℝ × ℝ)) (f : ℝ → ℝ) : Prop :=
  sorry

-- Define what it means for a curve to consist of two straight lines
def is_two_straight_lines (S : Set (ℝ × ℝ)) : Prop :=
  sorry

theorem curve_C_properties (m n : ℝ) :
  (m > n ∧ n > 0 → is_ellipse_with_foci_on_y_axis (C m n)) ∧
  (m * n < 0 → is_hyperbola_with_asymptotes (C m n) (λ x => Real.sqrt (-m/n) * x)) ∧
  (m = 0 ∧ n > 0 → is_two_straight_lines (C m n)) :=
  sorry

end NUMINAMATH_CALUDE_curve_C_properties_l1017_101759


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_is_1080_l1017_101784

/-- The polynomial for which we want to calculate the sum of squared coefficients -/
def p (x : ℝ) : ℝ := 6 * (x^3 + 4*x^2 + 2*x + 3)

/-- The sum of the squares of the coefficients of the polynomial p -/
def sum_of_squared_coefficients : ℝ :=
  let coeffs := [6, 24, 12, 18]
  coeffs.map (λ c => c^2) |>.sum

/-- Theorem stating that the sum of the squares of the coefficients of p is 1080 -/
theorem sum_of_squared_coefficients_is_1080 :
  sum_of_squared_coefficients = 1080 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_is_1080_l1017_101784


namespace NUMINAMATH_CALUDE_second_cube_surface_area_l1017_101711

theorem second_cube_surface_area (v1 v2 : ℝ) (h1 : v1 = 16) (h2 : v2 = 4 * v1) :
  6 * (v2 ^ (1/3 : ℝ))^2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_second_cube_surface_area_l1017_101711


namespace NUMINAMATH_CALUDE_min_spheres_to_cover_unit_cylinder_l1017_101753

/-- Represents a cylinder with given height and base radius -/
structure Cylinder where
  height : ℝ
  baseRadius : ℝ

/-- Represents a sphere with given radius -/
structure Sphere where
  radius : ℝ

/-- Function to determine the minimum number of spheres needed to cover a cylinder -/
def minSpheresToCoverCylinder (c : Cylinder) (s : Sphere) : ℕ :=
  sorry

/-- Theorem stating that a cylinder with height 1 and base radius 1 requires at least 3 unit spheres to cover it -/
theorem min_spheres_to_cover_unit_cylinder :
  let c := Cylinder.mk 1 1
  let s := Sphere.mk 1
  minSpheresToCoverCylinder c s = 3 :=
sorry

end NUMINAMATH_CALUDE_min_spheres_to_cover_unit_cylinder_l1017_101753


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l1017_101748

/-- Given two lines l₁ and l₂ in the real plane, prove that if l₁ with equation x + 2y - 1 = 0 
    is parallel to l₂ with equation mx - y = 0, then m = -1/2. -/
theorem parallel_lines_slope (m : ℝ) : 
  (∀ x y : ℝ, x + 2*y - 1 = 0 ↔ m*x - y = 0) → m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l1017_101748


namespace NUMINAMATH_CALUDE_multiplicative_inverse_mod_million_l1017_101778

def C : ℕ := 123456
def D : ℕ := 166666
def M : ℕ := 48

theorem multiplicative_inverse_mod_million :
  (M * (C * D)) % 1000000 = 1 :=
by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_mod_million_l1017_101778


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l1017_101798

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, 
    (Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4)) → 
    t = 37/10 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l1017_101798


namespace NUMINAMATH_CALUDE_solve_for_y_l1017_101700

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1017_101700


namespace NUMINAMATH_CALUDE_base6_divisibility_by_11_l1017_101707

/-- Converts a base-6 number of the form 2dd5₆ to base 10 --/
def base6ToBase10 (d : Nat) : Nat :=
  2 * 6^3 + d * 6^2 + d * 6^1 + 5

/-- Checks if a number is divisible by 11 --/
def isDivisibleBy11 (n : Nat) : Prop :=
  n % 11 = 0

/-- Represents a base-6 digit --/
def isBase6Digit (d : Nat) : Prop :=
  d < 6

theorem base6_divisibility_by_11 :
  ∃ (d : Nat), isBase6Digit d ∧ isDivisibleBy11 (base6ToBase10 d) ↔ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_base6_divisibility_by_11_l1017_101707

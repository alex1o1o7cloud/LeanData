import Mathlib

namespace perimeter_difference_inscribed_quadrilateral_l3717_371749

/-- A quadrilateral with an inscribed circle and two tangents -/
structure InscribedQuadrilateral where
  -- Sides of the quadrilateral
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  -- Ensure sides are positive
  side1_pos : side1 > 0
  side2_pos : side2 > 0
  side3_pos : side3 > 0
  side4_pos : side4 > 0
  -- Tangent points on each side
  tangent1 : ℝ
  tangent2 : ℝ
  tangent3 : ℝ
  tangent4 : ℝ
  -- Ensure tangent points are within side lengths
  tangent1_valid : 0 < tangent1 ∧ tangent1 < side1
  tangent2_valid : 0 < tangent2 ∧ tangent2 < side2
  tangent3_valid : 0 < tangent3 ∧ tangent3 < side3
  tangent4_valid : 0 < tangent4 ∧ tangent4 < side4

/-- Theorem about the difference in perimeters of cut-off triangles -/
theorem perimeter_difference_inscribed_quadrilateral 
  (q : InscribedQuadrilateral) 
  (h1 : q.side1 = 3) 
  (h2 : q.side2 = 5) 
  (h3 : q.side3 = 9) 
  (h4 : q.side4 = 7) :
  (2 * (q.tangent3 - q.tangent1) = 4 ∨ 2 * (q.tangent3 - q.tangent1) = 8) ∧
  (2 * (q.tangent4 - q.tangent2) = 4 ∨ 2 * (q.tangent4 - q.tangent2) = 8) :=
sorry

end perimeter_difference_inscribed_quadrilateral_l3717_371749


namespace power_sum_of_i_l3717_371729

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^203 = -2*i := by
  sorry

end power_sum_of_i_l3717_371729


namespace simplified_expression_terms_l3717_371778

/-- The number of terms in the simplified form of (x+y+z)^2010 + (x-y-z)^2010 -/
def num_terms : ℕ := 1012036

/-- The exponent used in the expression -/
def exponent : ℕ := 2010

theorem simplified_expression_terms :
  num_terms = (exponent / 2 + 1)^2 := by sorry

end simplified_expression_terms_l3717_371778


namespace remainder_mod_11_l3717_371763

theorem remainder_mod_11 : (7 * 10^20 + 2^20) % 11 = 8 := by
  sorry

end remainder_mod_11_l3717_371763


namespace torus_grid_piece_placement_impossible_l3717_371727

theorem torus_grid_piece_placement_impossible :
  ∀ (a b c : ℕ) (x y z : ℕ),
    a + b + c = 50 →
    2 * a ≤ x ∧ x ≤ 2 * b →
    2 * b ≤ y ∧ y ≤ 2 * c →
    2 * c ≤ z ∧ z ≤ 2 * a →
    False :=
by sorry

end torus_grid_piece_placement_impossible_l3717_371727


namespace f_min_at_five_thirds_l3717_371789

/-- The function f(x) = 3x³ - 2x² - 18x + 9 -/
def f (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 - 18 * x + 9

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 9 * x^2 - 4 * x - 18

/-- The second derivative of f(x) -/
def f'' (x : ℝ) : ℝ := 18 * x - 4

theorem f_min_at_five_thirds :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 5/3 ∧ |x - 5/3| < ε → f x > f (5/3) :=
sorry

end f_min_at_five_thirds_l3717_371789


namespace hyperbola_point_distance_to_x_axis_l3717_371723

/-- The distance from a point on a hyperbola to the x-axis, given specific conditions -/
theorem hyperbola_point_distance_to_x_axis 
  (x y : ℝ) -- Coordinates of point P
  (h1 : x^2 / 9 - y^2 / 16 = 1) -- Equation of the hyperbola
  (h2 : (y - 0) * (y - 0) = -(x + 5) * (x - 5)) -- Condition for PF₁ ⊥ PF₂
  : |y| = 16 / 5 := by
  sorry


end hyperbola_point_distance_to_x_axis_l3717_371723


namespace quadratic_expression_value_l3717_371706

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 6) 
  (eq2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 98.08 := by
sorry

end quadratic_expression_value_l3717_371706


namespace decreasing_function_inequality_l3717_371748

theorem decreasing_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_decreasing : ∀ x y, x ≤ y → f x ≥ f y) 
  (h_sum : a + b ≤ 0) : 
  f a + f b ≥ f (-a) + f (-b) := by
sorry

end decreasing_function_inequality_l3717_371748


namespace divisible_by_2_4_5_under_300_l3717_371782

theorem divisible_by_2_4_5_under_300 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0) (Finset.range 300)).card = 15 := by
  sorry

end divisible_by_2_4_5_under_300_l3717_371782


namespace pages_in_book_l3717_371799

/-- 
Given a person who reads a fixed number of pages per day and finishes a book in a certain number of days,
this theorem proves the total number of pages in the book.
-/
theorem pages_in_book (pages_per_day : ℕ) (days_to_finish : ℕ) : 
  pages_per_day = 8 → days_to_finish = 12 → pages_per_day * days_to_finish = 96 := by
  sorry

end pages_in_book_l3717_371799


namespace smallest_valid_seating_l3717_371735

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone. -/
def valid_seating (table : CircularTable) : Prop :=
  table.seated_people > 0 ∧ 
  table.seated_people ≤ table.total_chairs ∧
  ∀ k : ℕ, k < table.total_chairs → ∃ i j : ℕ, 
    i < table.seated_people ∧ 
    j < table.seated_people ∧ 
    (k - i) % table.total_chairs ≤ 2 ∧ 
    (j - k) % table.total_chairs ≤ 2

/-- The theorem stating the smallest number of people that can be seated. -/
theorem smallest_valid_seating :
  ∀ n : ℕ, n < 20 → ¬(valid_seating ⟨60, n⟩) ∧ 
  valid_seating ⟨60, 20⟩ :=
sorry

end smallest_valid_seating_l3717_371735


namespace unit_digit_15_power_100_l3717_371736

theorem unit_digit_15_power_100 : ∃ n : ℕ, 15^100 = 10 * n + 5 := by
  sorry

end unit_digit_15_power_100_l3717_371736


namespace cosine_inequality_existence_l3717_371743

theorem cosine_inequality_existence (a b c : ℝ) :
  ∃ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) + c * Real.cos (9 * x) ≥ 1/2 * (|a| + |b| + |c|) := by
  sorry

end cosine_inequality_existence_l3717_371743


namespace dots_not_visible_l3717_371734

/-- The number of dice -/
def num_dice : ℕ := 5

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The list of visible numbers on the dice -/
def visible_numbers : List ℕ := [1, 2, 2, 3, 3, 3, 4, 4, 5, 6]

/-- The theorem stating the number of dots not visible -/
theorem dots_not_visible :
  num_dice * die_sum - visible_numbers.sum = 72 := by sorry

end dots_not_visible_l3717_371734


namespace sequence_term_formula_l3717_371750

/-- Given a sequence a_n with sum S_n, prove the general term formula -/
theorem sequence_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n^2 + 1 → a n = 2*n - 1) ∧
  (∀ n, S n = 2*n^2 → a n = 4*n - 2) := by
  sorry

end sequence_term_formula_l3717_371750


namespace rationalize_sqrt_sum_l3717_371768

def rationalize_denominator (x y z : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ :=
  sorry

theorem rationalize_sqrt_sum : 
  let (A, B, C, D, E, F) := rationalize_denominator (Real.sqrt 3) (Real.sqrt 5) (Real.sqrt 11)
  A + B + C + D + E + F = 97 := by sorry

end rationalize_sqrt_sum_l3717_371768


namespace exactly_three_solutions_l3717_371797

-- Define S(n) as the sum of digits of n
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Define the main equation
def satisfiesEquation (n : ℕ) : Prop :=
  n + sumOfDigits n + sumOfDigits (sumOfDigits n) = 2023

-- Theorem statement
theorem exactly_three_solutions :
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ n, n ∈ s ↔ satisfiesEquation n :=
sorry

end exactly_three_solutions_l3717_371797


namespace jacket_price_correct_l3717_371769

/-- The original price of the jacket -/
def original_price : ℝ := 250

/-- The regular discount percentage -/
def regular_discount : ℝ := 0.4

/-- The special sale discount percentage -/
def special_discount : ℝ := 0.1

/-- The final price after both discounts -/
def final_price : ℝ := original_price * (1 - regular_discount) * (1 - special_discount)

theorem jacket_price_correct : final_price = 135 := by
  sorry

end jacket_price_correct_l3717_371769


namespace pizza_and_toppings_count_l3717_371721

/-- Calculates the total number of pieces of pizza and toppings carried by fourth-graders --/
theorem pizza_and_toppings_count : 
  let pieces_per_pizza : ℕ := 6
  let num_fourth_graders : ℕ := 10
  let pizzas_per_child : ℕ := 20
  let pepperoni_per_pizza : ℕ := 5
  let mushrooms_per_pizza : ℕ := 3
  let olives_per_pizza : ℕ := 8

  let total_pizzas : ℕ := num_fourth_graders * pizzas_per_child
  let total_pieces : ℕ := total_pizzas * pieces_per_pizza
  let total_pepperoni : ℕ := total_pizzas * pepperoni_per_pizza
  let total_mushrooms : ℕ := total_pizzas * mushrooms_per_pizza
  let total_olives : ℕ := total_pizzas * olives_per_pizza
  let total_toppings : ℕ := total_pepperoni + total_mushrooms + total_olives

  total_pieces + total_toppings = 4400 := by
  sorry

end pizza_and_toppings_count_l3717_371721


namespace complex_real_condition_l3717_371709

theorem complex_real_condition (m : ℝ) :
  let z : ℂ := m^2 * (1 + Complex.I) - m * (m + Complex.I)
  (z.im = 0) ↔ (m = 0 ∨ m = 1) := by
  sorry

end complex_real_condition_l3717_371709


namespace cos_squared_sixty_degrees_l3717_371758

theorem cos_squared_sixty_degrees :
  let cos_sixty : ℝ := 1 / 2
  (cos_sixty ^ 2 : ℝ) = 1 / 4 := by
  sorry

end cos_squared_sixty_degrees_l3717_371758


namespace line_passes_through_fixed_point_min_dot_product_midpoint_locus_l3717_371744

-- Define the line l: mx - y - m + 2 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  m * x - y - m + 2 = 0

-- Define the circle C: x^2 + y^2 = 9
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

-- Define the intersection points A and B
def intersection_points (m : ℝ) (A B : ℝ × ℝ) : Prop :=
  line_l m A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l m B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem 1: Line l passes through (1, 2) for all m
theorem line_passes_through_fixed_point (m : ℝ) :
  line_l m 1 2 := by sorry

-- Theorem 2: Minimum value of AC · AB is 8
theorem min_dot_product (m : ℝ) (A B : ℝ × ℝ) :
  intersection_points m A B →
  ∃ (C : ℝ × ℝ), circle_C C.1 C.2 ∧
  (∀ (D : ℝ × ℝ), circle_C D.1 D.2 →
    (A.1 - C.1) * (B.1 - A.1) + (A.2 - C.2) * (B.2 - A.2) ≥ 8) := by sorry

-- Theorem 3: Locus of midpoint of AB is a circle
theorem midpoint_locus (m : ℝ) (A B : ℝ × ℝ) :
  intersection_points m A B →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ((A.1 + B.1) / 2 - center.1)^2 + ((A.2 + B.2) / 2 - center.2)^2 = radius^2 := by sorry

end line_passes_through_fixed_point_min_dot_product_midpoint_locus_l3717_371744


namespace alan_needs_17_votes_to_win_l3717_371798

/-- Represents the number of votes for each candidate -/
structure VoteCount where
  sally : Nat
  katie : Nat
  alan : Nat

/-- The problem setup -/
def totalVoters : Nat := 130

def currentVotes : VoteCount := {
  sally := 24,
  katie := 29,
  alan := 37
}

/-- Alan needs at least this many more votes to be certain of winning -/
def minVotesNeeded : Nat := 17

theorem alan_needs_17_votes_to_win : 
  ∀ (finalVotes : VoteCount),
  finalVotes.sally ≥ currentVotes.sally ∧ 
  finalVotes.katie ≥ currentVotes.katie ∧
  finalVotes.alan ≥ currentVotes.alan ∧
  finalVotes.sally + finalVotes.katie + finalVotes.alan = totalVoters →
  (finalVotes.alan = currentVotes.alan + minVotesNeeded - 1 → 
   ¬(finalVotes.alan > finalVotes.sally ∧ finalVotes.alan > finalVotes.katie)) ∧
  (finalVotes.alan ≥ currentVotes.alan + minVotesNeeded → 
   finalVotes.alan > finalVotes.sally ∧ finalVotes.alan > finalVotes.katie) :=
by sorry

#check alan_needs_17_votes_to_win

end alan_needs_17_votes_to_win_l3717_371798


namespace f_range_and_tan_A_l3717_371781

noncomputable section

def f (x : ℝ) := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_range_and_tan_A :
  (∀ x ∈ Set.Icc (-π/6) (π/3), f x ∈ Set.Icc 0 3) ∧
  (∀ A B C : ℝ, 
    f C = 2 → 
    2 * Real.sin B = Real.cos (A - C) - Real.cos (A + C) → 
    Real.tan A = (3 + Real.sqrt 3) / 2) := by
  sorry

end f_range_and_tan_A_l3717_371781


namespace cuboid_surface_area_example_l3717_371766

/-- The surface area of a cuboid -/
def cuboid_surface_area (width length height : ℝ) : ℝ :=
  2 * (width * length + width * height + length * height)

/-- Theorem: The surface area of a cuboid with width 8 cm, length 5 cm, and height 10 cm is 340 cm² -/
theorem cuboid_surface_area_example : cuboid_surface_area 8 5 10 = 340 := by
  sorry

end cuboid_surface_area_example_l3717_371766


namespace percent_relation_l3717_371745

theorem percent_relation (x y : ℝ) (h : 0.7 * (x - y) = 0.3 * (x + y)) : y / x = 0.4 := by
  sorry

end percent_relation_l3717_371745


namespace max_value_fraction_l3717_371711

theorem max_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 5) :
  (∀ a b : ℝ, -6 ≤ a ∧ a ≤ -3 → 1 ≤ b ∧ b ≤ 5 → (a + b + 1) / (a + 1) ≤ (x + y + 1) / (x + 1)) →
  (x + y + 1) / (x + 1) = 0 :=
by sorry

end max_value_fraction_l3717_371711


namespace naclo4_formation_l3717_371714

-- Define the chemical reaction
structure Reaction where
  naoh : ℝ
  hclo4 : ℝ
  naclo4 : ℝ
  h2o : ℝ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.naoh = r.hclo4 ∧ r.naoh = r.naclo4 ∧ r.naoh = r.h2o

-- Define the initial conditions
def initial_conditions (initial_naoh initial_hclo4 : ℝ) (r : Reaction) : Prop :=
  initial_naoh = 3 ∧ initial_hclo4 = 3 ∧ r.naoh ≤ initial_naoh ∧ r.hclo4 ≤ initial_hclo4

-- Theorem statement
theorem naclo4_formation 
  (initial_naoh initial_hclo4 : ℝ) 
  (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : initial_conditions initial_naoh initial_hclo4 r) :
  r.naclo4 = min initial_naoh initial_hclo4 :=
sorry

end naclo4_formation_l3717_371714


namespace smallest_in_set_l3717_371785

def S : Set ℚ := {1/2, 2/3, 1/4, 5/6, 7/12}

theorem smallest_in_set : ∀ x ∈ S, 1/4 ≤ x := by sorry

end smallest_in_set_l3717_371785


namespace compound_interest_rate_exists_unique_l3717_371760

theorem compound_interest_rate_exists_unique (P : ℝ) (h1 : P > 0) :
  ∃! r : ℝ, r > 0 ∧ r < 1 ∧ 
    800 = P * (1 + r)^3 ∧
    820 = P * (1 + r)^4 :=
sorry

end compound_interest_rate_exists_unique_l3717_371760


namespace sufficient_not_necessary_l3717_371770

/-- The solution set of (ax-1)(x-1) > 0 is (1/a, 1) -/
def SolutionSet (a : ℝ) : Prop :=
  ∀ x, (a * x - 1) * (x - 1) > 0 ↔ 1/a < x ∧ x < 1

theorem sufficient_not_necessary :
  (∀ a : ℝ, SolutionSet a → a < 1/2) ∧
  (∃ a : ℝ, a < 1/2 ∧ ¬(SolutionSet a)) := by
  sorry

end sufficient_not_necessary_l3717_371770


namespace skateboard_price_l3717_371791

theorem skateboard_price (upfront_payment : ℝ) (upfront_percentage : ℝ) 
  (h1 : upfront_payment = 60)
  (h2 : upfront_percentage = 20) : 
  let full_price := upfront_payment / (upfront_percentage / 100)
  full_price = 300 := by
  sorry

end skateboard_price_l3717_371791


namespace sum_due_calculation_l3717_371776

/-- Given a banker's discount and true discount, calculate the sum due -/
def sum_due (bankers_discount true_discount : ℚ) : ℚ :=
  (true_discount^2) / (bankers_discount - true_discount)

/-- Theorem: The sum due is 2400 when banker's discount is 576 and true discount is 480 -/
theorem sum_due_calculation : sum_due 576 480 = 2400 := by
  sorry

end sum_due_calculation_l3717_371776


namespace lcm_of_23_46_827_l3717_371755

theorem lcm_of_23_46_827 : Nat.lcm 23 (Nat.lcm 46 827) = 38042 := by
  sorry

end lcm_of_23_46_827_l3717_371755


namespace artist_paintings_l3717_371780

def june_paintings : ℕ := 2

def july_paintings : ℕ := 2 * june_paintings

def august_paintings : ℕ := 3 * july_paintings

def total_paintings : ℕ := june_paintings + july_paintings + august_paintings

theorem artist_paintings : total_paintings = 18 := by
  sorry

end artist_paintings_l3717_371780


namespace student_group_combinations_l3717_371765

theorem student_group_combinations (n : ℕ) (h : n = 8) : 
  Nat.choose n 4 + Nat.choose n 5 = 126 := by
  sorry

#check student_group_combinations

end student_group_combinations_l3717_371765


namespace lcm_of_45_60_120_150_l3717_371764

theorem lcm_of_45_60_120_150 : Nat.lcm 45 (Nat.lcm 60 (Nat.lcm 120 150)) = 1800 := by
  sorry

end lcm_of_45_60_120_150_l3717_371764


namespace apple_pie_count_l3717_371795

/-- The number of halves in an apple pie -/
def halves_per_pie : ℕ := 2

/-- The number of bite-size samples in half an apple pie -/
def samples_per_half : ℕ := 5

/-- The number of people who can taste Sedrach's apple pies -/
def people_tasting : ℕ := 130

/-- The number of apple pies Sedrach has -/
def sedrachs_pies : ℕ := 13

theorem apple_pie_count :
  sedrachs_pies * halves_per_pie * samples_per_half = people_tasting := by
  sorry

end apple_pie_count_l3717_371795


namespace sampling_probabilities_l3717_371738

/-- Simple random sampling without replacement -/
def SimpleRandomSampling (population_size : ℕ) (sample_size : ℕ) : Prop :=
  sample_size ≤ population_size

/-- Probability of selecting a specific individual on the first draw -/
def ProbFirstDraw (population_size : ℕ) : ℚ :=
  1 / population_size

/-- Probability of selecting a specific individual on the second draw -/
def ProbSecondDraw (population_size : ℕ) : ℚ :=
  (population_size - 1) / population_size * (1 / (population_size - 1))

/-- Theorem stating the probabilities for the given scenario -/
theorem sampling_probabilities 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 10) 
  (h2 : sample_size = 3) 
  (h3 : SimpleRandomSampling population_size sample_size) :
  ProbFirstDraw population_size = 1/10 ∧ 
  ProbSecondDraw population_size = 1/10 := by
  sorry

end sampling_probabilities_l3717_371738


namespace mess_expenditure_theorem_l3717_371747

/-- Calculates the original expenditure of a mess given the initial and new conditions --/
def original_expenditure (initial_students : ℕ) (new_students : ℕ) (expense_increase : ℕ) (avg_decrease : ℕ) : ℕ :=
  let total_students : ℕ := initial_students + new_students
  let x : ℕ := (expense_increase + total_students * avg_decrease) / (total_students - initial_students)
  initial_students * x

/-- Theorem stating the original expenditure of the mess --/
theorem mess_expenditure_theorem :
  original_expenditure 35 7 84 1 = 630 := by
  sorry

end mess_expenditure_theorem_l3717_371747


namespace largest_value_l3717_371790

theorem largest_value (a b c d e : ℕ) : 
  a = 3 + 1 + 2 + 4 →
  b = 3 * 1 + 2 + 4 →
  c = 3 + 1 * 2 + 4 →
  d = 3 + 1 + 2 * 4 →
  e = 3 * 1 * 2 * 4 →
  e ≥ a ∧ e ≥ b ∧ e ≥ c ∧ e ≥ d :=
by sorry

end largest_value_l3717_371790


namespace comparison_of_powers_l3717_371704

theorem comparison_of_powers : 0.2^3 < 2^0.3 := by
  sorry

end comparison_of_powers_l3717_371704


namespace calculation_proof_l3717_371720

theorem calculation_proof : 2 * Real.sin (30 * π / 180) - (8 : ℝ) ^ (1/3) + (2 - Real.pi) ^ 0 + (-1) ^ 2023 = -1 := by
  sorry

end calculation_proof_l3717_371720


namespace stock_loss_percentage_l3717_371774

theorem stock_loss_percentage 
  (total_stock : ℝ) 
  (profit_percentage : ℝ) 
  (profit_stock_ratio : ℝ) 
  (overall_loss : ℝ) :
  total_stock = 22500 →
  profit_percentage = 10 →
  profit_stock_ratio = 20 →
  overall_loss = 450 →
  ∃ (loss_percentage : ℝ),
    loss_percentage = 5 ∧
    overall_loss = (loss_percentage / 100 * (100 - profit_stock_ratio) / 100 * total_stock) - 
                   (profit_percentage / 100 * profit_stock_ratio / 100 * total_stock) := by
  sorry

end stock_loss_percentage_l3717_371774


namespace decaf_coffee_percentage_l3717_371787

/-- Calculates the percentage of decaffeinated coffee in the total stock -/
theorem decaf_coffee_percentage 
  (initial_stock : ℝ) 
  (initial_decaf_percent : ℝ)
  (additional_stock : ℝ)
  (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 50) :
  let total_stock := initial_stock + additional_stock
  let total_decaf := (initial_stock * initial_decaf_percent / 100) + 
                     (additional_stock * additional_decaf_percent / 100)
  let final_decaf_percent := (total_decaf / total_stock) * 100
  final_decaf_percent = 26 := by
sorry

end decaf_coffee_percentage_l3717_371787


namespace chemical_solution_concentration_l3717_371702

theorem chemical_solution_concentration 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (drained_volume : ℝ) 
  (added_concentration : ℝ) 
  (final_volume : ℝ)
  (h1 : initial_volume = 50)
  (h2 : initial_concentration = 0.60)
  (h3 : drained_volume = 35)
  (h4 : added_concentration = 0.40)
  (h5 : final_volume = initial_volume) :
  let remaining_volume := initial_volume - drained_volume
  let initial_chemical := initial_volume * initial_concentration
  let drained_chemical := drained_volume * initial_concentration
  let remaining_chemical := initial_chemical - drained_chemical
  let added_chemical := drained_volume * added_concentration
  let final_chemical := remaining_chemical + added_chemical
  let final_concentration := final_chemical / final_volume
  final_concentration = 0.46 := by
sorry

end chemical_solution_concentration_l3717_371702


namespace largest_of_three_consecutive_evens_l3717_371772

/-- Given three consecutive even integers where the sum of these integers
    is 18 greater than the smallest integer, prove that the largest integer is 10. -/
theorem largest_of_three_consecutive_evens (x : ℤ) : 
  (x % 2 = 0) →  -- x is even
  (x + (x + 2) + (x + 4) = x + 18) →  -- sum condition
  (x + 4 = 10) :=  -- largest number is 10
by sorry

end largest_of_three_consecutive_evens_l3717_371772


namespace negative_reciprocal_of_opposite_of_neg_abs_three_l3717_371716

-- Definition of opposite numbers
def opposite (a b : ℝ) : Prop := a = -b

-- Definition of reciprocal
def reciprocal (a b : ℝ) : Prop := a * b = 1

-- Theorem to prove
theorem negative_reciprocal_of_opposite_of_neg_abs_three :
  ∃ x : ℝ, opposite (-|(-3)|) x ∧ reciprocal (-1/3) (-x) := by sorry

end negative_reciprocal_of_opposite_of_neg_abs_three_l3717_371716


namespace d_72_eq_22_l3717_371713

/-- D(n) is the number of ways to write n as an ordered product of integers greater than 1 -/
def D (n : ℕ) : ℕ := sorry

/-- The main theorem: D(72) = 22 -/
theorem d_72_eq_22 : D 72 = 22 := by sorry

end d_72_eq_22_l3717_371713


namespace multiplication_associativity_l3717_371762

theorem multiplication_associativity (x y z : ℝ) : (x * y) * z = x * (y * z) := by
  sorry

end multiplication_associativity_l3717_371762


namespace fraction_inequality_l3717_371730

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : a / b < a / c := by
  sorry

end fraction_inequality_l3717_371730


namespace french_students_count_l3717_371775

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 78 → german = 22 → both = 9 → neither = 24 → 
  ∃ french : ℕ, french = 41 ∧ french + german - both + neither = total :=
sorry

end french_students_count_l3717_371775


namespace number_count_proof_l3717_371792

theorem number_count_proof (avg_all : ℝ) (avg_pair1 avg_pair2 avg_pair3 : ℝ) 
  (h1 : avg_all = 5.40)
  (h2 : avg_pair1 = 5.2)
  (h3 : avg_pair2 = 5.80)
  (h4 : avg_pair3 = 5.200000000000003) :
  (2 * avg_pair1 + 2 * avg_pair2 + 2 * avg_pair3) / avg_all = 6 := by
  sorry

end number_count_proof_l3717_371792


namespace exponential_function_coefficient_l3717_371779

def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ (b c : ℝ), b > 0 ∧ b ≠ 1 ∧ (∀ x, f x = c * b^x)

theorem exponential_function_coefficient (a : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  is_exponential_function (λ x => (a^2 - 3*a + 3) * a^x) →
  a = 2 := by sorry

end exponential_function_coefficient_l3717_371779


namespace survival_rate_all_survived_survival_rate_97_trees_l3717_371737

/-- The survival rate of trees given the number of surviving trees and the total number of planted trees. -/
def survival_rate (surviving : ℕ) (total : ℕ) : ℚ :=
  (surviving : ℚ) / (total : ℚ)

/-- Theorem stating that the survival rate is 100% when all planted trees survive. -/
theorem survival_rate_all_survived (n : ℕ) (h : n > 0) :
  survival_rate n n = 1 := by
  sorry

/-- The specific case for 97 trees. -/
theorem survival_rate_97_trees :
  survival_rate 97 97 = 1 := by
  sorry

end survival_rate_all_survived_survival_rate_97_trees_l3717_371737


namespace mingi_initial_tomatoes_l3717_371742

/-- The number of cherry tomatoes Mingi gave to each classmate -/
def tomatoes_per_classmate : ℕ := 15

/-- The number of classmates Mingi gave cherry tomatoes to -/
def number_of_classmates : ℕ := 20

/-- The number of cherry tomatoes Mingi had left after giving them away -/
def remaining_tomatoes : ℕ := 6

/-- The total number of cherry tomatoes Mingi had initially -/
def initial_tomatoes : ℕ := tomatoes_per_classmate * number_of_classmates + remaining_tomatoes

theorem mingi_initial_tomatoes : initial_tomatoes = 306 := by
  sorry

end mingi_initial_tomatoes_l3717_371742


namespace tan_sum_pi_fractions_l3717_371794

theorem tan_sum_pi_fractions : 
  Real.tan (π / 12) + Real.tan (7 * π / 12) = -(4 * (3 - Real.sqrt 3)) / 5 := by
  sorry

end tan_sum_pi_fractions_l3717_371794


namespace stamp_problem_l3717_371718

/-- Represents the number of ways to make a certain amount with given coin denominations -/
def numWays (amount : ℕ) (coins : List ℕ) : ℕ := sorry

/-- Represents the minimum number of coins needed to make a certain amount with given coin denominations -/
def minCoins (amount : ℕ) (coins : List ℕ) : ℕ := sorry

theorem stamp_problem :
  minCoins 74 [5, 7] = 12 := by sorry

end stamp_problem_l3717_371718


namespace angle_C_measure_l3717_371725

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  (Real.cos (ω * x))^2 + Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) - 1/2

theorem angle_C_measure (ω : ℝ) (a b : ℝ) (A : ℝ) :
  ω > 0 →
  (∀ x, f ω (x + π) = f ω x) →
  (∀ x, ¬∀ y, y ≠ x → y > 0 → f ω (x + y) = f ω x) →
  a = 1 →
  b = Real.sqrt 2 →
  f ω (A / 2) = Real.sqrt 3 / 2 →
  a < b →
  ∃ C, (C = 7 * π / 12 ∨ C = π / 12) ∧
       C + A + Real.arcsin (b * Real.sin A / a) = π :=
by sorry

end angle_C_measure_l3717_371725


namespace sara_payment_l3717_371733

/-- The amount Sara gave to the cashier --/
def amount_given (balloon_cost tablecloth_cost streamer_cost banner_cost confetti_cost change : ℚ) : ℚ :=
  balloon_cost + tablecloth_cost + streamer_cost + banner_cost + confetti_cost + change

/-- Theorem stating the amount Sara gave to the cashier --/
theorem sara_payment :
  amount_given 3.5 18.25 9.1 14.65 7.4 6.38 = 59.28 := by
  sorry

end sara_payment_l3717_371733


namespace travis_payment_l3717_371757

/-- Calculates the payment for Travis given the specified conditions --/
def calculate_payment (total_bowls : ℕ) (base_fee : ℚ) (safe_delivery_fee : ℚ) 
  (broken_glass_charge : ℚ) (broken_ceramic_charge : ℚ) (lost_glass_charge : ℚ) 
  (lost_ceramic_charge : ℚ) (glass_weight : ℚ) (ceramic_weight : ℚ) 
  (weight_fee : ℚ) (lost_glass : ℕ) (lost_ceramic : ℕ) (broken_glass : ℕ) 
  (broken_ceramic : ℕ) : ℚ :=
  let safe_bowls := total_bowls - (lost_glass + lost_ceramic + broken_glass + broken_ceramic)
  let safe_payment := safe_delivery_fee * safe_bowls
  let broken_lost_charges := broken_glass_charge * broken_glass + 
                             broken_ceramic_charge * broken_ceramic +
                             lost_glass_charge * lost_glass + 
                             lost_ceramic_charge * lost_ceramic
  let total_weight := glass_weight * (total_bowls - lost_ceramic - broken_ceramic) + 
                      ceramic_weight * (total_bowls - lost_glass - broken_glass)
  let weight_charge := weight_fee * total_weight
  base_fee + safe_payment - broken_lost_charges + weight_charge

/-- The payment for Travis should be $2894.25 given the specified conditions --/
theorem travis_payment : 
  calculate_payment 638 100 3 5 4 6 3 2 (3/2) (1/2) 9 3 10 5 = 2894.25 := by
  sorry

end travis_payment_l3717_371757


namespace perfect_square_polynomial_l3717_371788

theorem perfect_square_polynomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 8*x + k = (x + a)^2) → k = 16 := by
  sorry

end perfect_square_polynomial_l3717_371788


namespace probability_non_defective_second_draw_l3717_371793

def total_products : ℕ := 100
def defective_products : ℕ := 3

theorem probability_non_defective_second_draw :
  let remaining_total := total_products - 1
  let remaining_defective := defective_products - 1
  let remaining_non_defective := remaining_total - remaining_defective
  (remaining_non_defective : ℚ) / remaining_total = 97 / 99 :=
sorry

end probability_non_defective_second_draw_l3717_371793


namespace complex_magnitude_problem_l3717_371753

theorem complex_magnitude_problem (z : ℂ) (h : (4 + 3*Complex.I) * (z - 3*Complex.I) = 25) : 
  Complex.abs z = 4 := by
sorry

end complex_magnitude_problem_l3717_371753


namespace no_ripe_oranges_harvested_l3717_371786

/-- Given the daily harvest of unripe oranges and the total after 6 days,
    prove that no ripe oranges are harvested daily. -/
theorem no_ripe_oranges_harvested
  (unripe_daily : ℕ)
  (total_unripe_6days : ℕ)
  (h1 : unripe_daily = 65)
  (h2 : total_unripe_6days = 390)
  (h3 : unripe_daily * 6 = total_unripe_6days) :
  0 = (total_unripe_6days - unripe_daily * 6) / 6 :=
by sorry

end no_ripe_oranges_harvested_l3717_371786


namespace large_triangle_altitude_proof_l3717_371777

/-- The altitude of a triangle with area 1600 square feet, composed of two identical smaller triangles each with base 40 feet -/
def largeTriangleAltitude : ℝ := 40

theorem large_triangle_altitude_proof (largeArea smallBase : ℝ) 
  (h1 : largeArea = 1600)
  (h2 : smallBase = 40)
  (h3 : largeArea = 2 * (1/2 * smallBase * largeTriangleAltitude)) :
  largeTriangleAltitude = 40 := by
  sorry

#check large_triangle_altitude_proof

end large_triangle_altitude_proof_l3717_371777


namespace fraction_inequality_l3717_371712

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  a / b > (a + c) / (b + c) := by
  sorry

end fraction_inequality_l3717_371712


namespace circle_chords_area_theorem_l3717_371761

/-- Given a circle with radius 48, two chords of length 84 intersecting at a point 24 units from
    the center, the area of the region consisting of a smaller sector and one triangle formed by
    the chords and intersection point can be expressed as m*π - n*√d, where m, n, d are positive
    integers, d is not divisible by any prime square, and m + n + d = 1302. -/
theorem circle_chords_area_theorem (r : ℝ) (chord_length : ℝ) (intersection_distance : ℝ)
    (h1 : r = 48)
    (h2 : chord_length = 84)
    (h3 : intersection_distance = 24) :
    ∃ (m n d : ℕ), 
      (m > 0 ∧ n > 0 ∧ d > 0) ∧ 
      (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ d)) ∧
      (m + n + d = 1302) ∧
      (∃ (area : ℝ), area = m * Real.pi - n * Real.sqrt d) :=
by sorry

end circle_chords_area_theorem_l3717_371761


namespace elizabeth_pencil_cost_l3717_371739

/-- The cost of a pencil given Elizabeth's shopping constraints -/
def pencil_cost (total_money : ℚ) (pen_cost : ℚ) (num_pens : ℕ) (num_pencils : ℕ) : ℚ :=
  (total_money - pen_cost * num_pens) / num_pencils

theorem elizabeth_pencil_cost :
  pencil_cost 20 2 6 5 = 1.60 := by
  sorry

end elizabeth_pencil_cost_l3717_371739


namespace cubic_sum_minus_product_l3717_371726

theorem cubic_sum_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 14) 
  (h2 : x*y + x*z + y*z = 32) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 1400 := by
  sorry

end cubic_sum_minus_product_l3717_371726


namespace remaining_pepper_l3717_371732

/-- Calculates the remaining amount of pepper after usage and addition -/
theorem remaining_pepper (initial : ℝ) (used : ℝ) (added : ℝ) (remaining : ℝ) :
  initial = 0.25 →
  used = 0.16 →
  remaining = initial - used + added →
  remaining = 0.09 + added :=
by sorry

end remaining_pepper_l3717_371732


namespace circumcircle_perpendicular_to_tangent_circles_l3717_371773

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define the given conditions
def are_externally_tangent (c1 c2 : Circle) (p : Point) : Prop :=
  -- The point of tangency is on the line connecting the centers
  -- and the distance between centers is the sum of radii
  sorry

def circumcircle (p1 p2 p3 : Point) : Circle :=
  sorry

def is_perpendicular (c1 c2 : Circle) : Prop :=
  sorry

-- The main theorem
theorem circumcircle_perpendicular_to_tangent_circles 
  (c1 c2 c3 : Circle) (A B C : Point) : 
  are_externally_tangent c1 c2 A ∧ 
  are_externally_tangent c2 c3 B ∧ 
  are_externally_tangent c3 c1 C → 
  is_perpendicular (circumcircle A B C) c1 ∧ 
  is_perpendicular (circumcircle A B C) c2 ∧ 
  is_perpendicular (circumcircle A B C) c3 :=
by
  sorry

end circumcircle_perpendicular_to_tangent_circles_l3717_371773


namespace plain_cookie_price_l3717_371717

/-- The price of each box of plain cookies, given the total number of boxes sold,
    the combined value of all boxes, the number of plain cookie boxes sold,
    and the price of each box of chocolate chip cookies. -/
theorem plain_cookie_price
  (total_boxes : ℝ)
  (combined_value : ℝ)
  (plain_boxes : ℝ)
  (choc_chip_price : ℝ)
  (h1 : total_boxes = 1585)
  (h2 : combined_value = 1586.75)
  (h3 : plain_boxes = 793.375)
  (h4 : choc_chip_price = 1.25) :
  (combined_value - (total_boxes - plain_boxes) * choc_chip_price) / plain_boxes = 0.7525 := by
  sorry

end plain_cookie_price_l3717_371717


namespace area_of_M_l3717_371759

-- Define the set M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (abs y + abs (4 + y) ≤ 4) ∧
               ((x - y^2 - 4*y - 3) / (2*y - x + 3) ≥ 0) ∧
               (-4 ≤ y) ∧ (y ≤ 0)}

-- Define the area function
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_M : area M = 8 := by sorry

end area_of_M_l3717_371759


namespace optimal_room_allocation_l3717_371710

theorem optimal_room_allocation (total_people : Nat) (large_room_capacity : Nat) 
  (h1 : total_people = 26) (h2 : large_room_capacity = 3) : 
  ∃ (small_room_capacity : Nat), 
    small_room_capacity = total_people - large_room_capacity ∧ 
    small_room_capacity > 0 ∧
    (∀ (x : Nat), x > 0 ∧ x < small_room_capacity → 
      (total_people - large_room_capacity) % x ≠ 0) :=
by sorry

end optimal_room_allocation_l3717_371710


namespace bottle_cap_wrapper_difference_l3717_371741

-- Define the initial counts and newly found items
def initial_bottle_caps : ℕ := 12
def initial_wrappers : ℕ := 11
def found_bottle_caps : ℕ := 58
def found_wrappers : ℕ := 25

-- Define the total counts
def total_bottle_caps : ℕ := initial_bottle_caps + found_bottle_caps
def total_wrappers : ℕ := initial_wrappers + found_wrappers

-- State the theorem
theorem bottle_cap_wrapper_difference :
  total_bottle_caps - total_wrappers = 34 := by
  sorry

end bottle_cap_wrapper_difference_l3717_371741


namespace initial_caps_count_l3717_371752

-- Define the variables
def lost_caps : ℕ := 66
def current_caps : ℕ := 25

-- Define the theorem
theorem initial_caps_count : ∃ initial_caps : ℕ, initial_caps = lost_caps + current_caps :=
  sorry

end initial_caps_count_l3717_371752


namespace fraction_simplification_l3717_371731

theorem fraction_simplification (x : ℝ) : (3*x + 2) / 4 + (x - 4) / 3 = (13*x - 10) / 12 := by
  sorry

end fraction_simplification_l3717_371731


namespace dog_bone_collection_l3717_371783

theorem dog_bone_collection (initial_bones : ℝ) (found_multiplier : ℝ) (given_away : ℝ) (return_fraction : ℝ) : 
  initial_bones = 425.5 →
  found_multiplier = 3.5 →
  given_away = 127.25 →
  return_fraction = 1/4 →
  let total_after_finding := initial_bones + found_multiplier * initial_bones
  let total_after_giving := total_after_finding - given_away
  let returned_bones := return_fraction * given_away
  let final_total := total_after_giving + returned_bones
  final_total = 1819.3125 := by
sorry

end dog_bone_collection_l3717_371783


namespace fraction_to_decimal_l3717_371754

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end fraction_to_decimal_l3717_371754


namespace arrangement_plans_count_l3717_371771

/-- The number of ways to arrange 4 students into 2 classes out of 6 --/
def arrangement_count : ℕ := 90

/-- The total number of classes --/
def total_classes : ℕ := 6

/-- The number of classes to be selected --/
def selected_classes : ℕ := 2

/-- The total number of students to be arranged --/
def total_students : ℕ := 4

/-- The number of students per selected class --/
def students_per_class : ℕ := 2

/-- Theorem stating that the number of arrangement plans is 90 --/
theorem arrangement_plans_count :
  (Nat.choose total_classes selected_classes) *
  (Nat.choose total_students students_per_class) = arrangement_count :=
sorry

end arrangement_plans_count_l3717_371771


namespace jills_earnings_l3717_371751

/-- Calculates the total earnings of a waitress given her work conditions --/
def waitress_earnings (hourly_wage : ℝ) (tip_rate : ℝ) (shifts : ℕ) (hours_per_shift : ℕ) (average_orders_per_hour : ℝ) : ℝ :=
  let total_hours : ℝ := shifts * hours_per_shift
  let wage_earnings : ℝ := total_hours * hourly_wage
  let total_orders : ℝ := total_hours * average_orders_per_hour
  let tip_earnings : ℝ := total_orders * tip_rate
  wage_earnings + tip_earnings

/-- Proves that Jill's earnings for the week are $240.00 --/
theorem jills_earnings : 
  waitress_earnings 4 0.15 3 8 40 = 240 := by
  sorry

end jills_earnings_l3717_371751


namespace white_shirts_count_l3717_371767

/-- The number of white T-shirts in each pack -/
def white_shirts_per_pack : ℕ := sorry

/-- The number of packs of white T-shirts bought -/
def white_packs : ℕ := 3

/-- The number of packs of blue T-shirts bought -/
def blue_packs : ℕ := 2

/-- The number of blue T-shirts in each pack -/
def blue_shirts_per_pack : ℕ := 4

/-- The total number of T-shirts bought -/
def total_shirts : ℕ := 26

theorem white_shirts_count : white_shirts_per_pack = 6 := by
  sorry

end white_shirts_count_l3717_371767


namespace percent_less_than_l3717_371724

theorem percent_less_than (p q : ℝ) (h : p = 1.25 * q) : 
  (p - q) / p = 1/5 := by
  sorry

end percent_less_than_l3717_371724


namespace train_passing_platform_l3717_371756

/-- Given a train of length 1200 meters that crosses a tree in 120 seconds,
    prove that it takes 180 seconds to pass a platform of length 600 meters. -/
theorem train_passing_platform
  (train_length : ℝ)
  (tree_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 1200)
  (h2 : tree_crossing_time = 120)
  (h3 : platform_length = 600) :
  (train_length + platform_length) / (train_length / tree_crossing_time) = 180 :=
sorry

end train_passing_platform_l3717_371756


namespace person_speed_l3717_371719

theorem person_speed (street_length : Real) (crossing_time : Real) (speed : Real) :
  street_length = 300 →
  crossing_time = 4 →
  speed = (street_length / 1000) / (crossing_time / 60) →
  speed = 4.5 := by
  sorry

end person_speed_l3717_371719


namespace monthly_growth_rate_price_reduction_l3717_371784

-- Define the sales data
def august_sales : ℕ := 50000
def october_sales : ℕ := 72000

-- Define the pricing and sales data
def cost_price : ℚ := 40
def original_price : ℚ := 80
def initial_daily_sales : ℕ := 20
def sales_increase_rate : ℚ := 4  -- 2 units per $0.5 decrease = 4 units per $1 decrease
def desired_daily_profit : ℚ := 1400

-- Part 1: Monthly average growth rate
theorem monthly_growth_rate :
  ∃ (x : ℝ), x ≥ 0 ∧ x ≤ 1 ∧ 
  (↑august_sales * (1 + x)^2 : ℝ) = october_sales ∧
  x = 0.2 := by sorry

-- Part 2: Price reduction for promotion
theorem price_reduction :
  ∃ (y : ℚ), y > 0 ∧ y < original_price - cost_price ∧
  (original_price - y - cost_price) * (initial_daily_sales + sales_increase_rate * y) = desired_daily_profit ∧
  y = 30 := by sorry

end monthly_growth_rate_price_reduction_l3717_371784


namespace triangle_with_angle_ratio_2_3_5_is_right_triangle_l3717_371707

theorem triangle_with_angle_ratio_2_3_5_is_right_triangle (a b c : ℝ) 
  (h_triangle : a + b + c = 180)
  (h_ratio : ∃ (x : ℝ), a = 2*x ∧ b = 3*x ∧ c = 5*x) :
  c = 90 := by
sorry

end triangle_with_angle_ratio_2_3_5_is_right_triangle_l3717_371707


namespace max_planes_of_symmetry_l3717_371703

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  planes_of_symmetry : ℕ

/-- Two convex polyhedra do not intersect -/
def do_not_intersect (A B : ConvexPolyhedron) : Prop :=
  sorry

/-- The number of planes of symmetry for a figure consisting of two polyhedra -/
def combined_planes_of_symmetry (A B : ConvexPolyhedron) : ℕ :=
  sorry

theorem max_planes_of_symmetry (A B : ConvexPolyhedron) 
  (h1 : do_not_intersect A B)
  (h2 : A.planes_of_symmetry = 2012)
  (h3 : B.planes_of_symmetry = 2013) :
  combined_planes_of_symmetry A B = 2013 :=
sorry

end max_planes_of_symmetry_l3717_371703


namespace lines_perpendicular_to_plane_are_parallel_l3717_371715

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (l m : Line) (α : Plane) : 
  perpendicular l α → perpendicular m α → parallel l m :=
sorry

end lines_perpendicular_to_plane_are_parallel_l3717_371715


namespace max_xy_value_l3717_371701

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) : 
  x * y ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 4 * x₀^2 + 9 * y₀^2 + 3 * x₀ * y₀ = 30 ∧ x₀ * y₀ = 2 :=
sorry

end max_xy_value_l3717_371701


namespace dartboard_angle_l3717_371708

theorem dartboard_angle (p : ℝ) (θ : ℝ) : 
  0 < p → p < 1 → 0 ≤ θ → θ ≤ 360 →
  (p = θ / 360) → (p = 1 / 8) → θ = 45 := by
  sorry

end dartboard_angle_l3717_371708


namespace vinnie_word_count_excess_l3717_371705

def word_limit : ℕ := 1000
def friday_words : ℕ := 450
def saturday_words : ℕ := 650
def sunday_words : ℕ := 300
def friday_articles : ℕ := 25
def saturday_articles : ℕ := 40
def sunday_articles : ℕ := 15

theorem vinnie_word_count_excess :
  (friday_words + saturday_words + sunday_words) -
  (friday_articles + saturday_articles + sunday_articles) -
  word_limit = 320 := by
  sorry

end vinnie_word_count_excess_l3717_371705


namespace coin_denomination_problem_l3717_371740

theorem coin_denomination_problem (total_coins : ℕ) (twenty_paise_coins : ℕ) (total_value : ℕ) :
  total_coins = 324 →
  twenty_paise_coins = 220 →
  total_value = 7000 →
  (twenty_paise_coins * 20 + (total_coins - twenty_paise_coins) * 25 = total_value) :=
by sorry

end coin_denomination_problem_l3717_371740


namespace total_road_signs_l3717_371722

/-- The number of road signs at four intersections -/
def road_signs (first second third fourth : ℕ) : Prop :=
  (second = first + first / 4) ∧
  (third = 2 * second) ∧
  (fourth = third - 20) ∧
  (first + second + third + fourth = 270)

/-- Theorem: There are 270 road signs in total given the conditions -/
theorem total_road_signs : ∃ (first second third fourth : ℕ),
  first = 40 ∧ road_signs first second third fourth := by
  sorry

end total_road_signs_l3717_371722


namespace min_moves_to_guarantee_coin_find_l3717_371746

/-- Represents the game state with thimbles and a hidden coin. -/
structure ThimbleGame where
  numThimbles : Nat
  numFlipPerMove : Nat

/-- Represents a strategy for playing the game. -/
structure Strategy where
  numMoves : Nat

/-- Determines if a strategy is guaranteed to find the coin. -/
def isGuaranteedStrategy (game : ThimbleGame) (strategy : Strategy) : Prop :=
  ∀ (coinPosition : Nat), coinPosition < game.numThimbles → 
    ∃ (move : Nat), move < strategy.numMoves ∧ 
      (∃ (flippedThimble : Nat), flippedThimble < game.numFlipPerMove ∧ 
        (coinPosition + move) % game.numThimbles = flippedThimble)

/-- The main theorem stating the minimum number of moves required. -/
theorem min_moves_to_guarantee_coin_find (game : ThimbleGame) 
    (h1 : game.numThimbles = 100) (h2 : game.numFlipPerMove = 4) : 
    ∃ (strategy : Strategy), 
      isGuaranteedStrategy game strategy ∧ 
      strategy.numMoves = 33 ∧
      (∀ (otherStrategy : Strategy), 
        isGuaranteedStrategy game otherStrategy → 
        otherStrategy.numMoves ≥ 33) :=
  sorry

end min_moves_to_guarantee_coin_find_l3717_371746


namespace integer_solution_equation_l3717_371728

theorem integer_solution_equation :
  ∀ x y : ℤ, (3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3) ↔ 
  ((x = -6 ∧ y = -6) ∨ (x = 0 ∧ y = 0) ∨ (x = 6 ∧ y = 6)) := by
  sorry

end integer_solution_equation_l3717_371728


namespace scientific_notation_460_billion_l3717_371700

theorem scientific_notation_460_billion :
  (460 * 10^9 : ℝ) = 4.6 * 10^11 := by sorry

end scientific_notation_460_billion_l3717_371700


namespace math_city_intersections_l3717_371796

/-- The number of intersections for n non-parallel streets where no three streets meet at a single point -/
def intersections (n : ℕ) : ℕ := (n - 1) * n / 2

/-- The number of streets in Math City -/
def num_streets : ℕ := 10

theorem math_city_intersections :
  intersections num_streets = 45 :=
by sorry

end math_city_intersections_l3717_371796

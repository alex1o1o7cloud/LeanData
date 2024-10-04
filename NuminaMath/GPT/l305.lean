import Mathlib

namespace total_distance_yards_remaining_yards_l305_305053

structure Distance where
  miles : Nat
  yards : Nat

def marathon_distance : Distance :=
  { miles := 26, yards := 385 }

def miles_to_yards (miles : Nat) : Nat :=
  miles * 1760

def total_yards_in_marathon (d : Distance) : Nat :=
  miles_to_yards d.miles + d.yards

def total_distance_in_yards (d : Distance) (n : Nat) : Nat :=
  n * total_yards_in_marathon d

def remaining_yards (total_yards : Nat) (yards_in_mile : Nat) : Nat :=
  total_yards % yards_in_mile

theorem total_distance_yards_remaining_yards :
    let total_yards := total_distance_in_yards marathon_distance 15
    remaining_yards total_yards 1760 = 495 :=
by
  sorry

end total_distance_yards_remaining_yards_l305_305053


namespace shaded_area_l305_305766

theorem shaded_area 
  (side_of_square : ℝ)
  (arc_radius : ℝ)
  (side_length_eq_sqrt_two : side_of_square = Real.sqrt 2)
  (radius_eq_one : arc_radius = 1) :
  let square_area := 4
  let sector_area := 3 * Real.pi
  let shaded_area := square_area + sector_area
  shaded_area = 4 + 3 * Real.pi :=
by
  sorry

end shaded_area_l305_305766


namespace no_extension_to_countably_additive_l305_305891

-- Define the base set Omega and the algebra A
def Omega : Set ℚ := Set.univ

def A : Set (Set ℚ) := 
  {s | ∃ (U : Finset (ℚ × ℚ)), s = ⋃ (p ∈ U), Set.Ioc p.1 p.2 }

-- Define the finitely additive measure mu
def mu (s : Set ℚ) : ℝ :=
  if h : s ∈ A then ∑ p in (Finset.filter (λ (x : ℚ × ℚ), Set.Ioc x.1 x.2 ⊆ s) (Finset.univ : Finset (ℚ × ℚ))), (p.2 - p.1) else 0

-- The main statement asserting that mu cannot be extended to a countably additive measure on sigma(A)
theorem no_extension_to_countably_additive :
  ¬ ∃ (mu' : Measure ℚ), 
    (∀ s ∈ A, mu' s = mu s) ∧ 
    (∀ (s : Set (Set ℚ)), (countable s ∧ s ⊆ A) → mu' (⋃₀ s) = ∑' (t : Set ℚ) in s, mu' t) := 
begin
  sorry
end

end no_extension_to_countably_additive_l305_305891


namespace total_population_l305_305551

variables (b g t : ℕ)

-- Conditions
def cond1 := b = 4 * g
def cond2 := g = 2 * t

-- Theorem statement
theorem total_population (h1 : cond1 b g) (h2 : cond2 g t) : b + g + t = 11 * b / 8 :=
by sorry

end total_population_l305_305551


namespace awareness_survey_sampling_l305_305213

theorem awareness_survey_sampling
  (students : Set ℝ) -- assumption that defines the set of students
  (grades : Set ℝ) -- assumption that defines the set of grades
  (awareness : ℝ → ℝ) -- assumption defining the awareness function
  (significant_differences : ∀ g1 g2 : ℝ, g1 ≠ g2 → awareness g1 ≠ awareness g2) -- significant differences in awareness among grades
  (first_grade_students : Set ℝ) -- assumption defining the set of first grade students
  (second_grade_students : Set ℝ) -- assumption defining the set of second grade students
  (third_grade_students : Set ℝ) -- assumption defining the set of third grade students
  (students_from_grades : students = first_grade_students ∪ second_grade_students ∪ third_grade_students) -- assumption that the students are from first, second, and third grades
  (representative_method : (simple_random_sampling → False) ∧ (systematic_sampling_method → False))
  : stratified_sampling_method := 
sorry

end awareness_survey_sampling_l305_305213


namespace sqrt_neg2_sq_l305_305509

theorem sqrt_neg2_sq : Real.sqrt ((-2 : ℝ) ^ 2) = 2 := by
  sorry

end sqrt_neg2_sq_l305_305509


namespace max_value_of_linear_combination_l305_305847

theorem max_value_of_linear_combination (x y : ℝ) (h : x^2 - 3 * x + 4 * y = 7) : 
  3 * x + 4 * y ≤ 16 :=
sorry

end max_value_of_linear_combination_l305_305847


namespace find_max_side_length_l305_305949

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l305_305949


namespace smallest_composite_no_prime_factors_less_than_20_l305_305660

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l305_305660


namespace distance_between_X_and_Y_l305_305452

def distance_XY := 31

theorem distance_between_X_and_Y
  (yolanda_rate : ℕ) (bob_rate : ℕ) (bob_walked : ℕ) (time_difference : ℕ) :
  yolanda_rate = 1 →
  bob_rate = 2 →
  bob_walked = 20 →
  time_difference = 1 →
  distance_XY = bob_walked + (bob_walked / bob_rate + time_difference) * yolanda_rate :=
by
  intros hy hb hbw htd
  sorry

end distance_between_X_and_Y_l305_305452


namespace final_score_l305_305074

def dart1 : ℕ := 50
def dart2 : ℕ := 0
def dart3 : ℕ := dart1 / 2

theorem final_score : dart1 + dart2 + dart3 = 75 := by
  sorry

end final_score_l305_305074


namespace max_side_length_l305_305971

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l305_305971


namespace statement_is_true_l305_305420

theorem statement_is_true (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h : ∀ x : ℝ, |x + 2| < b → |(3 * x + 2) + 4| < a) : b ≤ a / 3 :=
by
  sorry

end statement_is_true_l305_305420


namespace total_cost_3m3_topsoil_l305_305609

def topsoil_cost (V C : ℕ) : ℕ :=
  V * C

theorem total_cost_3m3_topsoil : topsoil_cost 3 12 = 36 :=
by
  unfold topsoil_cost
  exact rfl

end total_cost_3m3_topsoil_l305_305609


namespace branches_and_ornaments_l305_305649

def numberOfBranchesAndOrnaments (b t : ℕ) : Prop :=
  (b = t - 1) ∧ (2 * b = t - 1)

theorem branches_and_ornaments : ∃ (b t : ℕ), numberOfBranchesAndOrnaments b t ∧ b = 3 ∧ t = 4 :=
by
  sorry

end branches_and_ornaments_l305_305649


namespace difference_in_money_in_cents_l305_305570

theorem difference_in_money_in_cents (p : ℤ) (h₁ : ℤ) (h₂ : ℤ) 
  (h₁ : Linda_nickels = 7 * p - 2) (h₂ : Carol_nickels = 3 * p + 4) :
  5 * (Linda_nickels - Carol_nickels) = 20 * p - 30 := 
by sorry

end difference_in_money_in_cents_l305_305570


namespace otimes_neg2_neg1_l305_305386

def otimes (a b : Int) : Int := a ^ 2 - Int.natAbs b

theorem otimes_neg2_neg1 : otimes (-2) (-1) = 3 :=
by
  rw otimes
  rfl

end otimes_neg2_neg1_l305_305386


namespace perpendicular_vectors_l305_305030

theorem perpendicular_vectors (b : ℝ) :
  (5 * b - 12 = 0) → b = 12 / 5 :=
by
  intro h
  sorry

end perpendicular_vectors_l305_305030


namespace find_difference_of_a_and_b_l305_305097

-- Define the conditions
variables (a b : ℝ)
axiom cond1 : 4 * a + 3 * b = 8
axiom cond2 : 3 * a + 4 * b = 6

-- Statement for the proof
theorem find_difference_of_a_and_b : a - b = 2 :=
by
  sorry

end find_difference_of_a_and_b_l305_305097


namespace complex_number_in_second_quadrant_l305_305654

open Complex

theorem complex_number_in_second_quadrant (z : ℂ) :
  (Complex.abs z = Real.sqrt 7) →
  (z.re < 0 ∧ z.im > 0) →
  z = -2 + Real.sqrt 3 * Complex.I :=
by
  intros h1 h2
  sorry

end complex_number_in_second_quadrant_l305_305654


namespace infinite_product_a_eq_3_div_5_l305_305942

noncomputable def a : ℕ → ℝ
| 0       := 1 / 3
| (n + 1) := 1 + (a n - 1)^2

theorem infinite_product_a_eq_3_div_5 : 
    (∏ n, a n) = 3 / 5 :=
sorry

end infinite_product_a_eq_3_div_5_l305_305942


namespace sum_other_y_coordinates_l305_305532

-- Given points
structure Point where
  x : ℝ
  y : ℝ

def opposite_vertices (p1 p2 : Point) : Prop :=
  -- conditions defining opposite vertices of a rectangle
  (p1.x ≠ p2.x) ∧ (p1.y ≠ p2.y)

-- Function to sum y-coordinates of two points
def sum_y_coords (p1 p2 : Point) : ℝ :=
  p1.y + p2.y

-- Main theorem to prove
theorem sum_other_y_coordinates (p1 p2 : Point) (h : opposite_vertices p1 p2) :
  sum_y_coords p1 p2 = 11 ↔ 
  (p1 = {x := 1, y := 19} ∨ p1 = {x := 7, y := -8}) ∧ 
  (p2 = {x := 1, y := 19} ∨ p2 = {x := 7, y := -8}) :=
by {
  sorry
}

end sum_other_y_coordinates_l305_305532


namespace max_triangle_side_l305_305988

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l305_305988


namespace max_triangle_side_l305_305987

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l305_305987


namespace sum_invariant_under_permutation_l305_305268

theorem sum_invariant_under_permutation (b : List ℝ) (σ : List ℕ) (hσ : σ.Perm (List.range b.length)) :
  (List.sum b) = (List.sum (σ.map (b.get!))) := by
  sorry

end sum_invariant_under_permutation_l305_305268


namespace point_in_fourth_quadrant_l305_305432

def inFourthQuadrant (x y : Int) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  inFourthQuadrant 2 (-3) :=
by
  sorry

end point_in_fourth_quadrant_l305_305432


namespace focus_of_hyperbola_l305_305514

-- Define the given hyperbola equation and its conversion to standard form
def hyperbola_eq (x y : ℝ) : Prop := -2 * (x - 2)^2 + 3 * (y + 3)^2 - 28 = 0

-- Define the standard form equation of the hyperbola
def standard_form (x y : ℝ) : Prop :=
  ((y + 3)^2 / (28 / 3)) - ((x - 2)^2 / 14) = 1

-- Define the coordinates of one of the foci of the hyperbola
def focus (x y : ℝ) : Prop :=
  x = 2 ∧ y = -3 + Real.sqrt (70 / 3)

-- The theorem statement proving the given coordinates is a focus of the hyperbola
theorem focus_of_hyperbola :
  ∃ x y, hyperbola_eq x y ∧ standard_form x y → focus x y :=
by
  existsi 2, (-3 + Real.sqrt (70 / 3))
  sorry -- Proof is required to substantiate it, placeholder here.

end focus_of_hyperbola_l305_305514


namespace youngest_age_is_20_l305_305202

-- Definitions of the ages
def siblings_ages (y : ℕ) : List ℕ := [y, y+2, y+7, y+11]

-- Condition of the problem: average age is 25
def average_age_25 (y : ℕ) : Prop := (siblings_ages y).sum = 100

-- The statement to be proven
theorem youngest_age_is_20 (y : ℕ) (h : average_age_25 y) : y = 20 :=
  sorry

end youngest_age_is_20_l305_305202


namespace abs_nested_expression_l305_305086

theorem abs_nested_expression : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry

end abs_nested_expression_l305_305086


namespace min_ab_l305_305113

theorem min_ab (a b : ℝ) (h : (1 / a) + (1 / b) = Real.sqrt (a * b)) : a * b ≥ 2 := by
  sorry

end min_ab_l305_305113


namespace inequality_solution_set_l305_305023

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) : (x - 1) / x > 1 ↔ x < 0 :=
by
  sorry

end inequality_solution_set_l305_305023


namespace max_side_length_l305_305970

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l305_305970


namespace nina_weeks_to_afford_game_l305_305886

noncomputable def game_cost : ℝ := 50
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def weekly_allowance : ℝ := 10
noncomputable def saving_rate : ℝ := 0.5

noncomputable def total_cost : ℝ := game_cost + (game_cost * sales_tax_rate)
noncomputable def savings_per_week : ℝ := weekly_allowance * saving_rate
noncomputable def weeks_needed : ℝ := total_cost / savings_per_week

theorem nina_weeks_to_afford_game : weeks_needed = 11 := by
  sorry

end nina_weeks_to_afford_game_l305_305886


namespace range_of_a_for_extreme_points_l305_305541

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * Real.exp (2 * x)

theorem range_of_a_for_extreme_points :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    ∀ a : ℝ, 0 < a ∧ a < (1 / 2) →
    (Real.exp x₁ * (x₁ + 1 - 2 * a * Real.exp x₁) = 0) ∧ 
    (Real.exp x₂ * (x₂ + 1 - 2 * a * Real.exp x₂) = 0)) ↔ 
  ∀ a : ℝ, 0 < a ∧ a < (1 / 2) :=
sorry

end range_of_a_for_extreme_points_l305_305541


namespace park_area_correct_l305_305771

noncomputable def rect_park_area (speed_km_hr : ℕ) (time_min : ℕ) (ratio_l_b : ℕ) : ℕ := by
  let speed_m_min := speed_km_hr * 1000 / 60
  let perimeter := speed_m_min * time_min
  let B := perimeter * 3 / 8
  let L := B / 3
  let area := L * B
  exact area

theorem park_area_correct : rect_park_area 12 8 3 = 120000 := by
  sorry

end park_area_correct_l305_305771


namespace interest_rate_l305_305197

/-- 
Given a principal amount that doubles itself in 10 years at simple interest,
prove that the rate of interest per annum is 10%.
-/
theorem interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (h1 : SI = P) (h2 : T = 10) (h3 : SI = P * R * T / 100) : 
  R = 10 := by
  sorry

end interest_rate_l305_305197


namespace total_roasted_marshmallows_l305_305259

-- Definitions based on problem conditions
def dadMarshmallows : ℕ := 21
def joeMarshmallows := 4 * dadMarshmallows
def dadRoasted := dadMarshmallows / 3
def joeRoasted := joeMarshmallows / 2

-- Theorem to prove the total roasted marshmallows
theorem total_roasted_marshmallows : dadRoasted + joeRoasted = 49 := by
  sorry -- Proof omitted

end total_roasted_marshmallows_l305_305259


namespace initial_books_from_library_l305_305141

-- Definitions of the problem conditions
def booksGivenAway : ℝ := 23.0
def booksLeft : ℝ := 31.0

-- Statement of the problem, proving that the initial number of books
def initialBooks (x : ℝ) : Prop :=
  x = booksGivenAway + booksLeft

-- Main theorem
theorem initial_books_from_library : initialBooks 54.0 :=
by
  -- Proof pending
  sorry

end initial_books_from_library_l305_305141


namespace eval_infinite_product_l305_305507

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, (3:ℝ)^(2 * n / (3:ℝ)^n)

theorem eval_infinite_product : infinite_product = (3:ℝ)^(9 / 2) := by
  sorry

end eval_infinite_product_l305_305507


namespace product_of_consecutive_integers_between_sqrt_50_l305_305323

theorem product_of_consecutive_integers_between_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (sqrt 50 ∈ set.Icc (m : ℝ) (n : ℝ)) ∧ (m * n = 56) := by
  sorry

end product_of_consecutive_integers_between_sqrt_50_l305_305323


namespace shorter_piece_length_l305_305629

/-- A 69-inch board is cut into 2 pieces. One piece is 2 times the length of the other.
    Prove that the length of the shorter piece is 23 inches. -/
theorem shorter_piece_length (x : ℝ) :
  let shorter := x
  let longer := 2 * x
  (shorter + longer = 69) → shorter = 23 :=
by
  intro h
  sorry

end shorter_piece_length_l305_305629


namespace incorrect_locus_proof_l305_305793

-- Conditions given in the problem
def condition_A (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∈ locus ↔ conditions p)

def condition_B (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∉ locus ↔ ¬ conditions p) ∧ (conditions p ↔ p ∈ locus)

def condition_C (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∈ locus → conditions p) ∧ (∃ q, conditions q ∧ q ∈ locus)

def condition_D (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∉ locus ↔ ¬ conditions p) ∧ (p ∈ locus ↔ conditions p)

def condition_E (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (conditions p ↔ p ∈ locus) ∧ (¬ conditions p ↔ p ∉ locus)

-- Statement to be proved
theorem incorrect_locus_proof (locus : Set Point) (conditions : Point → Prop) :
  ¬ condition_C locus conditions :=
sorry

end incorrect_locus_proof_l305_305793


namespace substance_volume_proportional_l305_305201

theorem substance_volume_proportional (k : ℝ) (V₁ V₂ : ℝ) (W₁ W₂ : ℝ) 
  (h1 : V₁ = k * W₁) 
  (h2 : V₂ = k * W₂) 
  (h3 : V₁ = 48) 
  (h4 : W₁ = 112) 
  (h5 : W₂ = 84) 
  : V₂ = 36 := 
  sorry

end substance_volume_proportional_l305_305201


namespace find_x_l305_305815

variable (x : ℕ)

def f (x : ℕ) : ℕ := 2 * x + 5
def g (y : ℕ) : ℕ := 3 * y

theorem find_x (h : g (f x) = 123) : x = 18 :=
by {
  sorry
}

end find_x_l305_305815


namespace point_in_fourth_quadrant_l305_305431

def inFourthQuadrant (x y : Int) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  inFourthQuadrant 2 (-3) :=
by
  sorry

end point_in_fourth_quadrant_l305_305431


namespace domain_of_f_i_l305_305103

variable (f : ℝ → ℝ)

theorem domain_of_f_i (h : ∀ x, -1 ≤ x + 1 ∧ x + 1 ≤ 1) : ∀ x, -2 ≤ x ∧ x ≤ 0 :=
by
  intro x
  specialize h x
  sorry

end domain_of_f_i_l305_305103


namespace homework_duration_reduction_l305_305031

theorem homework_duration_reduction (x : ℝ) (initial_duration final_duration : ℝ) (h_initial : initial_duration = 90) (h_final : final_duration = 60) : 
  90 * (1 - x)^2 = 60 :=
by
  sorry

end homework_duration_reduction_l305_305031


namespace total_daily_salary_l305_305427

def manager_salary : ℕ := 5
def clerk_salary : ℕ := 2
def num_managers : ℕ := 2
def num_clerks : ℕ := 3

theorem total_daily_salary : num_managers * manager_salary + num_clerks * clerk_salary = 16 := by
    sorry

end total_daily_salary_l305_305427


namespace remainder_when_divided_by_8_l305_305203

theorem remainder_when_divided_by_8 (x : ℤ) (h : ∃ k : ℤ, x = 72 * k + 19) : x % 8 = 3 :=
by
  sorry

end remainder_when_divided_by_8_l305_305203


namespace max_triangle_side_l305_305997

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l305_305997


namespace minimum_balls_same_color_minimum_balls_two_white_l305_305777

-- Define the number of black and white balls.
def num_black_balls : Nat := 100
def num_white_balls : Nat := 100

-- Problem 1: Ensure at least 2 balls of the same color.
theorem minimum_balls_same_color (n_black n_white : Nat) (h_black : n_black = num_black_balls) (h_white : n_white = num_white_balls) : 
  3 ≥ 2 :=
by
  sorry

-- Problem 2: Ensure at least 2 white balls.
theorem minimum_balls_two_white (n_black n_white : Nat) (h_black: n_black = num_black_balls) (h_white: n_white = num_white_balls) :
  102 ≥ 2 :=
by
  sorry

end minimum_balls_same_color_minimum_balls_two_white_l305_305777


namespace hamza_bucket_problem_l305_305544

-- Definitions reflecting the problem conditions
def bucket_2_5_capacity : ℝ := 2.5
def bucket_3_0_capacity : ℝ := 3.0
def bucket_5_6_capacity : ℝ := 5.6
def bucket_6_5_capacity : ℝ := 6.5

def initial_fill_in_5_6 : ℝ := bucket_5_6_capacity
def pour_5_6_to_3_0_remaining : ℝ := 5.6 - 3.0
def remaining_in_5_6_after_second_fill : ℝ := bucket_5_6_capacity - 0.5

-- Main problem statement
theorem hamza_bucket_problem : (bucket_6_5_capacity - 2.6 = 3.9) :=
by sorry

end hamza_bucket_problem_l305_305544


namespace michael_final_revenue_l305_305755

noncomputable def total_revenue_before_discount : ℝ :=
  (3 * 45) + (5 * 22) + (7 * 16) + (8 * 10) + (10 * 5)

noncomputable def discount : ℝ := 0.10 * total_revenue_before_discount

noncomputable def discounted_revenue : ℝ := total_revenue_before_discount - discount

noncomputable def sales_tax : ℝ := 0.06 * discounted_revenue

noncomputable def final_revenue : ℝ := discounted_revenue + sales_tax

theorem michael_final_revenue : final_revenue = 464.60 :=
by
  sorry

end michael_final_revenue_l305_305755


namespace ab_value_l305_305721

theorem ab_value (a b : ℚ) 
  (h1 : (a + b) ^ 2 + |b + 5| = b + 5) 
  (h2 : 2 * a - b + 1 = 0) : 
  a * b = -1 / 9 :=
by
  sorry

end ab_value_l305_305721


namespace number_of_solutions_to_subsets_l305_305065

theorem number_of_solutions_to_subsets :
  (nat.card { X : finset ℕ // {1, 2, 3} ⊆ X ∧ X ⊆ {1, 2, 3, 4, 5, 6, 7} } = 16) :=
by
  sorry

end number_of_solutions_to_subsets_l305_305065


namespace tom_jerry_coffee_total_same_amount_total_coffee_l305_305181

noncomputable def total_coffee_drunk (x : ℚ) : ℚ := 
  let jerry_coffee := 1.25 * x
  let tom_drinks := (2/3) * x
  let jerry_drinks := (2/3) * jerry_coffee
  let jerry_remainder := (5/12) * x
  let jerry_gives_tom := (5/48) * x + 3
  tom_drinks + jerry_gives_tom

theorem tom_jerry_coffee_total (x : ℚ) : total_coffee_drunk x = jerry_drinks + (1.25 * x - jerry_gives_tom) := sorry

theorem same_amount_total_coffee (x : ℚ) 
  (h : total_coffee_drunk x = (5/4) * x - ((5/48) * x + 3)) : 
  (1.25 * x + x = 36) :=
by sorry

end tom_jerry_coffee_total_same_amount_total_coffee_l305_305181


namespace cos_diff_trigonometric_identity_l305_305204

-- Problem 1
theorem cos_diff :
  (Real.cos (25 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) - 
   Real.cos (65 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)) = 
  1/2 :=
sorry

-- Problem 2
theorem trigonometric_identity (θ : Real) (h : Real.sin θ + 2 * Real.cos θ = 0) :
  (Real.cos (2 * θ) - Real.sin (2 * θ)) / (1 + (Real.cos θ)^2) = 5/6 :=
sorry

end cos_diff_trigonometric_identity_l305_305204


namespace m_squared_plus_reciprocal_squared_l305_305401

theorem m_squared_plus_reciprocal_squared (m : ℝ) (h : m^2 - 2 * m - 1 = 0) : m^2 + 1 / m^2 = 6 :=
by
  sorry

end m_squared_plus_reciprocal_squared_l305_305401


namespace record_withdrawal_example_l305_305484

-- Definitions based on conditions
def ten_thousand_dollars := 10000
def record_deposit (amount : ℕ) : ℤ := amount / ten_thousand_dollars
def record_withdrawal (amount : ℕ) : ℤ := -(amount / ten_thousand_dollars)

-- Lean 4 statement to prove the problem
theorem record_withdrawal_example :
  (record_deposit 30000 = 3) → (record_withdrawal 20000 = -2) :=
by
  intro h
  sorry

end record_withdrawal_example_l305_305484


namespace how_many_more_red_balls_l305_305745

def r_packs : ℕ := 12
def y_packs : ℕ := 9
def r_balls_per_pack : ℕ := 24
def y_balls_per_pack : ℕ := 20

theorem how_many_more_red_balls :
  (r_packs * r_balls_per_pack) - (y_packs * y_balls_per_pack) = 108 :=
by
  sorry

end how_many_more_red_balls_l305_305745


namespace four_gt_sqrt_fourteen_l305_305066

theorem four_gt_sqrt_fourteen : 4 > Real.sqrt 14 := 
  sorry

end four_gt_sqrt_fourteen_l305_305066


namespace winnie_keeps_balloons_l305_305925

theorem winnie_keeps_balloons :
  let blueBalloons := 15
  let yellowBalloons := 40
  let purpleBalloons := 70
  let orangeBalloons := 90
  let friends := 9
  let totalBalloons := blueBalloons + yellowBalloons + purpleBalloons + orangeBalloons
  (totalBalloons % friends) = 8 := 
by 
  -- Definitions
  let blueBalloons := 15
  let yellowBalloons := 40
  let purpleBalloons := 70
  let orangeBalloons := 90
  let friends := 9
  let totalBalloons := blueBalloons + yellowBalloons + purpleBalloons + orangeBalloons
  -- Conclusion
  show totalBalloons % friends = 8
  sorry

end winnie_keeps_balloons_l305_305925


namespace length_of_bridge_l305_305945

theorem length_of_bridge (train_length : ℕ) (train_speed_kmph : ℕ) (cross_time_sec : ℕ) (bridge_length: ℕ):
  train_length = 110 →
  train_speed_kmph = 45 →
  cross_time_sec = 30 →
  bridge_length = 265 :=
by
  intros h1 h2 h3
  sorry

end length_of_bridge_l305_305945


namespace range_omega_l305_305696

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def f' (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := ω * Real.cos (ω * x + φ)

theorem range_omega (t ω φ : ℝ) (hω_pos : ω > 0) (hf_t_zero : f t ω φ = 0) (hf'_t_pos : f' t ω φ > 0) (no_min_value : ∀ x, t ≤ x ∧ x < t + 1 → ∃ y, y ≠ x ∧ f y ω φ < f x ω φ) : π < ω ∧ ω ≤ (3 * π / 2) :=
sorry

end range_omega_l305_305696


namespace probability_at_least_8_heads_eq_l305_305494

-- Definitions for the given conditions
def total_outcomes : ℕ := 1024

def successful_outcomes : ℕ :=
  Nat.choose 10 8 + Nat.choose 10 9 + Nat.choose 10 10

def probability_of_success : ℚ :=
  successful_outcomes / total_outcomes

-- Theorem stating the final proof problem
theorem probability_at_least_8_heads_eq :
  probability_of_success = 7 / 128 := by
  sorry

end probability_at_least_8_heads_eq_l305_305494


namespace additional_workers_needed_l305_305693

theorem additional_workers_needed :
  let initial_workers := 4
  let initial_parts := 108
  let initial_hours := 3
  let target_parts := 504
  let target_hours := 8
  (target_parts / target_hours) / (initial_parts / (initial_hours * initial_workers)) - initial_workers = 3 := by
  sorry

end additional_workers_needed_l305_305693


namespace find_e_l305_305912

-- Define the conditions and state the theorem.
def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) 
  (h1: ∃ a b c : ℝ, (a + b + c)/3 = -3 ∧ a * b * c = -3 ∧ 3 + d + e + f = -3)
  (h2: Q 0 d e f = 9) : e = -42 :=
by
  sorry

end find_e_l305_305912


namespace parameter_range_exists_solution_l305_305226

theorem parameter_range_exists_solution :
  (∃ b : ℝ, -14 < b ∧ b < 9 ∧ ∃ a : ℝ, ∃ x y : ℝ,
    x^2 + y^2 + 2 * b * (b + x + y) = 81 ∧ y = 5 / ((x - a)^2 + 1)) :=
sorry

end parameter_range_exists_solution_l305_305226


namespace gum_pieces_per_package_l305_305282

theorem gum_pieces_per_package :
  (∀ (packages pieces each_package : ℕ), packages = 9 ∧ pieces = 135 → each_package = pieces / packages → each_package = 15) := 
by
  intros packages pieces each_package
  sorry

end gum_pieces_per_package_l305_305282


namespace total_cost_two_rackets_l305_305455

theorem total_cost_two_rackets (full_price : ℕ) (discount : ℕ) (total_cost : ℕ) :
  (full_price = 60) →
  (discount = full_price / 2) →
  (total_cost = full_price + (full_price - discount)) →
  total_cost = 90 :=
by
  intros h_full_price h_discount h_total_cost
  rw [h_full_price, h_discount] at h_total_cost
  sorry

end total_cost_two_rackets_l305_305455


namespace pencils_per_student_l305_305358

theorem pencils_per_student (total_pencils : ℤ) (num_students : ℤ) (pencils_per_student : ℤ)
  (h1 : total_pencils = 195)
  (h2 : num_students = 65) :
  total_pencils / num_students = 3 :=
by
  sorry

end pencils_per_student_l305_305358


namespace simplify_expression_l305_305491

theorem simplify_expression (x y : ℝ) : 
  (5 * x ^ 2 - 3 * x + 2) * (107 - 107) + (7 * y ^ 2 + 4 * y - 1) * (93 - 93) = 0 := 
by 
  sorry

end simplify_expression_l305_305491


namespace contradiction_example_l305_305697

theorem contradiction_example (x y : ℝ) (h1 : x + y > 2) (h2 : x ≤ 1) (h3 : y ≤ 1) : False :=
by
  sorry

end contradiction_example_l305_305697


namespace smallest_positive_value_is_A_l305_305686

noncomputable def expr_A : ℝ := 12 - 4 * Real.sqrt 8
noncomputable def expr_B : ℝ := 4 * Real.sqrt 8 - 12
noncomputable def expr_C : ℝ := 20 - 6 * Real.sqrt 10
noncomputable def expr_D : ℝ := 60 - 15 * Real.sqrt 16
noncomputable def expr_E : ℝ := 15 * Real.sqrt 16 - 60

theorem smallest_positive_value_is_A :
  expr_A = 12 - 4 * Real.sqrt 8 ∧ 
  expr_B = 4 * Real.sqrt 8 - 12 ∧ 
  expr_C = 20 - 6 * Real.sqrt 10 ∧ 
  expr_D = 60 - 15 * Real.sqrt 16 ∧ 
  expr_E = 15 * Real.sqrt 16 - 60 ∧ 
  expr_A > 0 ∧ 
  expr_A < expr_C := 
sorry

end smallest_positive_value_is_A_l305_305686


namespace reciprocal_neg_half_l305_305166

theorem reciprocal_neg_half : (1 / (- (1 / 2) : ℚ) = -2) :=
by
  sorry

end reciprocal_neg_half_l305_305166


namespace min_value_of_a_plus_b_minus_c_l305_305702

open Real

theorem min_value_of_a_plus_b_minus_c (a b c : ℝ) :
  (∀ (x y : ℝ), 3 * x + 4 * y - 5 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ 3 * x + 4 * y + 5) →
  (∃ c_min, c_min = 2 ∧ ∀ c', c' = a + b - c → c' ≥ c_min) :=
by
  sorry

end min_value_of_a_plus_b_minus_c_l305_305702


namespace tarun_garden_area_l305_305179

theorem tarun_garden_area :
  ∀ (side : ℝ), 
  (1500 / 8 = 4 * side) → 
  (30 * side = 1500) → 
  side^2 = 2197.265625 :=
by
  sorry

end tarun_garden_area_l305_305179


namespace study_group_members_l305_305350

theorem study_group_members (x : ℕ) (h : x * (x - 1) = 90) : x = 10 :=
sorry

end study_group_members_l305_305350


namespace ratio_of_wire_lengths_l305_305221

theorem ratio_of_wire_lengths 
  (bonnie_wire_length : ℕ := 80)
  (roark_wire_length : ℕ := 12000) :
  bonnie_wire_length / roark_wire_length = 1 / 150 :=
by
  sorry

end ratio_of_wire_lengths_l305_305221


namespace ab_cd_not_prime_l305_305748

theorem ab_cd_not_prime (a b c d : ℕ) (ha : a > b) (hb : b > c) (hc : c > d) (hd : d > 0)
  (h : a * c + b * d = (b + d + a - c) * (b + d - a + c)) : ¬ Nat.Prime (a * b + c * d) := 
sorry

end ab_cd_not_prime_l305_305748


namespace total_number_of_workers_l305_305489

-- Definitions based on the given conditions
def avg_salary_total : ℝ := 8000
def avg_salary_technicians : ℝ := 12000
def avg_salary_non_technicians : ℝ := 6000
def num_technicians : ℕ := 7

-- Problem statement in Lean
theorem total_number_of_workers
    (W : ℕ) (N : ℕ)
    (h1 : W * avg_salary_total = num_technicians * avg_salary_technicians + N * avg_salary_non_technicians)
    (h2 : W = num_technicians + N) :
    W = 21 :=
sorry

end total_number_of_workers_l305_305489


namespace distance_between_lines_l305_305160

/-- The graph of the function y = x^2 + ax + b is drawn on a board.
Let the parabola intersect the horizontal lines y = s and y = t at points A, B and C, D respectively,
with A B = 5 and C D = 11. Then the distance between the lines y = s and y = t is 24. -/
theorem distance_between_lines 
  (a b s t : ℝ)
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + a * x1 + b = s) ∧ (x2^2 + a * x2 + b = s) ∧ |x1 - x2| = 5)
  (h2 : ∃ x3 x4 : ℝ, x3 ≠ x4 ∧ (x3^2 + a * x3 + b = t) ∧ (x4^2 + a * x4 + b = t) ∧ |x3 - x4| = 11) :
  |t - s| = 24 := 
by
  sorry

end distance_between_lines_l305_305160


namespace different_meal_combinations_l305_305191

-- Defining the conditions explicitly
def items_on_menu : ℕ := 12

-- A function representing possible combinations of choices for Yann and Camille
def meal_combinations (menu_items : ℕ) : ℕ :=
  menu_items * (menu_items - 1)

-- Theorem stating that given 12 items on the menu, the different combinations of meals is 132
theorem different_meal_combinations : meal_combinations items_on_menu = 132 :=
by
  sorry

end different_meal_combinations_l305_305191


namespace part2_inequality_l305_305710

-- Define the function f and its conditions
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- The main theorem we want to prove
theorem part2_inequality (a b c : ℝ) (h : a^2 + 2 * b^2 + 3 * c^2 = 6) : 
  |a + 2 * b + 3 * c| ≤ 6 :=
by {
-- Proof goes here
sorry
}

end part2_inequality_l305_305710


namespace approximate_number_of_fish_l305_305624

/-
  In a pond, 50 fish were tagged and returned. 
  Later, in another catch of 50 fish, 2 were tagged. 
  Assuming the proportion of tagged fish in the second catch approximates that of the pond,
  prove that the total number of fish in the pond is approximately 1250.
-/

theorem approximate_number_of_fish (N : ℕ) 
  (tagged_in_pond : ℕ := 50) 
  (total_in_second_catch : ℕ := 50) 
  (tagged_in_second_catch : ℕ := 2) 
  (proportion_approx : tagged_in_second_catch / total_in_second_catch = tagged_in_pond / N) :
  N = 1250 :=
by
  sorry

end approximate_number_of_fish_l305_305624


namespace number_of_nintendo_games_to_give_away_l305_305129

-- Define the conditions
def initial_nintendo_games : ℕ := 20
def desired_nintendo_games_left : ℕ := 12

-- Define the proof problem as a Lean theorem
theorem number_of_nintendo_games_to_give_away :
  initial_nintendo_games - desired_nintendo_games_left = 8 :=
by
  sorry

end number_of_nintendo_games_to_give_away_l305_305129


namespace smallest_composite_no_prime_factors_less_than_20_l305_305680

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l305_305680


namespace extra_interest_is_correct_l305_305806

def principal : ℝ := 5000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def simple_interest (P R T : ℝ) : ℝ := P * R * T

def interest1 : ℝ := simple_interest principal rate1 time
def interest2 : ℝ := simple_interest principal rate2 time

def extra_interest : ℝ := interest1 - interest2

theorem extra_interest_is_correct : extra_interest = 600 := by
  sorry

end extra_interest_is_correct_l305_305806


namespace complete_square_k_value_l305_305118

noncomputable def complete_square_form (x : ℝ) : ℝ := x^2 - 7 * x

theorem complete_square_k_value : ∃ a h k : ℝ, complete_square_form x = a * (x - h)^2 + k ∧ k = -49 / 4 :=
by
  use [1, 7/2, -49/4]
  -- This proof step will establish the relationships and the equality
  sorry

end complete_square_k_value_l305_305118


namespace percentage_sales_other_l305_305586

theorem percentage_sales_other (p_pens p_pencils p_markers p_other : ℕ)
(h_pens : p_pens = 25)
(h_pencils : p_pencils = 30)
(h_markers : p_markers = 20)
(h_other : p_other = 100 - (p_pens + p_pencils + p_markers)): p_other = 25 :=
by
  rw [h_pens, h_pencils, h_markers] at h_other
  exact h_other


end percentage_sales_other_l305_305586


namespace product_of_integers_around_sqrt_50_l305_305312

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l305_305312


namespace correct_calculation_l305_305039

theorem correct_calculation (a b : ℝ) : 
  ¬(a * a^3 = a^3) ∧ ¬((a^2)^3 = a^5) ∧ (-a^2 * b)^2 = a^4 * b^2 ∧ ¬(a^3 / a = a^3) :=
by {
  sorry
}

end correct_calculation_l305_305039


namespace max_value_of_y_in_interval_l305_305093

theorem max_value_of_y_in_interval (x : ℝ) (h : 0 < x ∧ x < 1 / 3) : 
  ∃ y_max, ∀ x, 0 < x ∧ x < 1 / 3 → x * (1 - 3 * x) ≤ y_max ∧ y_max = 1 / 12 :=
by sorry

end max_value_of_y_in_interval_l305_305093


namespace dot_product_MN_MO_is_8_l305_305851

-- Define the circle O as a set of points (x, y) such that x^2 + y^2 = 9
def is_circle (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the length of the chord MN in the circle
def chord_length (M N : ℝ × ℝ) : Prop :=
  let (x1, y1) := M
  let (x2, y2) := N
  (x1 - x2)^2 + (y1 - y2)^2 = 16

-- Define the vector MN and MO
def vector_dot_product (M N O : ℝ × ℝ) : ℝ :=
  let (x1, y1) := M
  let (x2, y2) := N
  let (x0, y0) := O
  let v1 := (x2 - x1, y2 - y1)
  let v2 := (x0 - x1, y0 - y1)
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the origin point O (center of the circle)
def O : ℝ × ℝ := (0, 0)

-- The theorem to prove
theorem dot_product_MN_MO_is_8 (M N : ℝ × ℝ) (hM : is_circle M.1 M.2) (hN : is_circle N.1 N.2) (hMN : chord_length M N) :
  vector_dot_product M N O = 8 :=
sorry

end dot_product_MN_MO_is_8_l305_305851


namespace solve_inequality_l305_305152

theorem solve_inequality (x : ℝ) : (3 * x - 5) / 2 > 2 * x → x < -5 :=
by
  sorry

end solve_inequality_l305_305152


namespace problem_solution_l305_305616

theorem problem_solution : (121^2 - 110^2) / 11 = 231 := 
by
  sorry

end problem_solution_l305_305616


namespace sqrt_50_between_consecutive_integers_product_l305_305298

theorem sqrt_50_between_consecutive_integers_product :
  ∃ (m n : ℕ), (m + 1 = n) ∧ (m * m < 50) ∧ (50 < n * n) ∧ (m * n = 56) :=
begin
  sorry
end

end sqrt_50_between_consecutive_integers_product_l305_305298


namespace children_left_on_bus_l305_305803

-- Definitions based on the conditions
def initial_children := 43
def children_got_off := 22

-- The theorem we want to prove
theorem children_left_on_bus (initial_children children_got_off : ℕ) : 
  initial_children - children_got_off = 21 :=
by
  sorry

end children_left_on_bus_l305_305803


namespace solve_inequality_l305_305286

theorem solve_inequality {x : ℝ} : (x^2 - 9 * x + 18 ≤ 0) ↔ 3 ≤ x ∧ x ≤ 6 :=
by
sorry

end solve_inequality_l305_305286


namespace pyramid_distance_to_larger_cross_section_l305_305786

theorem pyramid_distance_to_larger_cross_section
  (A1 A2 : ℝ) (d : ℝ)
  (h : ℝ)
  (hA1 : A1 = 256 * Real.sqrt 2)
  (hA2 : A2 = 576 * Real.sqrt 2)
  (hd : d = 12)
  (h_ratio : (Real.sqrt (A1 / A2)) = 2 / 3) :
  h = 36 := 
  sorry

end pyramid_distance_to_larger_cross_section_l305_305786


namespace solution_set_x_squared_minus_3x_lt_0_l305_305297

theorem solution_set_x_squared_minus_3x_lt_0 : { x : ℝ | x^2 - 3 * x < 0 } = { x : ℝ | 0 < x ∧ x < 3 } :=
by {
  sorry
}

end solution_set_x_squared_minus_3x_lt_0_l305_305297


namespace expression_equals_64_l305_305040

theorem expression_equals_64 :
  let a := 2^3 + 2^3
  let b := 2^3 * 2^3
  let c := (2^3)^3
  let d := 2^12 / 2^2
  b = 2^6 :=
by
  sorry

end expression_equals_64_l305_305040


namespace arithmetic_geometric_sequence_l305_305403

theorem arithmetic_geometric_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (q : ℕ)
  (h₀ : ∀ n, a n = 2^(n-1))
  (h₁ : a 1 = 1)
  (h₂ : a 1 + a 2 + a 3 = 7)
  (h₃ : q > 0) :
  (∀ n, a n = 2^(n-1)) ∧ (∀ n, S n = 2^n - 1) :=
by {
  sorry
}

end arithmetic_geometric_sequence_l305_305403


namespace sqrt_50_between_7_and_8_l305_305315

theorem sqrt_50_between_7_and_8 (x y : ℕ) (h1 : sqrt 50 > 7) (h2 : sqrt 50 < 8) (h3 : y = x + 1) : x * y = 56 :=
by sorry

end sqrt_50_between_7_and_8_l305_305315


namespace percentage_of_material_A_in_second_solution_l305_305817

theorem percentage_of_material_A_in_second_solution 
  (material_A_first_solution : ℝ)
  (material_B_first_solution : ℝ)
  (material_B_second_solution : ℝ)
  (material_A_mixture : ℝ)
  (percentage_first_solution_in_mixture : ℝ)
  (percentage_second_solution_in_mixture : ℝ)
  (total_mixture: ℝ)
  (hyp1 : material_A_first_solution = 20 / 100)
  (hyp2 : material_B_first_solution = 80 / 100)
  (hyp3 : material_B_second_solution = 70 / 100)
  (hyp4 : material_A_mixture = 22 / 100)
  (hyp5 : percentage_first_solution_in_mixture = 80 / 100)
  (hyp6 : percentage_second_solution_in_mixture = 20 / 100)
  (hyp7 : percentage_first_solution_in_mixture + percentage_second_solution_in_mixture = total_mixture)
  : ∃ (x : ℝ), x = 30 := by
  sorry

end percentage_of_material_A_in_second_solution_l305_305817


namespace otimes_example_l305_305380

def otimes (a b : ℤ) : ℤ := a^2 - abs b

theorem otimes_example : otimes (-2) (-1) = 3 := by
  -- Define the variables
  let a := -2
  let b := -1
  -- Unfold the definition of otimes
  have h1 : otimes a b = a^2 - abs b := rfl
  -- Calculate a^2
  have h2 : a^2 = 4 := rfl
  -- Calculate abs b
  have h3 : abs b = 1 := rfl
  -- Calculate otimes a b
  show otimes a b = 3 from by
    rw [h1, h2, h3]
    rfl

end otimes_example_l305_305380


namespace symmetric_line_eq_l305_305594

-- Definitions for the given line equations
def l1 (x y : ℝ) : Prop := 3 * x - y - 3 = 0
def l2 (x y : ℝ) : Prop := x + y - 1 = 0
def l3 (x y : ℝ) : Prop := x - 3 * y - 1 = 0

-- The theorem to prove
theorem symmetric_line_eq (x y : ℝ) (h1: l1 x y) (h2: l2 x y) : l3 x y :=
sorry

end symmetric_line_eq_l305_305594


namespace max_area_enclosed_l305_305146

theorem max_area_enclosed (p : ℕ) (hp : p = 156) (hside : ∀ x, x ∈ ([0, p / 2])) : 
  ∃ A, ∀ x y : ℕ, 2 * (x + y) = p → A ≤ x * y := 
begin
  sorry
end

end max_area_enclosed_l305_305146


namespace unobserved_planet_exists_l305_305004

theorem unobserved_planet_exists
  (n : ℕ) (h_n_eq : n = 15)
  (planets : Fin n → Type)
  (dist : ∀ (i j : Fin n), ℝ)
  (h_distinct : ∀ (i j : Fin n), i ≠ j → dist i j ≠ dist j i)
  (nearest : ∀ i : Fin n, Fin n)
  (h_nearest : ∀ i : Fin n, nearest i ≠ i)
  : ∃ i : Fin n, ∀ j : Fin n, nearest j ≠ i := by
  sorry

end unobserved_planet_exists_l305_305004


namespace max_prime_sequence_l305_305655

open Nat

def sequence_max_prime (k : ℕ) : ℕ :=
  (finset.range 100).countp (λ i, prime (k + i))

theorem max_prime_sequence : ∀ k ≥ 1, sequence_max_prime 2 ≥ sequence_max_prime k :=
begin
  sorry
end

end max_prime_sequence_l305_305655


namespace base_of_isosceles_triangle_l305_305589

theorem base_of_isosceles_triangle (b : ℝ) (h1 : 7 + 7 + b = 22) : b = 8 :=
by {
  sorry
}

end base_of_isosceles_triangle_l305_305589


namespace mutually_exclusive_event_l305_305635

theorem mutually_exclusive_event (A B C D: Prop) 
  (h_A: ¬ (A ∧ (¬D)) ∧ ¬ ¬ D)
  (h_B: ¬ (B ∧ (¬D)) ∧ ¬ ¬ D)
  (h_C: ¬ (C ∧ (¬D)) ∧ ¬ ¬ D)
  (h_D: ¬ (D ∧ (¬D)) ∧ ¬ ¬ D) :
  D :=
sorry

end mutually_exclusive_event_l305_305635


namespace product_of_consecutive_integers_sqrt_50_l305_305318

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (√50 ≥ m) ∧ (√50 < n) ∧ (m * n = 56) :=
by
  use 7, 8
  split
  exact Nat.lt_succ_self 7
  split
  norm_num
  split
  norm_num
  norm_num

end product_of_consecutive_integers_sqrt_50_l305_305318


namespace max_sum_of_digits_of_S_l305_305746

def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def distinctDigits (n : ℕ) : Prop :=
  let digits := (n.digits 10).toFinset
  digits.card = (n.digits 10).length

def digitsRange (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → 1 ≤ d ∧ d ≤ 9

theorem max_sum_of_digits_of_S : ∃ a b S, 
  isThreeDigit a ∧ 
  isThreeDigit b ∧ 
  distinctDigits a ∧ 
  distinctDigits b ∧ 
  digitsRange a ∧ 
  digitsRange b ∧ 
  isThreeDigit S ∧ 
  S = a + b ∧ 
  (S.digits 10).sum = 12 :=
sorry

end max_sum_of_digits_of_S_l305_305746


namespace value_of_stocks_l305_305723

def initial_investment (bonus : ℕ) (stocks : ℕ) : ℕ := bonus / stocks
def final_value_stock_A (initial : ℕ) : ℕ := initial * 2
def final_value_stock_B (initial : ℕ) : ℕ := initial * 2
def final_value_stock_C (initial : ℕ) : ℕ := initial / 2

theorem value_of_stocks 
    (bonus : ℕ) (stocks : ℕ) (h_bonus : bonus = 900) (h_stocks : stocks = 3) : 
    initial_investment bonus stocks * 2 + initial_investment bonus stocks * 2 + initial_investment bonus stocks / 2 = 1350 :=
by
    sorry

end value_of_stocks_l305_305723


namespace two_points_same_color_distance_one_l305_305770

theorem two_points_same_color_distance_one (colored_plane : ℝ × ℝ → Prop) (h : ∀ x, colored_plane x = C1 ∨ colored_plane x = C2) :
  ∃ x y : ℝ × ℝ, colored_plane x = colored_plane y ∧ dist x y = 1 := 
by
  let A := (0 : ℝ, 0 : ℝ)
  let B := (1 : ℝ, 0 : ℝ)
  let C := (0.5 : ℝ, real.sqrt 3 / 2)
  have vertices := [A, B, C]
  apply pigeonhole_principle
  exact vertices
  { exact sorry }

end two_points_same_color_distance_one_l305_305770


namespace largest_divisible_by_two_power_l305_305751
-- Import the necessary Lean library

open scoped BigOperators

-- Prime and Multiples calculation based conditions
def primes_count : ℕ := 25
def multiples_of_four_count : ℕ := 25

-- Number of subsets of {1, 2, 3, ..., 100} with more primes than multiples of 4
def N : ℕ :=
  let pow := 2^50
  pow * (pow / 2 - (∑ k in Finset.range 26, Nat.choose 25 k ^ 2))

-- Theorem stating that the largest integer k such that 2^k divides N is 52
theorem largest_divisible_by_two_power :
  ∃ (k : ℕ), (2^k ∣ N) ∧ (∀ m : ℕ, 2^m ∣ N → m ≤ 52) :=
sorry

end largest_divisible_by_two_power_l305_305751


namespace sqrt_four_squared_l305_305187

theorem sqrt_four_squared : (Real.sqrt 4) ^ 2 = 4 :=
  by
    sorry

end sqrt_four_squared_l305_305187


namespace determine_q_l305_305246

theorem determine_q (p q : ℝ) (hp : p > 1) (hq : q > 1) (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 4) : q = 2 := 
sorry

end determine_q_l305_305246


namespace gary_chickens_l305_305694

theorem gary_chickens (initial_chickens : ℕ) (multiplication_factor : ℕ) 
  (weekly_eggs : ℕ) (days_in_week : ℕ)
  (h1 : initial_chickens = 4)
  (h2 : multiplication_factor = 8)
  (h3 : weekly_eggs = 1344)
  (h4 : days_in_week = 7) :
  (weekly_eggs / days_in_week) / (initial_chickens * multiplication_factor) = 6 :=
by
  sorry

end gary_chickens_l305_305694


namespace steak_weight_in_ounces_l305_305754

-- Definitions from conditions
def pounds : ℕ := 15
def ounces_per_pound : ℕ := 16
def steaks : ℕ := 20

-- The theorem to prove
theorem steak_weight_in_ounces : 
  (pounds * ounces_per_pound) / steaks = 12 := by
  sorry

end steak_weight_in_ounces_l305_305754


namespace negation_of_exactly_one_is_even_l305_305032

def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_is_even (a b c : ℕ) : Prop :=
  ((is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
   (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
   (¬ is_even a ∧ ¬ is_even b ∧ is_even c))

def at_least_two_even (a b c : ℕ) : Prop :=
  ((is_even a ∧ is_even b) ∨ (is_even b ∧ is_even c) ∨ (is_even a ∧ is_even c))

def all_are_odd (a b c : ℕ) : Prop := ¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c 

theorem negation_of_exactly_one_is_even (a b c : ℕ) :
  ¬ exactly_one_is_even a b c ↔ at_least_two_even a b c ∨ all_are_odd a b c := by
  sorry

end negation_of_exactly_one_is_even_l305_305032


namespace total_cost_two_rackets_l305_305456

theorem total_cost_two_rackets (full_price : ℕ) (discount : ℕ) (total_cost : ℕ) :
  (full_price = 60) →
  (discount = full_price / 2) →
  (total_cost = full_price + (full_price - discount)) →
  total_cost = 90 :=
by
  intros h_full_price h_discount h_total_cost
  rw [h_full_price, h_discount] at h_total_cost
  sorry

end total_cost_two_rackets_l305_305456


namespace correct_conclusions_l305_305500

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem correct_conclusions :
  (∃ (a b : ℝ), a < b ∧ f a < f b ∧ ∀ x, a < x ∧ x < b → f x < f (x+1)) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = (x₁ - 2012) ∧ f x₂ = (x₂ - 2012)) :=
by
  sorry

end correct_conclusions_l305_305500


namespace op_evaluation_l305_305273

-- Define the custom operation ⊕
def op (a b c : ℝ) : ℝ := b^2 - 3 * a * c

-- Statement of the theorem we want to prove
theorem op_evaluation : op 2 3 4 = -15 :=
by 
  -- This is a placeholder for the actual proof,
  -- which in a real scenario would involve computing the operation.
  sorry

end op_evaluation_l305_305273


namespace reciprocal_neg_half_l305_305171

theorem reciprocal_neg_half : 1 / (-1 / 2 : ℝ) = (-2 : ℝ) :=
by
  sorry

end reciprocal_neg_half_l305_305171


namespace yellow_side_probability_correct_l305_305805

-- Define the problem scenario
structure CardBox where
  total_cards : ℕ := 8
  green_green_cards : ℕ := 4
  green_yellow_cards : ℕ := 2
  yellow_yellow_cards : ℕ := 2

noncomputable def yellow_side_probability 
  (box : CardBox)
  (picked_is_yellow : Bool) : ℚ :=
  if picked_is_yellow then
    let total_yellow_sides := 2 * box.green_yellow_cards + 2 * box.yellow_yellow_cards
    let yellow_yellow_sides := 2 * box.yellow_yellow_cards
    yellow_yellow_sides / total_yellow_sides
  else 0

theorem yellow_side_probability_correct :
  yellow_side_probability {total_cards := 8, green_green_cards := 4, green_yellow_cards := 2, yellow_yellow_cards := 2} true = 2 / 3 :=
by 
  sorry

end yellow_side_probability_correct_l305_305805


namespace solve_for_y_l305_305727

theorem solve_for_y (y : ℝ) (h : 9 / (y^2) = y / 81) : y = 9 :=
by
  sorry

end solve_for_y_l305_305727


namespace arithmetic_sequence_15th_term_eq_53_l305_305480

theorem arithmetic_sequence_15th_term_eq_53 (a1 : ℤ) (d : ℤ) (n : ℕ) (a_15 : ℤ) 
    (h1 : a1 = -3)
    (h2 : d = 4)
    (h3 : n = 15)
    (h4 : a_15 = a1 + (n - 1) * d) : 
    a_15 = 53 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end arithmetic_sequence_15th_term_eq_53_l305_305480


namespace marked_price_l305_305254

theorem marked_price (x : ℝ) (payment : ℝ) (discount : ℝ) (hx : (payment = 90) ∧ ((x ≤ 100 ∧ discount = 0.1) ∨ (x > 100 ∧ discount = 0.2))) :
  (x = 100 ∨ x = 112.5) := by
  sorry

end marked_price_l305_305254


namespace product_of_consecutive_integers_sqrt_50_l305_305328

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l305_305328


namespace solve_y_l305_305150

theorem solve_y :
  ∀ y : ℚ, 6 * (4 * y - 1) - 3 = 3 * (2 - 5 * y) ↔ y = 5 / 13 :=
by
  sorry

end solve_y_l305_305150


namespace smallest_composite_no_prime_factors_less_20_l305_305665

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l305_305665


namespace roger_current_money_l305_305010

def roger_initial_money : ℕ := 16
def roger_birthday_money : ℕ := 28
def roger_spent_money : ℕ := 25

theorem roger_current_money : roger_initial_money + roger_birthday_money - roger_spent_money = 19 := 
by sorry

end roger_current_money_l305_305010


namespace max_gold_coins_l305_305192

theorem max_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 100) : n = 94 := by
  sorry

end max_gold_coins_l305_305192


namespace average_score_all_students_l305_305336

theorem average_score_all_students 
  (n1 n2 : Nat) 
  (avg1 avg2 : Nat) 
  (h1 : n1 = 20) 
  (h2 : avg1 = 80) 
  (h3 : n2 = 30) 
  (h4 : avg2 = 70) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 74 := 
by
  sorry

end average_score_all_students_l305_305336


namespace inequality_has_no_solutions_l305_305687

theorem inequality_has_no_solutions (x : ℝ) : ¬ (3 * x^2 + 9 * x + 12 ≤ 0) :=
by {
  sorry
}

end inequality_has_no_solutions_l305_305687


namespace annie_milkshakes_l305_305220

theorem annie_milkshakes
  (A : ℕ) (C_hamburger : ℕ) (C_milkshake : ℕ) (H : ℕ) (L : ℕ)
  (initial_money : A = 120)
  (hamburger_cost : C_hamburger = 4)
  (milkshake_cost : C_milkshake = 3)
  (hamburgers_bought : H = 8)
  (money_left : L = 70) :
  ∃ (M : ℕ), A - H * C_hamburger - M * C_milkshake = L ∧ M = 6 :=
by
  sorry

end annie_milkshakes_l305_305220


namespace square_eq_four_implies_two_l305_305250

theorem square_eq_four_implies_two (x : ℝ) (h : x^2 = 4) : x = 2 := 
sorry

end square_eq_four_implies_two_l305_305250


namespace max_side_of_triangle_exists_max_side_of_elevent_l305_305979

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l305_305979


namespace percentage_of_invalid_papers_l305_305511

theorem percentage_of_invalid_papers (total_papers : ℕ) (valid_papers : ℕ) (invalid_papers : ℕ) (percentage_invalid : ℚ) 
  (h1 : total_papers = 400) 
  (h2 : valid_papers = 240) 
  (h3 : invalid_papers = total_papers - valid_ppapers)
  (h4 : percentage_invalid = (invalid_papers : ℚ) / total_papers * 100) : 
  percentage_invalid = 40 :=
by
  sorry

end percentage_of_invalid_papers_l305_305511


namespace six_digit_number_l305_305814

noncomputable def number_of_digits (N : ℕ) : ℕ := sorry

theorem six_digit_number :
  ∀ (N : ℕ),
    (N % 2020 = 0) ∧
    (∀ a b : ℕ, (a ≠ b ∧ N / 10^a % 10 ≠ N / 10^b % 10)) ∧
    (∀ a b : ℕ, (a ≠ b) → ((N / 10^a % 10 = N / 10^b % 10) -> (N % 2020 ≠ 0))) →
    number_of_digits N = 6 :=
sorry

end six_digit_number_l305_305814


namespace jessie_weight_before_jogging_l305_305126

theorem jessie_weight_before_jogging (current_weight lost_weight : ℕ) 
(hc : current_weight = 67)
(hl : lost_weight = 7) : 
current_weight + lost_weight = 74 := 
by
  -- Here we skip the proof part
  sorry

end jessie_weight_before_jogging_l305_305126


namespace smallest_AAB_value_l305_305216

theorem smallest_AAB_value {A B : ℕ} (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_distinct : A ≠ B) (h_eq : 10 * A + B = (1 / 9) * (100 * A + 10 * A + B)) :
  100 * A + 10 * A + B = 225 :=
by
  -- Insert proof here
  sorry

end smallest_AAB_value_l305_305216


namespace days_with_equal_sun_tue_l305_305809

theorem days_with_equal_sun_tue (days_in_month : ℕ) (weekdays : ℕ) (d1 d2 : ℕ) (h1 : days_in_month = 30)
  (h2 : weekdays = 7) (h3 : d1 = 4) (h4 : d2 = 2) :
  ∃ count, count = 3 := by
  sorry

end days_with_equal_sun_tue_l305_305809


namespace sequence_formula_min_value_Sn_min_value_Sn_completion_l305_305402

-- Define the sequence sum Sn
def Sn (n : ℕ) : ℤ := n^2 - 48 * n

-- General term of the sequence
def an (n : ℕ) : ℤ :=
  match n with
  | 0     => 0 -- Conventionally, sequences start from 1 in these problems
  | (n+1) => 2 * (n + 1) - 49

-- Prove that the general term of the sequence produces the correct sum
theorem sequence_formula (n : ℕ) (h : 0 < n) : an n = 2 * n - 49 := by
  sorry

-- Prove that the minimum value of Sn is -576 and occurs at n = 24
theorem min_value_Sn : ∃ n : ℕ, Sn n = -576 ∧ ∀ m : ℕ, Sn m ≥ -576 := by
  use 24
  sorry

-- Alternative form of the theorem using the square completion form 
theorem min_value_Sn_completion (n : ℕ) : Sn n = (n - 24)^2 - 576 := by
  sorry

end sequence_formula_min_value_Sn_min_value_Sn_completion_l305_305402


namespace game_prob_comparison_l305_305347

theorem game_prob_comparison
  (P_H : ℚ) (P_T : ℚ) (h : P_H = 3/4 ∧ P_T = 1/4)
  (independent : ∀ (n : ℕ), (1 - P_H)^n = (1 - P_T)^n) :
  ((P_H^4 + P_T^4) = (P_H^3 * P_T^2 + P_T^3 * P_H^2) + 1/4) :=
by
  sorry

end game_prob_comparison_l305_305347


namespace unit_digit_seven_power_500_l305_305790

def unit_digit (x : ℕ) : ℕ := x % 10

theorem unit_digit_seven_power_500 :
  unit_digit (7 ^ 500) = 1 := 
by
  sorry

end unit_digit_seven_power_500_l305_305790


namespace james_spent_6_dollars_l305_305740

-- Define the constants based on the conditions
def cost_milk : ℝ := 3
def cost_bananas : ℝ := 2
def tax_rate : ℝ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℝ := cost_milk + cost_bananas

-- Define the sales tax
def sales_tax : ℝ := total_cost_before_tax * tax_rate

-- Define the total amount spent
def total_amount_spent : ℝ := total_cost_before_tax + sales_tax

-- The theorem to prove that James spent $6
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end james_spent_6_dollars_l305_305740


namespace total_items_in_jar_l305_305605

/--
A jar contains 3409.0 pieces of candy and 145.0 secret eggs with a prize.
We aim to prove that the total number of items in the jar is 3554.0.
-/
theorem total_items_in_jar :
  let number_of_pieces_of_candy := 3409.0
  let number_of_secret_eggs := 145.0
  number_of_pieces_of_candy + number_of_secret_eggs = 3554.0 :=
by
  sorry

end total_items_in_jar_l305_305605


namespace total_telephone_bill_second_month_l305_305077

theorem total_telephone_bill_second_month
  (F C1 : ℝ) 
  (h1 : F + C1 = 46)
  (h2 : F + 2 * C1 = 76) :
  F + 2 * C1 = 76 :=
by
  sorry

end total_telephone_bill_second_month_l305_305077


namespace sqrt_expression_range_l305_305423

theorem sqrt_expression_range (x : ℝ) : x + 3 ≥ 0 ∧ x ≠ 0 ↔ x ≥ -3 ∧ x ≠ 0 :=
by
  sorry

end sqrt_expression_range_l305_305423


namespace rounding_and_scientific_notation_l305_305012

-- Define the original number
def original_number : ℕ := 1694000

-- Define the function to round to the nearest hundred thousand
def round_to_nearest_hundred_thousand (n : ℕ) : ℕ :=
  ((n + 50000) / 100000) * 100000

-- Define the function to convert to scientific notation
def to_scientific_notation (n : ℕ) : String :=
  let base := n / 1000000
  let exponent := 6
  s!"{base}.0 × 10^{exponent}"

-- Assert the equivalence
theorem rounding_and_scientific_notation :
  to_scientific_notation (round_to_nearest_hundred_thousand original_number) = "1.7 × 10^{6}" :=
by
  sorry

end rounding_and_scientific_notation_l305_305012


namespace park_attraction_children_count_l305_305436

theorem park_attraction_children_count
  (C : ℕ) -- Number of children
  (entrance_fee : ℕ := 5) -- Entrance fee per person
  (kids_attr_fee : ℕ := 2) -- Attraction fee for kids
  (adults_attr_fee : ℕ := 4) -- Attraction fee for adults
  (parents : ℕ := 2) -- Number of parents
  (grandmother : ℕ := 1) -- Number of grandmothers
  (total_cost : ℕ := 55) -- Total cost paid
  (entry_eq : entrance_fee * (C + parents + grandmother) + kids_attr_fee * C + adults_attr_fee * (parents + grandmother) = total_cost) :
  C = 4 :=
by
  sorry

end park_attraction_children_count_l305_305436


namespace product_of_consecutive_integers_sqrt_50_l305_305330

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l305_305330


namespace students_who_like_both_l305_305426

def total_students : ℕ := 50
def apple_pie_lovers : ℕ := 22
def chocolate_cake_lovers : ℕ := 20
def neither_dessert_lovers : ℕ := 15

theorem students_who_like_both : 
  (apple_pie_lovers + chocolate_cake_lovers) - (total_students - neither_dessert_lovers) = 7 :=
by
  -- Calculation steps (skipped)
  sorry

end students_who_like_both_l305_305426


namespace otimes_neg2_neg1_l305_305384

def otimes (a b : Int) : Int := a ^ 2 - Int.natAbs b

theorem otimes_neg2_neg1 : otimes (-2) (-1) = 3 :=
by
  rw otimes
  rfl

end otimes_neg2_neg1_l305_305384


namespace max_triangle_side_l305_305996

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l305_305996


namespace set_difference_P_M_l305_305856

open Set

noncomputable def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2009}
noncomputable def P : Set ℕ := {y | 2 ≤ y ∧ y ≤ 2010}

theorem set_difference_P_M : P \ M = {2010} :=
by
  sorry

end set_difference_P_M_l305_305856


namespace original_number_is_80_l305_305200

theorem original_number_is_80 (x : ℝ) (h1 : 1.125 * x - 0.75 * x = 30) : x = 80 :=
by
  sorry

end original_number_is_80_l305_305200


namespace max_triangle_side_l305_305986

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l305_305986


namespace determine_k_completed_square_l305_305119

theorem determine_k_completed_square (x : ℝ) :
  ∃ (a h k : ℝ), a * (x - h)^2 + k = x^2 - 7 * x ∧ k = -49/4 := sorry

end determine_k_completed_square_l305_305119


namespace fraction_of_termite_ridden_homes_collapsing_l305_305887

variable (T : ℕ) -- T represents the total number of homes
variable (termiteRiddenFraction : ℚ := 1/3) -- Fraction of homes that are termite-ridden
variable (termiteRiddenNotCollapsingFraction : ℚ := 1/7) -- Fraction of homes that are termite-ridden but not collapsing

theorem fraction_of_termite_ridden_homes_collapsing :
  termiteRiddenFraction - termiteRiddenNotCollapsingFraction = 4/21 :=
by
  -- Proof goes here
  sorry

end fraction_of_termite_ridden_homes_collapsing_l305_305887


namespace arithmetic_sequence_value_l305_305405

theorem arithmetic_sequence_value (a : ℕ) (h : 2 * a = 12) : a = 6 :=
by
  sorry

end arithmetic_sequence_value_l305_305405


namespace initial_bottle_caps_l305_305762

variable (x : Nat)

theorem initial_bottle_caps (h : x + 3 = 29) : x = 26 := by
  sorry

end initial_bottle_caps_l305_305762


namespace james_spent_6_dollars_l305_305742

-- Define the cost of items
def cost_milk : ℕ := 3
def cost_bananas : ℕ := 2

-- Define the sales tax rate as a decimal
def sales_tax_rate : ℚ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℕ := cost_milk + cost_bananas

-- Define the sales tax amount
def sales_tax_amount : ℚ := sales_tax_rate * total_cost_before_tax

-- Define the total amount spent
def total_amount_spent : ℚ := total_cost_before_tax + sales_tax_amount

-- The proof statement
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end james_spent_6_dollars_l305_305742


namespace range_of_g_l305_305613

noncomputable def g (x : ℝ) : ℝ := 1 / x ^ 2 + 3

theorem range_of_g : set.image g {x : ℝ | x ≠ 0} = set.Ici 3 := by
  sorry

end range_of_g_l305_305613


namespace debby_bottles_l305_305069

noncomputable def number_of_bottles_initial : ℕ := 301
noncomputable def number_of_bottles_drank : ℕ := 144
noncomputable def number_of_bottles_left : ℕ := 157

theorem debby_bottles:
  (number_of_bottles_initial - number_of_bottles_drank) = number_of_bottles_left :=
sorry

end debby_bottles_l305_305069


namespace find_i_value_for_S_i_l305_305257

theorem find_i_value_for_S_i :
  ∃ (i : ℕ), (3 * 6 - 2 ≤ i ∧ i < 3 * 6 + 1) ∧ (1000 ≤ 31 * 2^6) ∧ (31 * 2^6 ≤ 3000) ∧ i = 2 :=
by sorry

end find_i_value_for_S_i_l305_305257


namespace arithmetic_geometric_sequence_l305_305173

-- Let {a_n} be an arithmetic sequence
-- And let a_1, a_2, a_3 form a geometric sequence
-- Given that a_5 = 1, we aim to prove that a_10 = 1
theorem arithmetic_geometric_sequence (a : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_geom : a 1 * a 3 = (a 2) ^ 2)
  (h_a5 : a 5 = 1) :
  a 10 = 1 :=
sorry

end arithmetic_geometric_sequence_l305_305173


namespace amount_kept_by_Tim_l305_305784

-- Define the conditions
def totalAmount : ℝ := 100
def percentageGivenAway : ℝ := 0.2

-- Prove the question == answer
theorem amount_kept_by_Tim : totalAmount - totalAmount * percentageGivenAway = 80 :=
by
  -- Here the proof would take place
  sorry

end amount_kept_by_Tim_l305_305784


namespace solve_cubic_equation_l305_305896

theorem solve_cubic_equation : ∀ x : ℝ, (x^3 - 5*x^2 + 6*x - 2 = 0) → (x = 2) :=
by
  intro x
  intro h
  sorry

end solve_cubic_equation_l305_305896


namespace incentive_given_to_john_l305_305621

-- Conditions (definitions)
def commission_held : ℕ := 25000
def advance_fees : ℕ := 8280
def amount_given_to_john : ℕ := 18500

-- Problem statement
theorem incentive_given_to_john : (amount_given_to_john - (commission_held - advance_fees)) = 1780 := 
by
  sorry

end incentive_given_to_john_l305_305621


namespace no_solution_l305_305578

theorem no_solution (m n : ℕ) : (5 + 3 * Real.sqrt 2) ^ m ≠ (3 + 5 * Real.sqrt 2) ^ n :=
sorry

end no_solution_l305_305578


namespace inequality_has_no_solutions_l305_305688

theorem inequality_has_no_solutions (x : ℝ) : ¬ (3 * x^2 + 9 * x + 12 ≤ 0) :=
by {
  sorry
}

end inequality_has_no_solutions_l305_305688


namespace frank_won_skee_ball_tickets_l305_305042

noncomputable def tickets_whack_a_mole : ℕ := 33
noncomputable def candies_bought : ℕ := 7
noncomputable def tickets_per_candy : ℕ := 6
noncomputable def total_tickets_spent : ℕ := candies_bought * tickets_per_candy
noncomputable def tickets_skee_ball : ℕ := total_tickets_spent - tickets_whack_a_mole

theorem frank_won_skee_ball_tickets : tickets_skee_ball = 9 :=
  by
  sorry

end frank_won_skee_ball_tickets_l305_305042


namespace light_stripes_total_area_l305_305775

theorem light_stripes_total_area (x : ℝ) (h : 45 * x = 135) :
  2 * x + 4 * x + 6 * x + 8 * x = 60 := 
sorry

end light_stripes_total_area_l305_305775


namespace age_difference_l305_305052

theorem age_difference (M S : ℕ) (h1 : S = 16) (h2 : M + 2 = 2 * (S + 2)) : M - S = 18 :=
by
  sorry

end age_difference_l305_305052


namespace chooseOneFromEachCategory_chooseTwoDifferentTypes_l305_305046

-- Define the number of different paintings in each category
def traditionalChinesePaintings : ℕ := 5
def oilPaintings : ℕ := 2
def watercolorPaintings : ℕ := 7

-- Part (1): Prove that the number of ways to choose one painting from each category is 70
theorem chooseOneFromEachCategory : traditionalChinesePaintings * oilPaintings * watercolorPaintings = 70 := by
  sorry

-- Part (2): Prove that the number of ways to choose two paintings of different types is 59
theorem chooseTwoDifferentTypes :
  (traditionalChinesePaintings * oilPaintings) + 
  (traditionalChinesePaintings * watercolorPaintings) + 
  (oilPaintings * watercolorPaintings) = 59 := by
  sorry

end chooseOneFromEachCategory_chooseTwoDifferentTypes_l305_305046


namespace total_cost_of_two_rackets_l305_305454

axiom racket_full_price : ℕ
axiom price_of_first_racket : racket_full_price = 60
axiom price_of_second_racket : racket_full_price / 2 = 30

theorem total_cost_of_two_rackets : 60 + 30 = 90 :=
sorry

end total_cost_of_two_rackets_l305_305454


namespace roasted_marshmallows_total_l305_305260

def joe_marshmallows (dads_marshmallows : ℕ) := 4 * dads_marshmallows

def roasted_marshmallows (total_marshmallows : ℕ) (fraction : ℕ) := total_marshmallows / fraction

theorem roasted_marshmallows_total :
  let dads_marshmallows := 21 in
  let joe_marshmallows := joe_marshmallows dads_marshmallows in
  let dads_roasted := roasted_marshmallows dads_marshmallows 3 in
  let joe_roasted := roasted_marshmallows joe_marshmallows 2 in
  dads_roasted + joe_roasted = 49 :=
by
  sorry

end roasted_marshmallows_total_l305_305260


namespace parabola_distance_P_to_F_l305_305239

variables {t : ℝ} (P : ℝ × ℝ) (F : ℝ × ℝ)

-- Definitions for conditions
def parabola_param (t : ℝ) : ℝ × ℝ := (4 * t^2, 4 * t)
def P := (3 : ℝ, 2 * Real.sqrt 3)
def F := (1 : ℝ, 0 : ℝ)
def distance (a b : ℝ × ℝ) : ℝ := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- The proof statement
theorem parabola_distance_P_to_F : distance P F = 4 := by
  sorry

end parabola_distance_P_to_F_l305_305239


namespace groups_needed_for_sampling_l305_305334

def total_students : ℕ := 600
def sample_size : ℕ := 20

theorem groups_needed_for_sampling : (total_students / sample_size = 30) :=
by
  sorry

end groups_needed_for_sampling_l305_305334


namespace f_at_1_l305_305272

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom fg_eq : ∀ x : ℝ, f x + g x = x^3 - x^2 + 1

theorem f_at_1 : f 1 = 1 := by
  sorry

end f_at_1_l305_305272


namespace max_side_length_l305_305963

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l305_305963


namespace product_of_consecutive_integers_between_sqrt_50_l305_305324

theorem product_of_consecutive_integers_between_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (sqrt 50 ∈ set.Icc (m : ℝ) (n : ℝ)) ∧ (m * n = 56) := by
  sorry

end product_of_consecutive_integers_between_sqrt_50_l305_305324


namespace max_S_2017_l305_305706

noncomputable def max_S (a b c : ℕ) : ℕ := a + b + c

theorem max_S_2017 :
  ∀ (a b c : ℕ),
  a + b = 1014 →
  c - b = 497 →
  a > b →
  max_S a b c = 2017 :=
by
  intros a b c h1 h2 h3
  sorry

end max_S_2017_l305_305706


namespace star_eq_zero_iff_x_eq_5_l305_305388

/-- Define the operation * on real numbers -/
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

/-- Proposition stating that x = 5 is the solution to (x - 4) * 1 = 0 -/
theorem star_eq_zero_iff_x_eq_5 (x : ℝ) : (star (x-4) 1 = 0) ↔ x = 5 :=
by
  sorry

end star_eq_zero_iff_x_eq_5_l305_305388


namespace correct_oblique_projection_conclusions_l305_305183

def oblique_projection (shape : Type) : Type := shape

theorem correct_oblique_projection_conclusions :
  (oblique_projection Triangle = Triangle) ∧
  (oblique_projection Parallelogram = Parallelogram) ↔
  (oblique_projection Square ≠ Square) ∧
  (oblique_projection Rhombus ≠ Rhombus) :=
by
  sorry

end correct_oblique_projection_conclusions_l305_305183


namespace smallest_composite_no_prime_lt_20_l305_305677

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l305_305677


namespace max_side_length_l305_305969

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l305_305969


namespace expected_adjacent_black_l305_305588

noncomputable def ExpectedBlackPairs :=
  let totalCards := 104
  let blackCards := 52
  let totalPairs := 103
  let probAdjacentBlack := (blackCards - 1) / (totalPairs)
  blackCards * probAdjacentBlack

theorem expected_adjacent_black :
  ExpectedBlackPairs = 2601 / 103 :=
by
  sorry

end expected_adjacent_black_l305_305588


namespace probability_sum_odd_l305_305049

-- Conditions
def balls : Finset (Fin 15) := 
  { Fin.mk 0 (by norm_num) , Fin.mk 1 (by norm_num), Fin.mk 2 (by norm_num), Fin.mk 3 (by norm_num),
    Fin.mk 4 (by norm_num), Fin.mk 5 (by norm_num), Fin.mk 6 (by norm_num), Fin.mk 7 (by norm_num),
    Fin.mk 8 (by norm_num), Fin.mk 9 (by norm_num), Fin.mk 10 (by norm_num), Fin.mk 11 (by norm_num),
    Fin.mk 12 (by norm_num), Fin.mk 13 (by norm_num), Fin.mk 14 (by norm_num) }

def draw_count : Nat := 7

-- Proof statement
theorem probability_sum_odd : 
  (∃ favorable : ℚ, favorable = 3192 / 6435) →
  (∃ total : ℚ, total = 1) →
  ( (3192 / 6435 : ℚ) = (1064 / 2145 : ℚ) ) :=
by
  sorry

end probability_sum_odd_l305_305049


namespace Jane_possible_numbers_l305_305563

def is_factor (a b : ℕ) : Prop := b % a = 0
def in_range (n : ℕ) : Prop := 500 ≤ n ∧ n ≤ 4000

def Jane_number (m : ℕ) : Prop :=
  is_factor 180 m ∧
  is_factor 42 m ∧
  in_range m

theorem Jane_possible_numbers :
  Jane_number 1260 ∧ Jane_number 2520 ∧ Jane_number 3780 :=
by
  sorry

end Jane_possible_numbers_l305_305563


namespace max_triangle_side_l305_305989

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l305_305989


namespace geometric_number_difference_l305_305375

-- Definitions
def is_geometric_sequence (a b c d : ℕ) : Prop := ∃ r : ℚ, b = a * r ∧ c = a * r^2 ∧ d = a * r^3

def is_valid_geometric_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧  -- 4-digit number
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ -- distinct digits
    is_geometric_sequence a b c d ∧ -- geometric sequence
    n = a * 1000 + b * 100 + c * 10 + d -- digits form the number

-- Theorem statement
theorem geometric_number_difference : 
  ∃ (m M : ℕ), is_valid_geometric_number m ∧ is_valid_geometric_number M ∧ (M - m = 7173) :=
sorry

end geometric_number_difference_l305_305375


namespace range_of_m_l305_305412

open Real

noncomputable def f (x : ℝ) : ℝ := 1 + sin (2 * x)
noncomputable def g (x m : ℝ) : ℝ := 2 * (cos x)^2 + m

theorem range_of_m (x₀ : ℝ) (m : ℝ) (h₀ : 0 ≤ x₀ ∧ x₀ ≤ π / 2) (h₁ : f x₀ ≥ g x₀ m) : m ≤ sqrt 2 :=
by
  sorry

end range_of_m_l305_305412


namespace distance_between_vertices_hyperbola_l305_305395

-- Defining the hyperbola equation and necessary constants
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2) / 64 - (y^2) / 81 = 1

-- Proving the distance between the vertices is 16
theorem distance_between_vertices_hyperbola : ∀ x y : ℝ, hyperbola_eq x y → 16 = 16 :=
by
  intros x y h
  sorry

end distance_between_vertices_hyperbola_l305_305395


namespace determine_pairs_l305_305070

theorem determine_pairs (p : ℕ) (hp: Nat.Prime p) :
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ p^x - y^3 = 1 ∧ ((x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2)) := 
sorry

end determine_pairs_l305_305070


namespace fraction_of_eggs_hatched_l305_305572

variable (x : ℚ)
variable (survived_first_month_fraction : ℚ := 3/4)
variable (survived_first_year_fraction : ℚ := 2/5)
variable (geese_survived : ℕ := 100)
variable (total_eggs : ℕ := 500)

theorem fraction_of_eggs_hatched :
  (x * survived_first_month_fraction * survived_first_year_fraction * total_eggs : ℚ) = geese_survived → x = 2/3 :=
by 
  intro h
  sorry

end fraction_of_eggs_hatched_l305_305572


namespace number_of_women_l305_305367

theorem number_of_women
    (n : ℕ) -- number of men
    (d_m : ℕ) -- number of dances each man had
    (d_w : ℕ) -- number of dances each woman had
    (total_men : n = 15) -- there are 15 men
    (each_man_dances : d_m = 4) -- each man danced with 4 women
    (each_woman_dances : d_w = 3) -- each woman danced with 3 men
    (total_dances : n * d_m = w * d_w): -- total dances are the same when counted from both sides
  w = 20 := sorry -- There should be exactly 20 women.


end number_of_women_l305_305367


namespace variance_transformation_l305_305104

-- Definitions
def ξ_distribution (ξ : ℕ → ℝ) (a : ℝ) : Prop :=
ξ 0 = a ∧ ξ 1 = 1 - 2 * a ∧ ξ 2 = 1 / 4

def a_equation (a : ℝ) : Prop :=
a + (1 - 2 * a) + 1 / 4 = 1

def expected_value (ξ : ℕ → ℝ) (a : ℝ) : ℝ :=
0 * ξ 0 + 1 * ξ 1 + 2 * ξ 2

def variance (ξ : ℕ → ℝ) (a : ℝ) : ℝ :=
ξ 0 * (0 - expected_value ξ a) ^ 2 + ξ 1 * (1 - expected_value ξ a) ^ 2 + ξ 2 * (2 - expected_value ξ a) ^ 2

-- Proof statement
theorem variance_transformation (ξ : ℕ → ℝ) (a : ℝ) (h_dist : ξ_distribution ξ a) (h_eq : a_equation(a)) :
  4 * variance ξ a = 2 := 
by
  sorry

end variance_transformation_l305_305104


namespace floor_neg_seven_thirds_l305_305523

theorem floor_neg_seven_thirds : Int.floor (-7 / 3 : ℚ) = -3 := by
  sorry

end floor_neg_seven_thirds_l305_305523


namespace floor_neg_seven_thirds_l305_305522

theorem floor_neg_seven_thirds : ⌊-7 / 3⌋ = -3 :=
sorry

end floor_neg_seven_thirds_l305_305522


namespace diff_g_eq_l305_305389

def g (n : ℤ) : ℚ := (1/6) * n * (n+1) * (n+3)

theorem diff_g_eq :
  ∀ (r : ℤ), g r - g (r - 1) = (3/2) * r^2 + (5/2) * r :=
by
  intro r
  sorry

end diff_g_eq_l305_305389


namespace james_spent_6_dollars_l305_305743

-- Define the cost of items
def cost_milk : ℕ := 3
def cost_bananas : ℕ := 2

-- Define the sales tax rate as a decimal
def sales_tax_rate : ℚ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℕ := cost_milk + cost_bananas

-- Define the sales tax amount
def sales_tax_amount : ℚ := sales_tax_rate * total_cost_before_tax

-- Define the total amount spent
def total_amount_spent : ℚ := total_cost_before_tax + sales_tax_amount

-- The proof statement
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end james_spent_6_dollars_l305_305743


namespace simplify_polynomial_l305_305704

open Polynomial

def arithmetic_sequence (a_0 d : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a_0 + n * d

def p_n (a : ℕ → ℝ) (x : ℝ) (n : ℕ) : ℝ :=
  ∑ k in finset.range (n + 1), a k * (nat.choose n k) * (x ^ k) * ((1 - x) ^ (n - k))

theorem simplify_polynomial (a_0 d : ℝ) (a : ℕ → ℝ) (h : arithmetic_sequence a_0 d a)
  (n : ℕ) (x : ℝ) : p_n a x n = a_0 + n * d * x :=
by
  sorry

end simplify_polynomial_l305_305704


namespace smallest_composite_no_prime_factors_less_than_20_l305_305675

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l305_305675


namespace crayons_left_l305_305781

-- Define initial number of crayons and the number taken by Mary
def initial_crayons : ℝ := 7.5
def taken_crayons : ℝ := 2.25

-- Calculate remaining crayons
def remaining_crayons := initial_crayons - taken_crayons

-- Prove that the remaining crayons are 5.25
theorem crayons_left : remaining_crayons = 5.25 := by
  sorry

end crayons_left_l305_305781


namespace max_side_of_triangle_exists_max_side_of_elevent_l305_305975

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l305_305975


namespace magnitude_z_l305_305879

open Complex

theorem magnitude_z
  (z w : ℂ)
  (h1 : abs (2 * z - w) = 25)
  (h2 : abs (z + 2 * w) = 5)
  (h3 : abs (z + w) = 2) : abs z = 9 := 
by 
  sorry

end magnitude_z_l305_305879


namespace no_valid_conference_division_l305_305555

theorem no_valid_conference_division (num_teams : ℕ) (matches_per_team : ℕ) :
  num_teams = 30 → matches_per_team = 82 → 
  ¬ ∃ (k : ℕ) (x y z : ℕ), k + (num_teams - k) = num_teams ∧
                          x + y + z = (num_teams * matches_per_team) / 2 ∧
                          z = ((x + y + z) / 2) := 
by
  sorry

end no_valid_conference_division_l305_305555


namespace sin_double_angle_tangent_identity_l305_305094

theorem sin_double_angle_tangent_identity (x : ℝ) 
  (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.sin (2 * x) = 3 / 5 :=
by
  -- proof is omitted
  sorry

end sin_double_angle_tangent_identity_l305_305094


namespace find_max_side_length_l305_305951

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l305_305951


namespace min_sum_ab_l305_305846

theorem min_sum_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (2 / b) = 2) :
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_sum_ab_l305_305846


namespace point_on_xaxis_equidistant_l305_305208

theorem point_on_xaxis_equidistant :
  ∃ (A : ℝ × ℝ), A.2 = 0 ∧ 
                  dist A (-3, 2) = dist A (4, -5) ∧ 
                  A = (2, 0) :=
by
  sorry

end point_on_xaxis_equidistant_l305_305208


namespace max_triangle_side_l305_305990

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l305_305990


namespace find_sum_of_abs_roots_l305_305837

variable {p q r n : ℤ}

theorem find_sum_of_abs_roots (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2024) (h3 : p * q * r = -n) :
  |p| + |q| + |r| = 100 :=
  sorry

end find_sum_of_abs_roots_l305_305837


namespace triangle_angle_A_triangle_bc_range_l305_305548

theorem triangle_angle_A (a b c A B C : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (ha : a = b * Real.sin C + c * Real.sin B)
  (hb : b = c * Real.sin A + a * Real.sin C)
  (hc : c = a * Real.sin B + b * Real.sin A)
  (h_eq : (Real.sqrt 3) * a * Real.sin C + a * Real.cos C = c + b)
  (h_angles_sum : A + B + C = π) :
    A = π/3 := -- π/3 radians equals 60 degrees
sorry

theorem triangle_bc_range (a b c : ℝ) (h : a = Real.sqrt 3) :
  Real.sqrt 3 < b + c ∧ b + c ≤ 2 * Real.sqrt 3 := 
sorry

end triangle_angle_A_triangle_bc_range_l305_305548


namespace product_of_consecutive_integers_sqrt_50_l305_305303

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), 49 < 50 ∧ 50 < 64 ∧ n = m + 1 ∧ m * n = 56 :=
by {
  let m := 7,
  let n := 8,
  have h1 : 49 < 50 := by norm_num,
  have h2 : 50 < 64 := by norm_num,
  exact ⟨m, n, h1, h2, rfl, by norm_num⟩,
  sorry -- proof skipped
}

end product_of_consecutive_integers_sqrt_50_l305_305303


namespace transition_term_l305_305921

theorem transition_term (k : ℕ) : (2 * k + 2) + (2 * k + 3) = (2 * (k + 1) + 1) + (2 * k + 2) :=
by
  sorry

end transition_term_l305_305921


namespace conference_hall_initial_people_l305_305124

theorem conference_hall_initial_people (x : ℕ)  
  (h1 : 3 ∣ x) 
  (h2 : 4 ∣ (2 * x / 3))
  (h3 : (x / 2) = 27) : 
  x = 54 := 
by 
  sorry

end conference_hall_initial_people_l305_305124


namespace reciprocal_neg_half_l305_305165

theorem reciprocal_neg_half : (1 / (- (1 / 2) : ℚ) = -2) :=
by
  sorry

end reciprocal_neg_half_l305_305165


namespace find_omega_l305_305101

noncomputable def omega : ℝ := (Real.sqrt 3) / 3

theorem find_omega 
  (ω : ℝ) 
  (h_sym : ∀ (x : ℝ), 
    y = sqrt 2 * sin (ω * x) - cos (ω * x) → 
    P1 = (x, (sqrt 2 * sin (ω * x) - cos (ω * x))) → 
    P2 = (kπ + x, (sqrt 2 * sin (ω * (kπ + x)) - cos (ω * (kπ + x))))) 
  (h_perpendicular : ∀ (x1 x2 : ℝ), 
    tangent_at P1 = dy/dx ∣ P1 → 
    tangent_at P2 = dy/dx ∣ P2 → 
    (dy/dx ∣ P1) * (dy/dx ∣ P2) = -1) 
  (h_pos : ω > 0) : 
  ω = (Real.sqrt 3) / 3 := 
sorry

end find_omega_l305_305101


namespace find_max_side_length_l305_305958

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l305_305958


namespace sandwich_not_condiment_percentage_l305_305941

theorem sandwich_not_condiment_percentage :
  (total_weight : ℝ) → (condiment_weight : ℝ) →
  total_weight = 150 → condiment_weight = 45 →
  ((total_weight - condiment_weight) / total_weight) * 100 = 70 :=
by
  intros total_weight condiment_weight h_total h_condiment
  sorry

end sandwich_not_condiment_percentage_l305_305941


namespace reciprocal_of_minus_one_half_l305_305169

theorem reciprocal_of_minus_one_half : (1 / (-1 / 2)) = -2 := 
by sorry

end reciprocal_of_minus_one_half_l305_305169


namespace inverse_proportion_function_l305_305597

theorem inverse_proportion_function (f : ℝ → ℝ) (h : ∀ x, f x = 1/x) : f 1 = 1 := 
by
  sorry

end inverse_proportion_function_l305_305597


namespace isosceles_triangle_apex_angle_l305_305464

theorem isosceles_triangle_apex_angle (base_angle : ℝ) (h_base_angle : base_angle = 42) : 
  180 - 2 * base_angle = 96 :=
by
  sorry

end isosceles_triangle_apex_angle_l305_305464


namespace smallest_composite_no_prime_factors_lt_20_l305_305667

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l305_305667


namespace sum_of_specific_terms_l305_305133

theorem sum_of_specific_terms 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h1 : S 3 = 9) 
  (h2 : S 6 = 36) 
  (h3 : ∀ n, S n = n * (a 1) + d * n * (n - 1) / 2) :
  a 7 + a 8 + a 9 = 45 := 
sorry

end sum_of_specific_terms_l305_305133


namespace nina_weeks_to_afford_game_l305_305885

noncomputable def game_cost : ℝ := 50
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def weekly_allowance : ℝ := 10
noncomputable def saving_rate : ℝ := 0.5

noncomputable def total_cost : ℝ := game_cost + (game_cost * sales_tax_rate)
noncomputable def savings_per_week : ℝ := weekly_allowance * saving_rate
noncomputable def weeks_needed : ℝ := total_cost / savings_per_week

theorem nina_weeks_to_afford_game : weeks_needed = 11 := by
  sorry

end nina_weeks_to_afford_game_l305_305885


namespace solve_log_equation_l305_305603

theorem solve_log_equation (x : ℝ) :
  real.logb 2 (x^2 - 20 * x + 96) = 5 ↔ (x = 16 ∨ x = 4) :=
by
  sorry

end solve_log_equation_l305_305603


namespace map_float_time_l305_305519

theorem map_float_time
  (t₀ t₁ : Nat) -- times representing 12:00 PM and 12:21 PM in minutes since midnight
  (v_w v_b : ℝ) -- constant speed of water current and boat in still water
  (h₀ : t₀ = 12 * 60) -- t₀ is 12:00 PM
  (h₁ : t₁ = 12 * 60 + 21) -- t₁ is 12:21 PM
  : t₁ - t₀ = 21 := 
  sorry

end map_float_time_l305_305519


namespace num_roses_given_l305_305778

theorem num_roses_given (n : ℕ) (m : ℕ) (x : ℕ) :
  n = 28 → 
  (∀ (b g : ℕ), b + g = n → b * g = 45 * x) →
  (num_roses : ℕ) = 4 * x →
  (num_tulips : ℕ) = 10 * num_roses →
  (num_daffodils : ℕ) = x →
  num_roses = 16 :=
by
  sorry

end num_roses_given_l305_305778


namespace students_per_minibus_calculation_l305_305738

-- Define the conditions
variables (vans minibusses total_students students_per_van : ℕ)
variables (students_per_minibus : ℕ)

-- Define the given conditions based on the problem
axiom six_vans : vans = 6
axiom four_minibusses : minibusses = 4
axiom ten_students_per_van : students_per_van = 10
axiom total_students_are_156 : total_students = 156

-- Define the problem statement in Lean
theorem students_per_minibus_calculation
  (h1 : vans = 6)
  (h2 : minibusses = 4)
  (h3 : students_per_van = 10)
  (h4 : total_students = 156) :
  students_per_minibus = 24 :=
sorry

end students_per_minibus_calculation_l305_305738


namespace max_side_of_triangle_exists_max_side_of_elevent_l305_305985

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l305_305985


namespace difference_largest_smallest_geometric_l305_305374

open Nat

noncomputable def is_geometric_sequence (a b c d : ℕ) : Prop :=
  b = a * 2 / 3 ∧ c = a * (2 / 3)^2 ∧ d = a * (2 / 3)^3 ∧ 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem difference_largest_smallest_geometric : 
  exists (largest smallest : ℕ), 
  (is_geometric_sequence (largest / 1000) ((largest % 1000) / 100) ((largest % 100) / 10) (largest % 10)) ∧ 
  (is_geometric_sequence (smallest / 1000) ((smallest % 1000) / 100) ((smallest % 100) / 10) (smallest % 10)) ∧ 
  largest = 9648 ∧ smallest = 1248 ∧ largest - smallest = 8400 :=
begin
  sorry
end

end difference_largest_smallest_geometric_l305_305374


namespace reese_spending_l305_305144

-- Definitions used in Lean 4 statement
variable (S : ℝ := 11000)
variable (M : ℝ := 0.4 * S)
variable (A : ℝ := 1500)
variable (L : ℝ := 2900)

-- Lean 4 verification statement
theorem reese_spending :
  ∃ (P : ℝ), S - (P * S + M + A) = L ∧ P * 100 = 20 :=
by
  sorry

end reese_spending_l305_305144


namespace length_of_first_train_l305_305503

theorem length_of_first_train
    (speed_first_train_kmph : ℝ) 
    (speed_second_train_kmph : ℝ) 
    (time_to_cross_seconds : ℝ) 
    (length_second_train_meters : ℝ)
    (H1 : speed_first_train_kmph = 120)
    (H2 : speed_second_train_kmph = 80)
    (H3 : time_to_cross_seconds = 9)
    (H4 : length_second_train_meters = 300.04) : 
    ∃ (length_first_train : ℝ), length_first_train = 200 :=
by 
    let relative_speed_m_per_s := (speed_first_train_kmph + speed_second_train_kmph) * 1000 / 3600
    let combined_length := relative_speed_m_per_s * time_to_cross_seconds
    let length_first_train := combined_length - length_second_train_meters
    use length_first_train
    sorry

end length_of_first_train_l305_305503


namespace number_of_women_l305_305368

theorem number_of_women
    (n : ℕ) -- number of men
    (d_m : ℕ) -- number of dances each man had
    (d_w : ℕ) -- number of dances each woman had
    (total_men : n = 15) -- there are 15 men
    (each_man_dances : d_m = 4) -- each man danced with 4 women
    (each_woman_dances : d_w = 3) -- each woman danced with 3 men
    (total_dances : n * d_m = w * d_w): -- total dances are the same when counted from both sides
  w = 20 := sorry -- There should be exactly 20 women.


end number_of_women_l305_305368


namespace sqrt_50_between_7_and_8_l305_305314

theorem sqrt_50_between_7_and_8 (x y : ℕ) (h1 : sqrt 50 > 7) (h2 : sqrt 50 < 8) (h3 : y = x + 1) : x * y = 56 :=
by sorry

end sqrt_50_between_7_and_8_l305_305314


namespace find_angle_C_l305_305252

theorem find_angle_C (a b c A B C : ℝ) (h₀ : 0 < C) (h₁ : C < Real.pi)
  (h₂ : 2 * c * Real.sin A = a * Real.tan C) :
  C = Real.pi / 3 :=
sorry

end find_angle_C_l305_305252


namespace solve_for_x_l305_305895

theorem solve_for_x (x : ℝ) :
  (2 * x - 30) / 3 = (5 - 3 * x) / 4 + 1 → x = 147 / 17 := 
by
  intro h
  sorry

end solve_for_x_l305_305895


namespace correct_operation_among_options_l305_305186

theorem correct_operation_among_options (A B C D : Prop) (cond_A : A = (sqrt 4 = ±2))
  (cond_B : B = (sqrt 4)^2 = 4) (cond_C : C = (sqrt (-4)^2) = -4) (cond_D : D = (-sqrt 4)^2 = -4) :
  B ∧ ¬A ∧ ¬C ∧ ¬D :=
by
  sorry

end correct_operation_among_options_l305_305186


namespace savanna_more_giraffes_l305_305461

-- Definitions based on conditions
def lions_safari := 100
def snakes_safari := lions_safari / 2
def giraffes_safari := snakes_safari - 10

def lions_savanna := 2 * lions_safari
def snakes_savanna := 3 * snakes_safari

-- Totals given and to calculate giraffes in Savanna
def total_animals_savanna := 410

-- Prove that Savanna has 20 more giraffes than Safari
theorem savanna_more_giraffes :
  ∃ (giraffes_savanna : ℕ), giraffes_savanna = total_animals_savanna - lions_savanna - snakes_savanna ∧
  giraffes_savanna - giraffes_safari = 20 :=
  by
  sorry

end savanna_more_giraffes_l305_305461


namespace complete_the_square_l305_305218

theorem complete_the_square (x : ℝ) : (x^2 - 8*x + 15 = 0) → ((x - 4)^2 = 1) :=
by
  intro h
  have eq1 : x^2 - 8*x + 15 = 0 := h
  sorry

end complete_the_square_l305_305218


namespace expression_simplifies_to_one_l305_305894

-- Define x in terms of the given condition
def x : ℚ := (1 / 2) ^ (-1 : ℤ) + (-3) ^ (0 : ℤ)

-- Define the given expression
def expr (x : ℚ) : ℚ := (((x^2 - 1) / (x^2 - 2 * x + 1)) - (1 / (x - 1))) / (3 / (x - 1))

-- Define the theorem stating the equivalence
theorem expression_simplifies_to_one : expr x = 1 := by
  sorry

end expression_simplifies_to_one_l305_305894


namespace range_of_a_l305_305593

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 2 then -x + 6 else 3 + Real.log x / Real.log a

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x : ℝ, 4 ≤ x → (if x ≤ 2 then -x + 6 else 3 + Real.log x / Real.log a) ≥ 4) :
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l305_305593


namespace average_of_six_starting_from_d_plus_one_l305_305285

theorem average_of_six_starting_from_d_plus_one (c d : ℝ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  (c + 6) = ((d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 6 := 
by 
-- Proof omitted; end with sorry
sorry

end average_of_six_starting_from_d_plus_one_l305_305285


namespace transformed_curve_eq_l305_305083

-- Define the original ellipse curve
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Define the transformation
def transform (x y x' y' : ℝ) : Prop := x' = x ∧ y' = 2 * y

-- Prove the transformed curve satisfies x'^2 + y'^2 = 4
theorem transformed_curve_eq :
  ∀ (x y x' y' : ℝ), ellipse x y → transform x y x' y' → (x'^2 + y'^2 = 4) :=
by
  intros x y x' y' h_ellipse h_transform
  simp [ellipse, transform] at *
  sorry

end transformed_curve_eq_l305_305083


namespace find_percentage_l305_305933

theorem find_percentage (P : ℝ) (h : P / 100 * 3200 = 0.20 * 650 + 190) : P = 10 :=
by 
  sorry

end find_percentage_l305_305933


namespace water_level_decrease_3m_l305_305114

-- Definitions from conditions
def increase (amount : ℝ) : ℝ := amount
def decrease (amount : ℝ) : ℝ := -amount

-- The claim to be proven
theorem water_level_decrease_3m : decrease 3 = -3 :=
by
  sorry

end water_level_decrease_3m_l305_305114


namespace julie_can_print_100_newspapers_l305_305128

def num_boxes : ℕ := 2
def packages_per_box : ℕ := 5
def sheets_per_package : ℕ := 250
def sheets_per_newspaper : ℕ := 25

theorem julie_can_print_100_newspapers :
  (num_boxes * packages_per_box * sheets_per_package) / sheets_per_newspaper = 100 := by
  sorry

end julie_can_print_100_newspapers_l305_305128


namespace scaled_multiplication_l305_305862

theorem scaled_multiplication (h : 213 * 16 = 3408) : 0.016 * 2.13 = 0.03408 :=
by
  sorry

end scaled_multiplication_l305_305862


namespace bicycles_sold_saturday_l305_305756

variable (S : ℕ)

theorem bicycles_sold_saturday :
  let net_increase_friday := 15 - 10
  let net_increase_saturday := 8 - S
  let net_increase_sunday := 11 - 9
  (net_increase_friday + net_increase_saturday + net_increase_sunday = 3) → 
  S = 12 :=
by
  intros h
  sorry

end bicycles_sold_saturday_l305_305756


namespace part_I_part_II_l305_305411

noncomputable def f (x : ℝ) := x * (Real.log x - 1) + Real.log x + 1

theorem part_I :
  let f_tangent (x y : ℝ) := x - y - 1
  (∀ x y, f_tangent x y = 0 ↔ y = x - 1) ∧ f_tangent 1 (f 1) = 0 :=
by
  sorry

theorem part_II (m : ℝ) :
  (∀ x : ℝ, x > 0 → x^2 + x * (m - (Real.log x + 1 / x)) + 1 ≥ 0) → m ≥ -1 :=
by
  sorry

end part_I_part_II_l305_305411


namespace batsman_average_after_12th_innings_l305_305804

theorem batsman_average_after_12th_innings 
  (A : ℕ) 
  (h1 : 75 = (A + 12)) 
  (h2 : 11 * A + 75 = 12 * (A + 1)) :
  (A + 1) = 64 :=
by 
  sorry

end batsman_average_after_12th_innings_l305_305804


namespace find_max_side_length_l305_305953

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l305_305953


namespace central_angle_of_regular_hexagon_l305_305905

theorem central_angle_of_regular_hexagon:
  ∀ (α : ℝ), 
  (∃ n : ℕ, n = 6 ∧ n * α = 360) →
  α = 60 :=
by
  sorry

end central_angle_of_regular_hexagon_l305_305905


namespace complete_the_square_k_l305_305116

theorem complete_the_square_k (x : ℝ) : ∃ k : ℝ, (∃ a h: ℝ, (a = 1) ∧ (h = 7 / 2) ∧ (x^2 - 7*x = a * (x - h)^2 + k)) → k = -49 / 4 :=
by
  sorry

end complete_the_square_k_l305_305116


namespace tigers_home_games_l305_305287

-- Definitions based on the conditions
def losses : ℕ := 12
def ties : ℕ := losses / 2
def wins : ℕ := 38

-- Statement to prove
theorem tigers_home_games : losses + ties + wins = 56 := by
  sorry

end tigers_home_games_l305_305287


namespace ava_average_speed_l305_305076

noncomputable def initial_odometer : ℕ := 14941
noncomputable def final_odometer : ℕ := 15051
noncomputable def elapsed_time : ℝ := 4 -- hours

theorem ava_average_speed :
  (final_odometer - initial_odometer) / elapsed_time = 27.5 :=
by
  sorry

end ava_average_speed_l305_305076


namespace regina_total_cost_l305_305460

-- Definitions
def daily_cost : ℝ := 30
def mileage_cost : ℝ := 0.25
def days_rented : ℝ := 3
def miles_driven : ℝ := 450
def fixed_fee : ℝ := 15

-- Proposition for total cost
noncomputable def total_cost : ℝ := daily_cost * days_rented + mileage_cost * miles_driven + fixed_fee

-- Theorem statement
theorem regina_total_cost : total_cost = 217.5 := by
  sorry

end regina_total_cost_l305_305460


namespace nextSimultaneousRingingTime_l305_305818

-- Define the intervals
def townHallInterval := 18
def universityTowerInterval := 24
def fireStationInterval := 30

-- Define the start time (in minutes from 00:00)
def startTime := 8 * 60 -- 8:00 AM

-- Define the least common multiple function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Prove the next simultaneous ringing time
theorem nextSimultaneousRingingTime : 
  let lcmIntervals := lcm (lcm townHallInterval universityTowerInterval) fireStationInterval 
  startTime + lcmIntervals = 14 * 60 := -- 14:00 equals 2:00 PM in minutes
by
  -- You can replace the proof with the actual detailed proof.
  sorry

end nextSimultaneousRingingTime_l305_305818


namespace central_angle_of_regular_hexagon_l305_305903

-- Define the total degrees in a circle
def total_degrees_in_circle : ℝ := 360

-- Define the number of sides in a regular hexagon
def sides_in_hexagon : ℕ := 6

-- Theorems to prove that the central angle of a regular hexagon is 60°
theorem central_angle_of_regular_hexagon :
  total_degrees_in_circle / sides_in_hexagon = 60 :=
by
  sorry

end central_angle_of_regular_hexagon_l305_305903


namespace divisibility_theorem_l305_305266

theorem divisibility_theorem (n : ℕ) (h1 : n > 0) (h2 : ¬(2 ∣ n)) (h3 : ¬(3 ∣ n)) (k : ℤ) :
  (k + 1) ^ n - k ^ n - 1 ∣ k ^ 2 + k + 1 :=
sorry

end divisibility_theorem_l305_305266


namespace probability_sum_die_rolls_odd_l305_305474

theorem probability_sum_die_rolls_odd 
  (h1 : ∀ (c1 c2 c3 : bool), c1 ∨ c2 ∨ c3)
  (h2 : ∀ (num_heads : ℕ), num_heads ≤ 3) : 
  probability (sum_die_rolls_odd h1 h2) = 7 / 16 :=
sorry

-- Definitions required for the theorem
def sum_die_rolls_odd 
  (h1 : ∀ (c1 c2 c3 : bool), c1 ∨ c2 ∨ c3)
  (h2 : ∀ (num_heads : ℕ), num_heads ≤ 3) : Event :=
  -- Here the exact formalization of the conditions goes, which would need to
  -- encompass the scenario of number of heads and calculation of the sum being odd.
  sorry

end probability_sum_die_rolls_odd_l305_305474


namespace largest_whole_number_lt_150_l305_305598

theorem largest_whole_number_lt_150 : ∃ (x : ℕ), (x <= 16 ∧ ∀ y : ℕ, y < 17 → 9 * y < 150) :=
by
  sorry

end largest_whole_number_lt_150_l305_305598


namespace part1_part2_l305_305843

theorem part1 (m : ℂ) : (m * (m + 2)).re = 0 ∧ (m^2 + m - 2).im ≠ 0 → m = 0 := by
  sorry

theorem part2 (m : ℝ) : (m * (m + 2) > 0 ∧ m^2 + m - 2 < 0) → 0 < m ∧ m < 1 := by
  sorry

end part1_part2_l305_305843


namespace find_real_solutions_l305_305081

noncomputable def polynomial_expression (x : ℝ) : ℝ := (x - 2)^2 * (x - 4) * (x - 1)

theorem find_real_solutions :
  ∀ (x : ℝ), (x ≠ 3) ∧ (x ≠ 5) ∧ (polynomial_expression x = 1) ↔ (x = 1 ∨ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2) := sorry

end find_real_solutions_l305_305081


namespace f_at_neg_one_l305_305237

def f : ℝ → ℝ := sorry

theorem f_at_neg_one :
  (∀ x : ℝ, f (x / (1 + x)) = x) →
  f (-1) = -1 / 2 :=
by
  intro h
  -- proof omitted for clarity
  sorry

end f_at_neg_one_l305_305237


namespace determine_k_completed_square_l305_305120

theorem determine_k_completed_square (x : ℝ) :
  ∃ (a h k : ℝ), a * (x - h)^2 + k = x^2 - 7 * x ∧ k = -49/4 := sorry

end determine_k_completed_square_l305_305120


namespace joe_and_dad_total_marshmallows_roasted_l305_305258

theorem joe_and_dad_total_marshmallows_roasted :
  (let dads_marshmallows := 21
       dads_roasted := dads_marshmallows / 3
       joes_marshmallows := 4 * dads_marshmallows
       joes_roasted := joes_marshmallows / 2
   in dads_roasted + joes_roasted = 49) :=
by
  let dads_marshmallows := 21
  let dads_roasted := dads_marshmallows / 3
  let joes_marshmallows := 4 * dads_marshmallows
  let joes_roasted := joes_marshmallows / 2
  show dads_roasted + joes_roasted = 49 from sorry

end joe_and_dad_total_marshmallows_roasted_l305_305258


namespace part1_part2_l305_305715

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2*a + 1)

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
by sorry

theorem part2 (a : ℝ) : (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l305_305715


namespace algebra_statements_correct_l305_305092

theorem algebra_statements_correct (a b c : ℝ) (h₁ : a ≠ 0) :
  (∃ m n : ℝ, m ≠ n ∧ a * m^2 + b * m + c = a * n^2 + b * n + c) ∧
  (ac < 0 → ∃ m n : ℝ, m > n ∧ a * m^2 + b * m + c < 0 ∧ 0 < a * n^2 + b * n + c) ∧
  (ab > 0 → ∃ p q : ℝ, p ≠ q ∧ a * p^2 + b * p + c = a * q^2 + b * q + c ∧ p + q < 0) :=
sorry

end algebra_statements_correct_l305_305092


namespace distance_from_origin_l305_305637

theorem distance_from_origin :
  ∃ (m : ℝ), m = Real.sqrt (108 + 8 * Real.sqrt 10) ∧
              (∃ (x y : ℝ), y = 8 ∧ 
                            (x - 2)^2 + (y - 5)^2 = 49 ∧ 
                            x = 2 + 2 * Real.sqrt 10 ∧ 
                            m = Real.sqrt ((x^2) + (y^2))) :=
by
  sorry

end distance_from_origin_l305_305637


namespace solve_for_m_l305_305483

theorem solve_for_m (m : ℝ) : 
  (∀ x : ℝ, (x = 2) → ((m - 2) * x = 5 * (x + 1))) → (m = 19 / 2) :=
by
  intro h
  have h1 := h 2
  sorry  -- proof can be filled in later

end solve_for_m_l305_305483


namespace selling_price_correct_l305_305011

def initial_cost : ℕ := 800
def repair_cost : ℕ := 200
def gain_percent : ℕ := 40
def total_cost := initial_cost + repair_cost
def gain := (gain_percent * total_cost) / 100
def selling_price := total_cost + gain

theorem selling_price_correct : selling_price = 1400 := 
by
  sorry

end selling_price_correct_l305_305011


namespace central_angle_of_regular_hexagon_l305_305904

-- Define the total degrees in a circle
def total_degrees_in_circle : ℝ := 360

-- Define the number of sides in a regular hexagon
def sides_in_hexagon : ℕ := 6

-- Theorems to prove that the central angle of a regular hexagon is 60°
theorem central_angle_of_regular_hexagon :
  total_degrees_in_circle / sides_in_hexagon = 60 :=
by
  sorry

end central_angle_of_regular_hexagon_l305_305904


namespace polynomial_factorization_l305_305623

noncomputable def poly_1 : Polynomial ℤ := (Polynomial.C 1 * Polynomial.X ^ 3 + Polynomial.C 1 * Polynomial.X ^ 2 + Polynomial.C 1)
noncomputable def poly_2 : Polynomial ℤ :=
  Polynomial.C 1 * Polynomial.X ^ 12 - Polynomial.C 1 * Polynomial.X ^ 11 +
  Polynomial.C 1 * Polynomial.X ^ 9 - Polynomial.C 1 * Polynomial.X ^ 8 +
  Polynomial.C 1 * Polynomial.X ^ 6 - Polynomial.C 1 * Polynomial.X ^ 4 +
  Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 1 * Polynomial.X + Polynomial.C 1
noncomputable def polynomial_expression : Polynomial ℤ := Polynomial.X ^ 15 + Polynomial.X ^ 10 + Polynomial.C 1

theorem polynomial_factorization : polynomial_expression = poly_1 * poly_2 :=
  by { sorry }

end polynomial_factorization_l305_305623


namespace find_max_side_length_l305_305955

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l305_305955


namespace correct_calculation_l305_305417

theorem correct_calculation (x : ℝ) (h : 63 + x = 69) : 36 / x = 6 :=
by
  sorry

end correct_calculation_l305_305417


namespace problem_I_problem_II_l305_305236

-- Declaration of function f(x)
def f (x a b : ℝ) := |x + a| - |x - b|

-- Proof 1: When a = 1, b = 1, solve the inequality f(x) > 1
theorem problem_I (x : ℝ) : (f x 1 1) > 1 ↔ x > 1/2 := by
  sorry

-- Proof 2: If the maximum value of the function f(x) is 2, prove that (1/a) + (1/b) ≥ 2
theorem problem_II (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max_f : ∀ x, f x a b ≤ 2) : 1 / a + 1 / b ≥ 2 := by
  sorry

end problem_I_problem_II_l305_305236


namespace parabola_line_intersection_ratio_l305_305717

theorem parabola_line_intersection_ratio
  (p : ℝ) (hp : 0 < p)
  (A B F : ℝ × ℝ)
  (hA : A.2 ^ 2 = 2 * p * A.1)
  (hB : B.2 ^ 2 = 2 * p * B.1)
  (hF : F = (p / 2, 0))
  (hl : ∃ k : ℝ, k = sqrt 3 ∧ ∀ x y, y = k * (x - p / 2) → (x, y) = A ∨ (x, y) = B)
  (hQuads : ∃ (x1 x2 : ℝ), x1 = 3 / 2 * p ∧ x2 = 1 / 6 * p):
  (dist A F / dist B F) = 3 := sorry

end parabola_line_intersection_ratio_l305_305717


namespace part_I_part_II_part_III_l305_305539

-- Define the function f(x) = e^x - a * x - 1
def f (x : ℝ) (a : ℝ) : ℝ := exp x - a * x - 1

-- Given the conditions, we prove the related properties.

-- Part I: Find the value of a and the intervals where f(x) is monotonic
theorem part_I (x : ℝ) (a : ℝ) (h : f x a = exp x - a * x - 1) : 
  (a = 2) ∧ (∀ x, x < real.log 2 -> f' x 2 < 0) ∧ (∀ x, x > real.log 2 -> f' x 2 > 0) := 
sorry

-- Part II: Prove e^x > x^2 + 1 for x > 0
theorem part_II (x : ℝ) (h : x > 0) : exp x > x^2 + 1 :=
sorry

-- Part III: Prove sum of 1/k > ln((n+1)^3 / (3e)^n) for positive n
theorem part_III (n : ℕ) (h : 0 < n) : 
  ∑ k in finset.range n, (1 : ℝ) / (k + 1) > real.log ((n+1)^3 / (3*exp 1)^n) :=
sorry

end part_I_part_II_part_III_l305_305539


namespace area_difference_l305_305734

-- Definitions of the given conditions
structure Triangle :=
(base : ℝ)
(height : ℝ)

def area (t : Triangle) : ℝ :=
  0.5 * t.base * t.height

-- Conditions of the problem
def EFG : Triangle := {base := 8, height := 4}
def EFG' : Triangle := {base := 4, height := 2}

-- Proof statement
theorem area_difference :
  area EFG - area EFG' = 12 :=
by
  sorry

end area_difference_l305_305734


namespace cherry_tomatoes_ratio_l305_305506

theorem cherry_tomatoes_ratio (T P B : ℕ) (M : ℕ := 3) (h1 : P = 4 * T) (h2 : B = 4 * P) (h3 : B / 3 = 32) :
  (T : ℚ) / M = 2 :=
by
  sorry

end cherry_tomatoes_ratio_l305_305506


namespace sarah_annual_income_l305_305732

theorem sarah_annual_income (q : ℝ) (I T : ℝ)
    (h1 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * (I - 30000)) 
    (h2 : T = 0.01 * (q + 0.5) * I) : 
    I = 36000 := by
  sorry

end sarah_annual_income_l305_305732


namespace geometric_sequence_sum_l305_305871

theorem geometric_sequence_sum {a : ℕ → ℝ} (h : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * r) 
  (h_cond : (1 / (a 2 * a 4)) + (2 / (a 4 * a 4)) + (1 / (a 4 * a 6)) = 81) :
  (1 / a 3) + (1 / a 5) = 9 :=
sorry

end geometric_sequence_sum_l305_305871


namespace sufficient_condition_for_inequality_l305_305600

theorem sufficient_condition_for_inequality (x : ℝ) : (1 - 1/x > 0) → (x > 1) :=
by
  sorry

end sufficient_condition_for_inequality_l305_305600


namespace complex_ratio_proof_l305_305131

noncomputable def complex_ratio (x y : ℂ) : ℂ :=
  ((x^6 + y^6) / (x^6 - y^6)) - ((x^6 - y^6) / (x^6 + y^6))

theorem complex_ratio_proof (x y : ℂ) (h : ((x - y) / (x + y)) - ((x + y) / (x - y)) = 2) :
  complex_ratio x y = L :=
  sorry

end complex_ratio_proof_l305_305131


namespace parabola_focus_distance_l305_305592

theorem parabola_focus_distance (M : ℝ × ℝ) (h1 : (M.2)^2 = 4 * M.1) (h2 : dist M (1, 0) = 4) : M.1 = 3 :=
sorry

end parabola_focus_distance_l305_305592


namespace problem1_solution_l305_305752

theorem problem1_solution (p : ℕ) (hp : Nat.Prime p) (a b c : ℕ) (ha : 0 < a ∧ a ≤ p) (hb : 0 < b ∧ b ≤ p) (hc : 0 < c ∧ c ≤ p)
  (f : ℕ → ℕ) (hf : ∀ x : ℕ, 0 < x → p ∣ f x) :
  (∀ x, f x = a * x^2 + b * x + c) →
  (p = 2 → a + b + c = 4) ∧ (2 < p → p % 2 = 1 → a + b + c = 3 * p) :=
by
  sorry

end problem1_solution_l305_305752


namespace intersection_point_l305_305590

theorem intersection_point (x y : ℝ) (h1 : x - 2 * y = 0) (h2 : x + y - 3 = 0) : x = 2 ∧ y = 1 :=
by
  sorry

end intersection_point_l305_305590


namespace measure_of_angle_x_l305_305227

-- Defining the conditions
def angle_ABC : ℝ := 108
def angle_ABD : ℝ := 180 - angle_ABC
def angle_in_triangle_ABD_1 : ℝ := 26
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- The theorem to prove
theorem measure_of_angle_x (h1 : angle_ABD = 72)
                           (h2 : angle_in_triangle_ABD_1 = 26)
                           (h3 : sum_of_angles_in_triangle angle_ABD angle_in_triangle_ABD_1 x) :
  x = 82 :=
by {
  -- Since this is a formal statement, we leave the proof as an exercise 
  sorry
}

end measure_of_angle_x_l305_305227


namespace part_a_part_b_l305_305537

-- Part (a)
theorem part_a {x y n : ℕ} (h : x^3 + 2^n * y = y^3 + 2^n * x) : x = y :=
sorry

-- Part (b)
theorem part_b {x y : ℤ} {n : ℕ} (h : x ≠ 0 ∧ y ≠ 0 ∧ x^3 + 2^n * y = y^3 + 2^n * x) : |x| = |y| :=
sorry

end part_a_part_b_l305_305537


namespace proof_correct_judgments_l305_305105

def terms_are_like (t1 t2 : Expr) : Prop := sorry -- Define like terms
def is_polynomial (p : Expr) : Prop := sorry -- Define polynomial
def is_quadratic_trinomial (p : Expr) : Prop := sorry -- Define quadratic trinomial
def constant_term (p : Expr) : Expr := sorry -- Define extraction of constant term

theorem proof_correct_judgments :
  let t1 := (2 * Real.pi * (a ^ 2) * b)
  let t2 := ((1 / 3) * (a ^ 2) * b)
  let p1 := (5 * a + 4 * b - 1)
  let p2 := (x - 2 * x * y + y)
  let p3 := ((x + y) / 4)
  let p4 := (x / 2 + 1)
  let p5 := (a / 4)
  terms_are_like t1 t2 ∧ 
  constant_term p1 = 1 = False ∧
  is_quadratic_trinomial p2 ∧
  is_polynomial p3 ∧ is_polynomial p4 ∧ is_polynomial p5
  → ("①③④" = "C") :=
by
  sorry

end proof_correct_judgments_l305_305105


namespace smallest_integer_greater_than_100_with_gcd_24_eq_4_l305_305479

theorem smallest_integer_greater_than_100_with_gcd_24_eq_4 :
  ∃ x : ℤ, x > 100 ∧ x % 24 = 4 ∧ (∀ y : ℤ, y > 100 ∧ y % 24 = 4 → x ≤ y) :=
sorry

end smallest_integer_greater_than_100_with_gcd_24_eq_4_l305_305479


namespace sequence_term_l305_305560

theorem sequence_term (x : ℕ → ℝ)
  (h₀ : ∀ n ≥ 2, 2 / x n = 1 / x (n - 1) + 1 / x (n + 1))
  (h₁ : x 2 = 2 / 3)
  (h₂ : x 4 = 2 / 5) :
  x 10 = 2 / 11 := 
sorry

end sequence_term_l305_305560


namespace problem_l305_305016

theorem problem (a b c : ℤ) :
  (∀ x : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b)) →
  (∀ x : ℤ, x^2 - 23 * x + 132 = (x - b) * (x - c)) →
  a + b + c = 31 :=
by
  intros h1 h2
  sorry

end problem_l305_305016


namespace prob_one_AB_stuck_prob_at_least_two_stuck_l305_305632

-- Define the events and their probabilities.
def prob_traffic_I := 1 / 10
def prob_no_traffic_I := 9 / 10
def prob_traffic_II := 3 / 5
def prob_no_traffic_II := 2 / 5

-- Define the events
def event_A := prob_traffic_I
def not_event_A := prob_no_traffic_I
def event_B := prob_traffic_I
def not_event_B := prob_no_traffic_I
def event_C := prob_traffic_II
def not_event_C := prob_no_traffic_II

-- Define the probabilities as required in the problem
def prob_exactly_one_of_A_B_in_traffic :=
  event_A * not_event_B + not_event_A * event_B

def prob_at_least_two_in_traffic :=
  event_A * event_B * not_event_C +
  event_A * not_event_B * event_C +
  not_event_A * event_B * event_C +
  event_A * event_B * event_C

-- Proofs (statements only)
theorem prob_one_AB_stuck :
  prob_exactly_one_of_A_B_in_traffic = 9 / 50 := sorry

theorem prob_at_least_two_stuck :
  prob_at_least_two_in_traffic = 59 / 500 := sorry

end prob_one_AB_stuck_prob_at_least_two_stuck_l305_305632


namespace find_some_number_l305_305251

theorem find_some_number : 
  ∃ x : ℝ, 
  (6 + 9 * 8 / x - 25 = 5) ↔ (x = 3) :=
by 
  sorry

end find_some_number_l305_305251


namespace present_population_l305_305471

theorem present_population (P : ℝ)
  (h1 : P + 0.10 * P = 242) :
  P = 220 := 
sorry

end present_population_l305_305471


namespace transformation_result_l305_305335

noncomputable def initial_function (x : ℝ) : ℝ := Real.sin (2 * x)

noncomputable def translate_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x => f (x + a)

noncomputable def compress_horizontal (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ :=
  λ x => f (k * x)

theorem transformation_result :
  (compress_horizontal (translate_left initial_function (Real.pi / 3)) 2) x = Real.sin (4 * x + (2 * Real.pi / 3)) :=
sorry

end transformation_result_l305_305335


namespace circle_radius_five_d_value_l305_305397

theorem circle_radius_five_d_value :
  ∀ (d : ℝ), (∃ (x y : ℝ), (x - 4)^2 + (y + 5)^2 = 41 - d) → d = 16 :=
by
  intros d h
  sorry

end circle_radius_five_d_value_l305_305397


namespace rectangle_hall_length_l305_305927

variable (L B : ℝ)

theorem rectangle_hall_length (h1 : B = (2 / 3) * L) (h2 : L * B = 2400) : L = 60 :=
by sorry

end rectangle_hall_length_l305_305927


namespace smallest_composite_no_prime_factors_less_than_20_l305_305678

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l305_305678


namespace product_of_consecutive_integers_sqrt_50_l305_305326

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l305_305326


namespace problem_proof_l305_305229

theorem problem_proof (x y : ℝ) (h : x / (2 * y) = 3 / 2) : (7 * x + 2 * y) / (x - 2 * y) = 23 :=
by sorry

end problem_proof_l305_305229


namespace find_grade_C_boxes_l305_305584

theorem find_grade_C_boxes (m n t : ℕ) (h : 2 * t = m + n) (total_boxes : ℕ) (h_total : total_boxes = 420) : t = 140 :=
by
  sorry

end find_grade_C_boxes_l305_305584


namespace final_score_l305_305075

def dart1 : ℕ := 50
def dart2 : ℕ := 0
def dart3 : ℕ := dart1 / 2

theorem final_score : dart1 + dart2 + dart3 = 75 := by
  sorry

end final_score_l305_305075


namespace am_gm_inequality_example_l305_305098

theorem am_gm_inequality_example (x1 x2 x3 : ℝ)
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3)
  (h_sum1 : x1 + x2 + x3 = 1) :
  (x2^2 / x1) + (x3^2 / x2) + (x1^2 / x3) ≥ 1 :=
by
  sorry

end am_gm_inequality_example_l305_305098


namespace volume_at_20_deg_l305_305528

theorem volume_at_20_deg
  (ΔV_per_ΔT : ∀ ΔT : ℕ, ΔT = 5 → ∀ V : ℕ, V = 5)
  (initial_condition : ∀ V : ℕ, V = 40 ∧ ∀ T : ℕ, T = 40) :
  ∃ V : ℕ, V = 20 :=
by
  sorry

end volume_at_20_deg_l305_305528


namespace sqrt_50_between_7_and_8_l305_305316

theorem sqrt_50_between_7_and_8 (x y : ℕ) (h1 : sqrt 50 > 7) (h2 : sqrt 50 < 8) (h3 : y = x + 1) : x * y = 56 :=
by sorry

end sqrt_50_between_7_and_8_l305_305316


namespace compute_expression_l305_305067

theorem compute_expression :
  (5 + 7)^2 + (5^2 + 7^2) * 2 = 292 := by
  sorry

end compute_expression_l305_305067


namespace alligators_hiding_correct_l305_305371

def total_alligators := 75
def not_hiding_alligators := 56

def hiding_alligators (total not_hiding : Nat) : Nat :=
  total - not_hiding

theorem alligators_hiding_correct : hiding_alligators total_alligators not_hiding_alligators = 19 := 
by
  sorry

end alligators_hiding_correct_l305_305371


namespace jason_initial_cards_l305_305125

theorem jason_initial_cards (cards_sold : Nat) (cards_after_selling : Nat) (initial_cards : Nat) 
  (h1 : cards_sold = 224) 
  (h2 : cards_after_selling = 452) 
  (h3 : initial_cards = cards_after_selling + cards_sold) : 
  initial_cards = 676 := 
sorry

end jason_initial_cards_l305_305125


namespace greatest_possible_remainder_l305_305414

theorem greatest_possible_remainder (x : ℕ) : ∃ r, r < 11 ∧ x % 11 = r ∧ r = 10 :=
by
  exists 10
  sorry

end greatest_possible_remainder_l305_305414


namespace tan_alpha_value_l305_305096

open Real

-- Define the angle alpha in the third quadrant
variable {α : ℝ}

-- Given conditions
def third_quadrant (α : ℝ) : Prop :=  π < α ∧ α < 3 * π / 2
def sin_alpha (α : ℝ) : Prop := sin α = -4 / 5

-- Statement to prove
theorem tan_alpha_value (h1 : third_quadrant α) (h2 : sin_alpha α) : tan α = 4 / 3 :=
sorry

end tan_alpha_value_l305_305096


namespace coleen_sprinkles_l305_305825

theorem coleen_sprinkles : 
  let initial_sprinkles := 12
  let remaining_sprinkles := (initial_sprinkles / 2) - 3
  remaining_sprinkles = 3 :=
by
  let initial_sprinkles := 12
  let remaining_sprinkles := (initial_sprinkles / 2) - 3
  sorry

end coleen_sprinkles_l305_305825


namespace maximum_value_l305_305445

theorem maximum_value (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) : 
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) * (z^2 - z * x + x^2)  ≤ 1 :=
sorry

end maximum_value_l305_305445


namespace sum_of_numbers_l305_305911

theorem sum_of_numbers (a b c : ℕ) (h_order: a ≤ b ∧ b ≤ c) (h_median: b = 10) 
    (h_mean_least: (a + b + c) / 3 = a + 15) (h_mean_greatest: (a + b + c) / 3 = c - 20) :
    a + b + c = 45 :=
  by
  sorry

end sum_of_numbers_l305_305911


namespace otimes_example_l305_305382

def otimes (a b : ℤ) : ℤ := a^2 - abs b

theorem otimes_example : otimes (-2) (-1) = 3 := by
  -- Define the variables
  let a := -2
  let b := -1
  -- Unfold the definition of otimes
  have h1 : otimes a b = a^2 - abs b := rfl
  -- Calculate a^2
  have h2 : a^2 = 4 := rfl
  -- Calculate abs b
  have h3 : abs b = 1 := rfl
  -- Calculate otimes a b
  show otimes a b = 3 from by
    rw [h1, h2, h3]
    rfl

end otimes_example_l305_305382


namespace train_ride_cost_difference_l305_305362

-- Definitions based on the conditions
def bus_ride_cost : ℝ := 1.40
def total_cost : ℝ := 9.65

-- Lemma to prove the mathematical question
theorem train_ride_cost_difference :
  ∃ T : ℝ, T + bus_ride_cost = total_cost ∧ (T - bus_ride_cost) = 6.85 :=
by
  sorry

end train_ride_cost_difference_l305_305362


namespace negation_proposition_l305_305019

theorem negation_proposition :
  (∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) :=
by sorry

end negation_proposition_l305_305019


namespace calculate_expression_l305_305508

theorem calculate_expression : (1100 * 1100) / ((260 * 260) - (240 * 240)) = 121 := by
  sorry

end calculate_expression_l305_305508


namespace max_triangle_side_l305_305994

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l305_305994


namespace max_sum_mult_table_l305_305163

def isEven (n : ℕ) : Prop := n % 2 = 0
def isOdd (n : ℕ) : Prop := ¬ isEven n
def entries : List ℕ := [3, 4, 6, 8, 9, 12]
def sumOfList (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem max_sum_mult_table :
  ∃ (a b c d e f : ℕ), 
    a ∈ entries ∧ b ∈ entries ∧ c ∈ entries ∧ 
    d ∈ entries ∧ e ∈ entries ∧ f ∈ entries ∧ 
    (isEven a ∧ isEven b ∧ isOdd c ∨ isEven a ∧ isOdd b ∧ isOdd c ∨ isOdd a ∧ isEven b ∧ isEven c ∨ isOdd a ∧ isOdd b ∧ isOdd c ∨ isOdd a ∧ isEven b ∧ isOdd c ∨ isEven a ∧ isOdd b ∧ isEven c) ∧ 
    (isEven d ∧ isEven e ∧ isOdd f ∨ isEven d ∧ isOdd e ∧ isOdd f ∨ isOdd d ∧ isEven e ∧ isEven f ∨ isOdd d ∧ isOdd e ∧ isOdd f ∨ isOdd d ∧ isEven e ∧ isOdd f ∨ isEven d ∧ isOdd e ∧ isEven f) ∧ 
    (sumOfList [a, b, c] * sumOfList [d, e, f] = 425) := 
by
    sorry  -- Skipping the proof as instructed.

end max_sum_mult_table_l305_305163


namespace exists_pairs_with_equal_sums_and_product_difference_l305_305142

theorem exists_pairs_with_equal_sums_and_product_difference (N : ℕ) :
  ∃ a1 b1 a2 b2 : ℕ, a1 + b1 = a2 + b2 ∧ (a2 * b2 - a1 * b1 = N) :=
begin
  -- Skipping the proof body, as it’s not required.
  sorry,
end

end exists_pairs_with_equal_sums_and_product_difference_l305_305142


namespace original_water_amount_l305_305353

theorem original_water_amount (W : ℝ) 
    (evap_rate : ℝ := 0.03) 
    (days : ℕ := 22) 
    (evap_percent : ℝ := 0.055) 
    (total_evap : ℝ := evap_rate * days) 
    (evap_condition : evap_percent * W = total_evap) : W = 12 :=
by sorry

end original_water_amount_l305_305353


namespace solution_set_x_l305_305045

theorem solution_set_x (x : ℝ) (h₁ : 33 * 32 ≤ x)
  (h₂ : ⌊x⌋ + ⌈x⌉ = 5) : 2 < x ∧ x < 3 :=
by
  sorry

end solution_set_x_l305_305045


namespace reeyas_first_subject_score_l305_305761

theorem reeyas_first_subject_score
  (second_subject_score third_subject_score fourth_subject_score : ℕ)
  (num_subjects : ℕ)
  (average_score : ℕ)
  (total_subjects_score : ℕ)
  (condition1 : second_subject_score = 76)
  (condition2 : third_subject_score = 82)
  (condition3 : fourth_subject_score = 85)
  (condition4 : num_subjects = 4)
  (condition5 : average_score = 75)
  (condition6 : total_subjects_score = num_subjects * average_score) :
  67 = total_subjects_score - (second_subject_score + third_subject_score + fourth_subject_score) := 
  sorry

end reeyas_first_subject_score_l305_305761


namespace find_max_side_length_l305_305959

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l305_305959


namespace total_hats_purchased_l305_305787

theorem total_hats_purchased (B G : ℕ) (h1 : G = 38) (h2 : 6 * B + 7 * G = 548) : B + G = 85 := 
by 
  sorry

end total_hats_purchased_l305_305787


namespace domain_of_sqrt_tan_l305_305396

theorem domain_of_sqrt_tan :
  ∀ x : ℝ, (∃ k : ℤ, k * π ≤ x ∧ x < k * π + π / 2) ↔ 0 ≤ (Real.tan x) :=
sorry

end domain_of_sqrt_tan_l305_305396


namespace medians_sum_le_circumradius_l305_305866

-- Definition of the problem
variable (a b c R : ℝ) (m_a m_b m_c : ℝ)

-- Conditions: medians of triangle ABC, and R is the circumradius
def is_median (m : ℝ) (a b c : ℝ) : Prop :=
  m^2 = (2*b^2 + 2*c^2 - a^2) / 4

-- Main theorem to prove
theorem medians_sum_le_circumradius (h_ma : is_median m_a a b c)
  (h_mb : is_median m_b b a c) (h_mc : is_median m_c c a b) 
  (h_R : a^2 + b^2 + c^2 ≤ 9 * R^2) :
  m_a + m_b + m_c ≤ 9 / 2 * R :=
sorry

end medians_sum_le_circumradius_l305_305866


namespace cos_B_eq_find_b_eq_l305_305872

variable (A B C a b c : ℝ)

-- Given conditions
axiom sin_A_plus_C_eq : Real.sin (A + C) = 8 * Real.sin (B / 2) ^ 2
axiom a_plus_c : a + c = 6
axiom area_of_triangle : 1 / 2 * a * c * Real.sin B = 2

-- Proving cos B
theorem cos_B_eq :
  Real.cos B = 15 / 17 :=
sorry

-- Proving b given the area and sides condition
theorem find_b_eq :
  Real.cos B = 15 / 17 → b = 2 :=
sorry

end cos_B_eq_find_b_eq_l305_305872


namespace ice_forms_inner_surface_in_winter_l305_305607

-- Definitions based on conditions
variable (humid_air_inside : Prop) 
variable (heat_transfer_inner_surface : Prop) 
variable (heat_transfer_outer_surface : Prop) 
variable (temp_inner_surface_below_freezing : Prop) 
variable (condensation_inner_surface_below_freezing : Prop)
variable (ice_formation_inner_surface : Prop)
variable (cold_dry_air_outside : Prop)
variable (no_significant_condensation_outside : Prop)

-- Proof of the theorem
theorem ice_forms_inner_surface_in_winter :
  humid_air_inside ∧
  heat_transfer_inner_surface ∧
  heat_transfer_outer_surface ∧
  (¬sufficient_heating → temp_inner_surface_below_freezing) ∧
  (condensation_inner_surface_below_freezing ↔ (temp_inner_surface_below_freezing ∧ humid_air_inside)) ∧
  (ice_formation_inner_surface ↔ (condensation_inner_surface_below_freezing ∧ temp_inner_surface_below_freezing)) ∧
  (cold_dry_air_outside → ¬ice_formation_outer_surface)
  → ice_formation_inner_surface :=
sorry

end ice_forms_inner_surface_in_winter_l305_305607


namespace steve_bought_3_boxes_of_cookies_l305_305153

variable (total_cost : ℝ)
variable (milk_cost : ℝ)
variable (cereal_cost : ℝ)
variable (banana_cost : ℝ)
variable (apple_cost : ℝ)
variable (chicken_cost : ℝ)
variable (peanut_butter_cost : ℝ)
variable (bread_cost : ℝ)
variable (cookie_box_cost : ℝ)
variable (cookie_box_count : ℝ)

noncomputable def proves_steve_cookie_boxes : Prop :=
  total_cost = 50 ∧
  milk_cost = 4 ∧
  cereal_cost = 3 ∧
  banana_cost = 0.2 ∧
  apple_cost = 0.75 ∧
  chicken_cost = 10 ∧
  peanut_butter_cost = 5 ∧
  bread_cost = (2 * cereal_cost) / 2 ∧
  cookie_box_cost = (milk_cost + peanut_butter_cost) / 3 ∧
  cookie_box_count = (total_cost - (milk_cost + 3 * cereal_cost + 6 * banana_cost + 8 * apple_cost + chicken_cost + peanut_butter_cost + bread_cost)) / cookie_box_cost

theorem steve_bought_3_boxes_of_cookies :
  proves_steve_cookie_boxes 50 4 3 0.2 0.75 10 5 3 ((4 + 5) / 3) 3 :=
by
  sorry

end steve_bought_3_boxes_of_cookies_l305_305153


namespace average_income_Q_and_R_l305_305015

variable (P Q R: ℝ)

theorem average_income_Q_and_R:
  (P + Q) / 2 = 5050 →
  (P + R) / 2 = 5200 →
  P = 4000 →
  (Q + R) / 2 = 6250 :=
by
  sorry

end average_income_Q_and_R_l305_305015


namespace avg_people_per_hour_rounding_l305_305558

theorem avg_people_per_hour_rounding :
  let people := 3500
  let days := 5
  let hours := days * 24
  (people / hours : ℚ).round = 29 := 
by
  sorry

end avg_people_per_hour_rounding_l305_305558


namespace product_third_side_approximation_l305_305475

def triangle_third_side (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

noncomputable def product_of_third_side_lengths : ℝ :=
  Real.sqrt 41 * 3

theorem product_third_side_approximation (a b : ℝ) (h₁ : a = 4) (h₂ : b = 5) :
  ∃ (c₁ c₂ : ℝ), triangle_third_side a b c₁ ∧ triangle_third_side a b c₂ ∧
  abs ((c₁ * c₂) - 19.2) < 0.1 :=
sorry

end product_third_side_approximation_l305_305475


namespace train_speed_is_40_kmh_l305_305059

noncomputable def speed_of_train (train_length_m : ℝ) 
                                   (man_speed_kmh : ℝ) 
                                   (pass_time_s : ℝ) : ℝ :=
  let train_length_km := train_length_m / 1000
  let pass_time_h := pass_time_s / 3600
  let relative_speed_kmh := train_length_km / pass_time_h
  relative_speed_kmh - man_speed_kmh
  
theorem train_speed_is_40_kmh :
  speed_of_train 110 4 9 = 40 := 
by
  sorry

end train_speed_is_40_kmh_l305_305059


namespace glasses_total_l305_305342

theorem glasses_total :
  ∃ (S L e : ℕ), 
    (L = S + 16) ∧ 
    (12 * S + 16 * L) / (S + L) = 15 ∧ 
    (e = 12 * S + 16 * L) ∧ 
    e = 480 :=
by
  sorry

end glasses_total_l305_305342


namespace students_present_l305_305175

theorem students_present (total_students : ℕ) (absent_percent : ℝ) (total_absent : ℝ) (total_present : ℝ) :
  total_students = 50 → absent_percent = 0.12 → total_absent = total_students * absent_percent →
  total_present = total_students - total_absent →
  total_present = 44 :=
by
  intros _ _ _ _; sorry

end students_present_l305_305175


namespace theresa_hours_l305_305782

theorem theresa_hours (h1 h2 h3 h4 h5 h6 : ℕ) (avg : ℕ) (x : ℕ) 
  (H_cond : h1 = 10 ∧ h2 = 8 ∧ h3 = 9 ∧ h4 = 11 ∧ h5 = 6 ∧ h6 = 8)
  (H_avg : avg = 9) : 
  (h1 + h2 + h3 + h4 + h5 + h6 + x) / 7 = avg ↔ x = 11 :=
by
  sorry

end theresa_hours_l305_305782


namespace max_rectangle_area_l305_305828

theorem max_rectangle_area (l w : ℕ) (h : 3 * l + 5 * w ≤ 50) : (l * w ≤ 35) :=
by sorry

end max_rectangle_area_l305_305828


namespace basic_astrophysics_degrees_l305_305193

-- Define the percentages for various sectors
def microphotonics := 14
def home_electronics := 24
def food_additives := 15
def genetically_modified_microorganisms := 19
def industrial_lubricants := 8

-- The sum of the given percentages
def total_other_percentages := 
    microphotonics + home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants

-- The remaining percentage for basic astrophysics
def basic_astrophysics_percentage := 100 - total_other_percentages

-- Number of degrees in a full circle
def full_circle_degrees := 360

-- Calculate the degrees representing basic astrophysics
def degrees_for_basic_astrophysics := (basic_astrophysics_percentage * full_circle_degrees) / 100

-- Theorem statement
theorem basic_astrophysics_degrees : degrees_for_basic_astrophysics = 72 := 
by
  sorry

end basic_astrophysics_degrees_l305_305193


namespace max_value_g_l305_305513

-- Defining the conditions and goal as functions and properties
def condition_1 (f : ℕ → ℕ) : Prop :=
  (Finset.range 43).sum f ≤ 2022

def condition_2 (f g : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a >= b → g (a + b) ≤ f a + f b

-- Defining the main theorem to establish the maximum value
theorem max_value_g (f g : ℕ → ℕ) (h1 : condition_1 f) (h2 : condition_2 f g) :
  (Finset.range 85).sum g ≤ 7615 :=
sorry


end max_value_g_l305_305513


namespace find_other_number_l305_305024

-- Given conditions
def sum_of_numbers (x y : ℕ) : Prop := x + y = 72
def number_difference (x y : ℕ) : Prop := x = y + 12
def one_number_is_30 (x : ℕ) : Prop := x = 30

-- Theorem to prove
theorem find_other_number (y : ℕ) : 
  sum_of_numbers y 30 ∧ number_difference 30 y → y = 18 := by
  sorry

end find_other_number_l305_305024


namespace minimum_positive_difference_contains_amounts_of_numbers_on_strips_l305_305889

theorem minimum_positive_difference_contains_amounts_of_numbers_on_strips (a b c d e f : ℕ) 
  (h1 : a + f = 7) (h2 : b + e = 7) (h3 : c + d = 7) :
  ∃ (min_diff : ℕ), min_diff = 1 :=
by {
  -- The problem guarantees the minimum difference given the conditions.
  sorry
}

end minimum_positive_difference_contains_amounts_of_numbers_on_strips_l305_305889


namespace number_of_distinct_bad_arrangements_l305_305296

def is_bad_arrangement (l : List ℕ) : Prop :=
  l.perm [1, 2, 3, 4, 6] ∧
  ∀ n : ℕ, n > 0 ∧ n < 17 → 
    ¬ (∃ (a b : ℕ), a < b ∧ (finset.range (b - a) + a) = (finset.of_list l).sum (finset.range (b - a) + a))

def distinct_bad_arrangements : Finset (List ℕ) :=
  (List.permutations [1, 2, 3, 4, 6]).to_finset.filter is_bad_arrangement

theorem number_of_distinct_bad_arrangements : distinct_bad_arrangements.card = 2 := by
  sorry

end number_of_distinct_bad_arrangements_l305_305296


namespace annies_classmates_count_l305_305505

theorem annies_classmates_count (spent : ℝ) (cost_per_candy : ℝ) (candies_left : ℕ) (candies_per_classmate : ℕ) (expected_classmates : ℕ):
  spent = 8 ∧ cost_per_candy = 0.1 ∧ candies_left = 12 ∧ candies_per_classmate = 2 ∧ expected_classmates = 34 →
  (spent / cost_per_candy) - candies_left = (expected_classmates * candies_per_classmate) := 
by
  intros h
  sorry

end annies_classmates_count_l305_305505


namespace car_meeting_points_l305_305027

-- Define the conditions for the problem
variables {A B : ℝ}
variables {speed_ratio : ℝ} (ratio_pos : speed_ratio = 5 / 4)
variables {T1 T2 : ℝ} (T1_pos : T1 = 145) (T2_pos : T2 = 201)

-- The proof problem statement
theorem car_meeting_points (A B : ℝ) (ratio_pos : speed_ratio = 5 / 4) 
  (T1 T2 : ℝ) (T1_pos : T1 = 145) (T2_pos : T2 = 201) :
  A = 103 ∧ B = 229 :=
sorry

end car_meeting_points_l305_305027


namespace volume_of_stone_l305_305780

def width := 16
def length := 14
def full_height := 9
def initial_water_height := 4
def final_water_height := 9

def volume_before := length * width * initial_water_height
def volume_after := length * width * final_water_height

def volume_stone := volume_after - volume_before

theorem volume_of_stone : volume_stone = 1120 := by
  unfold volume_stone
  unfold volume_after volume_before
  unfold final_water_height initial_water_height width length
  sorry

end volume_of_stone_l305_305780


namespace smallest_perfect_cube_divisor_l305_305271

theorem smallest_perfect_cube_divisor (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (h : p ≠ q ∧ p ≠ r ∧ q ≠ r) :
  ∃ (a b c : ℕ), a = 6 ∧ b = 6 ∧ c = 6 ∧ (p^a * q^b * r^c) = (p^2 * q^2 * r^2)^3 ∧ 
  (p^a * q^b * r^c) % (p^2 * q^3 * r^4) = 0 := 
by
  sorry

end smallest_perfect_cube_divisor_l305_305271


namespace rectangle_area_error_83_percent_l305_305554

theorem rectangle_area_error_83_percent (L W : ℝ) :
  let actual_area := L * W
  let measured_length := 1.14 * L
  let measured_width := 0.95 * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  (error / actual_area) * 100 = 8.3 := by
  sorry

end rectangle_area_error_83_percent_l305_305554


namespace algebraic_expression_value_l305_305852

theorem algebraic_expression_value (x : ℝ) (h : 3 / (x^2 + x) - x^2 = 2 + x) :
  2 * x^2 + 2 * x = 2 :=
sorry

end algebraic_expression_value_l305_305852


namespace club_population_after_five_years_l305_305206

noncomputable def a : ℕ → ℕ
| 0     => 18
| (n+1) => 3 * (a n - 5) + 5

theorem club_population_after_five_years : a 5 = 3164 := by
  sorry

end club_population_after_five_years_l305_305206


namespace ferry_journey_difference_l305_305798

theorem ferry_journey_difference
  (time_P : ℝ) (speed_P : ℝ) (mult_Q : ℝ) (speed_diff : ℝ)
  (dist_P : ℝ := time_P * speed_P)
  (dist_Q : ℝ := mult_Q * dist_P)
  (speed_Q : ℝ := speed_P + speed_diff)
  (time_Q : ℝ := dist_Q / speed_Q) :
  time_P = 3 ∧ speed_P = 6 ∧ mult_Q = 3 ∧ speed_diff = 3 → time_Q - time_P = 3 := by
  sorry

end ferry_journey_difference_l305_305798


namespace minimum_dimes_to_afford_sneakers_l305_305376

-- Define constants and conditions using Lean
def sneaker_cost : ℝ := 45.35
def ten_dollar_bills_count : ℕ := 3
def quarter_count : ℕ := 4
def dime_value : ℝ := 0.1
def quarter_value : ℝ := 0.25
def ten_dollar_bill_value : ℝ := 10.0

-- Define a function to calculate the total amount based on the number of dimes
def total_amount (dimes : ℕ) : ℝ :=
  (ten_dollar_bills_count * ten_dollar_bill_value) +
  (quarter_count * quarter_value) +
  (dimes * dime_value)

-- The main theorem to be proven
theorem minimum_dimes_to_afford_sneakers (n : ℕ) : total_amount n ≥ sneaker_cost ↔ n ≥ 144 :=
by
  sorry

end minimum_dimes_to_afford_sneakers_l305_305376


namespace product_of_consecutive_integers_sqrt_50_l305_305304

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), 49 < 50 ∧ 50 < 64 ∧ n = m + 1 ∧ m * n = 56 :=
by {
  let m := 7,
  let n := 8,
  have h1 : 49 < 50 := by norm_num,
  have h2 : 50 < 64 := by norm_num,
  exact ⟨m, n, h1, h2, rfl, by norm_num⟩,
  sorry -- proof skipped
}

end product_of_consecutive_integers_sqrt_50_l305_305304


namespace tetrahedron_dihedral_face_areas_l305_305566

variables {S₁ S₂ a b : ℝ} {α φ : ℝ}

theorem tetrahedron_dihedral_face_areas :
  S₁^2 + S₂^2 - 2 * S₁ * S₂ * Real.cos α = (a * b * Real.sin φ / 4)^2 :=
sorry

end tetrahedron_dihedral_face_areas_l305_305566


namespace find_b_plus_m_l305_305418

section MatrixPower

open Matrix

-- Define our matrices
def A (b m : ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![1, 3, b], 
    ![0, 1, 5], 
    ![0, 0, 1]]

def B : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![1, 27, 3008], 
    ![0, 1, 45], 
    ![0, 0, 1]]

-- The problem statement
noncomputable def power_eq_matrix (b m : ℕ) : Prop :=
  (A b m) ^ m = B

-- The final goal
theorem find_b_plus_m (b m : ℕ) (h : power_eq_matrix b m) : b + m = 283 := sorry

end MatrixPower

end find_b_plus_m_l305_305418


namespace original_count_l305_305867

-- Conditions
def original_count_eq (ping_pong_balls shuttlecocks : ℕ) : Prop :=
  ping_pong_balls = shuttlecocks

def removal_count (x : ℕ) : Prop :=
  5 * x - 3 * x = 16

-- Theorem to prove the original number of ping-pong balls and shuttlecocks
theorem original_count (ping_pong_balls shuttlecocks : ℕ) (x : ℕ) (h1 : original_count_eq ping_pong_balls shuttlecocks) (h2 : removal_count x) : ping_pong_balls = 40 ∧ shuttlecocks = 40 :=
  sorry

end original_count_l305_305867


namespace office_assignment_l305_305492

/-- Assign four people to clean three offices, with at least one person assigned to each office. 
There are exactly 36 different possible assignments. -/
theorem office_assignment : 
  let people := {1, 2, 3, 4}
  let offices := {A, B, C}
  ∃ (f : people → offices), (∀ o ∈ offices, ∃ p ∈ people, f p = o) ∧ (f.range.card = 3) → 
  (36 = 6 * 3 * 2) :=
by 
  let people := {1, 2, 3, 4}
  let offices := {A, B, C}
  have h_num_ways : (36 = 6 * 3 * 2) := by sorry
  exact ⟨f, fun H => sorry, rfl, h_num_ways⟩

end office_assignment_l305_305492


namespace max_side_of_triangle_exists_max_side_of_elevent_l305_305974

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l305_305974


namespace vector_addition_simplification_l305_305764

variable {V : Type*} [AddCommGroup V]

theorem vector_addition_simplification
  (AB BC AC DC CD : V)
  (h1 : AB + BC = AC)
  (h2 : - DC = CD) :
  AB + BC - AC - DC = CD :=
by
  -- Placeholder for the proof
  sorry

end vector_addition_simplification_l305_305764


namespace final_score_proof_l305_305072

def final_score (bullseye_points : ℕ) (miss_points : ℕ) (half_bullseye_points : ℕ) : ℕ :=
  bullseye_points + miss_points + half_bullseye_points

theorem final_score_proof : final_score 50 0 25 = 75 :=
by
  -- Considering the given conditions
  -- bullseye_points = 50
  -- miss_points = 0
  -- half_bullseye_points = half of bullseye_points = 25
  -- Summing them up: 50 + 0 + 25 = 75
  sorry

end final_score_proof_l305_305072


namespace each_person_eats_3_Smores_l305_305006

-- Definitions based on the conditions in (a)
def people := 8
def cost_per_4_Smores := 3
def total_cost := 18

-- The statement we need to prove
theorem each_person_eats_3_Smores (h1 : total_cost = people * (cost_per_4_Smores * 4 / 3)) :
  (total_cost / cost_per_4_Smores) * 4 / people = 3 :=
by
  sorry

end each_person_eats_3_Smores_l305_305006


namespace quadrilateral_area_is_11_l305_305550

def point := (ℤ × ℤ)

def A : point := (0, 0)
def B : point := (1, 4)
def C : point := (4, 3)
def D : point := (3, 0)

def area_of_quadrilateral (p1 p2 p3 p4 : point) : ℤ :=
  let ⟨x1, y1⟩ := p1
  let ⟨x2, y2⟩ := p2
  let ⟨x3, y3⟩ := p3
  let ⟨x4, y4⟩ := p4
  (|x1*y2 - y1*x2 + x2*y3 - y2*x3 + x3*y4 - y3*x4 + x4*y1 - y4*x1|) / 2

theorem quadrilateral_area_is_11 : area_of_quadrilateral A B C D = 11 := by 
  sorry

end quadrilateral_area_is_11_l305_305550


namespace part1_part2_l305_305711

noncomputable def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1 (a : ℝ) (h : a = 2) :
  ∀ x : ℝ, f x a ≥ 4 ↔ x ≤ (3 / 2 : ℝ) ∨ x ≥ (11 / 2 : ℝ) :=
by 
  rw h
  sorry

theorem part2 (h : ∀ x a : ℝ, f x a ≥ 4) :
  ∀ a : ℝ, (a - 1)^2 ≥ 4 ↔ a ≤ -1 ∨ a ≥ 3 :=
by 
  sorry

end part1_part2_l305_305711


namespace puzzle_pieces_count_l305_305610

variable (border_pieces : ℕ) (trevor_pieces : ℕ) (joe_pieces : ℕ) (missing_pieces : ℕ)

def total_puzzle_pieces (border_pieces trevor_pieces joe_pieces missing_pieces : ℕ) : ℕ :=
  border_pieces + trevor_pieces + joe_pieces + missing_pieces

theorem puzzle_pieces_count :
  border_pieces = 75 → 
  trevor_pieces = 105 → 
  joe_pieces = 3 * trevor_pieces → 
  missing_pieces = 5 → 
  total_puzzle_pieces border_pieces trevor_pieces joe_pieces missing_pieces = 500 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  -- proof step to get total_number_pieces = 75 + 105 + (3 * 105) + 5
  -- hence total_puzzle_pieces = 500
  sorry

end puzzle_pieces_count_l305_305610


namespace coastal_city_spending_l305_305159

def beginning_of_may_spending : ℝ := 1.2
def end_of_september_spending : ℝ := 4.5

theorem coastal_city_spending :
  (end_of_september_spending - beginning_of_may_spending) = 3.3 :=
by
  -- Proof can be filled in here
  sorry

end coastal_city_spending_l305_305159


namespace product_of_consecutive_integers_sqrt_50_l305_305331

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l305_305331


namespace mrs_lovely_class_l305_305002

-- Define the number of students in Mrs. Lovely's class
def number_of_students (g b : ℕ) : ℕ := g + b

theorem mrs_lovely_class (g b : ℕ): 
  (b = g + 3) →
  (500 - 10 = g * g + b * b) →
  number_of_students g b = 23 :=
by
  sorry

end mrs_lovely_class_l305_305002


namespace rational_square_l305_305577

theorem rational_square (a b c : ℚ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) : ∃ r : ℚ, (1 / (a - b)^2) + (1 / (b - c)^2) + (1 / (c - a)^2) = r^2 := 
by 
  sorry

end rational_square_l305_305577


namespace hexagon_cyclic_identity_l305_305556

variables (a a' b b' c c' a₁ b₁ c₁ : ℝ)

theorem hexagon_cyclic_identity :
  a₁ * b₁ * c₁ = a * b * c + a' * b' * c' + a * a' * a₁ + b * b' * b₁ + c * c' * c₁ :=
by
  sorry

end hexagon_cyclic_identity_l305_305556


namespace part1_part2_l305_305716

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2*a + 1)

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
by sorry

theorem part2 (a : ℝ) : (∀ x, f x a ≥ 4) ↔ (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l305_305716


namespace point_on_number_line_l305_305470

theorem point_on_number_line (a : ℤ) (h : abs (a + 3) = 4) : a = 1 ∨ a = -7 := 
sorry

end point_on_number_line_l305_305470


namespace exists_common_point_in_intervals_l305_305747

theorem exists_common_point_in_intervals
  (n : ℕ)
  (a b : Fin n → ℝ)
  (h : ∀ i j : Fin n, ∃ x : ℝ, a i ≤ x ∧ x ≤ b i ∧ a j ≤ x ∧ x ≤ b j) :
  ∃ p : ℝ, ∀ i : Fin n, a i ≤ p ∧ p ≤ b i :=
sorry

end exists_common_point_in_intervals_l305_305747


namespace f_2015_l305_305534

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f (x + 2) = f (2 - x) + 4 * f 2
axiom symmetric_about_neg1 : ∀ x : ℝ, f (x + 1) = f (-2 - (x + 1))
axiom f_at_1 : f 1 = 3

theorem f_2015 : f 2015 = -3 :=
by
  apply sorry

end f_2015_l305_305534


namespace trig_function_properties_l305_305596

theorem trig_function_properties :
  ∀ x : ℝ, 
    (1 - 2 * (Real.sin (x - π / 4))^2) = Real.sin (2 * x) ∧ 
    (∀ x : ℝ, Real.sin (2 * (-x)) = -Real.sin (2 * x)) ∧ 
    2 * π / 2 = π :=
by
  sorry

end trig_function_properties_l305_305596


namespace smallest_composite_no_prime_factors_less_than_twenty_l305_305671

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l305_305671


namespace reciprocal_neg_half_l305_305170

theorem reciprocal_neg_half : 1 / (-1 / 2 : ℝ) = (-2 : ℝ) :=
by
  sorry

end reciprocal_neg_half_l305_305170


namespace train_length_l305_305642

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (bridge_length_m : ℕ) (conversion_factor : ℝ) :
  speed_kmh = 54 →
  time_s = 33333333333333336 / 1000000000000000 →
  bridge_length_m = 140 →
  conversion_factor = 1000 / 3600 →
  ∃ (train_length_m : ℝ), 
    speed_kmh * conversion_factor * time_s + bridge_length_m = train_length_m + bridge_length_m :=
by
  intros
  use 360
  sorry

end train_length_l305_305642


namespace olivia_initial_money_l305_305003

theorem olivia_initial_money (spent_supermarket : ℕ) (spent_showroom : ℕ) (left_money : ℕ) (initial_money : ℕ) :
  spent_supermarket = 31 → spent_showroom = 49 → left_money = 26 → initial_money = spent_supermarket + spent_showroom + left_money → initial_money = 106 :=
by
  intros h_supermarket h_showroom h_left h_initial 
  rw [h_supermarket, h_showroom, h_left] at h_initial
  exact h_initial

end olivia_initial_money_l305_305003


namespace at_least_one_not_less_than_two_l305_305446

theorem at_least_one_not_less_than_two (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + 1/y ≥ 2) ∨ (y + 1/z ≥ 2) ∨ (z + 1/x ≥ 2) := sorry

end at_least_one_not_less_than_two_l305_305446


namespace probability_six_highest_two_selected_l305_305050

noncomputable def calcProbability : ℚ :=
  let total_ways := Nat.choose 7 4 in
  let favorable_ways := Nat.choose 5 3 in
  (3 / 7) * (favorable_ways / total_ways)

theorem probability_six_highest_two_selected :
  calcProbability = 6 / 49 :=
by
  -- This is just a statement of the problem; the proof is omitted.
  sorry

end probability_six_highest_two_selected_l305_305050


namespace max_x_plus_2y_l305_305878

theorem max_x_plus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x + 2 * y ≤ 3 :=
sorry

end max_x_plus_2y_l305_305878


namespace number_of_ways_to_choose_one_book_l305_305757

theorem number_of_ways_to_choose_one_book:
  let chinese_books := 10
  let english_books := 7
  let mathematics_books := 5
  chinese_books + english_books + mathematics_books = 22 := by
    -- The actual proof should go here.
    sorry

end number_of_ways_to_choose_one_book_l305_305757


namespace imaginary_number_condition_fourth_quadrant_condition_l305_305840

-- Part 1: Prove that if \( z \) is purely imaginary, then \( m = 0 \)
theorem imaginary_number_condition (m : ℝ) :
  (m * (m + 2) = 0) ∧ (m^2 + m - 2 ≠ 0) → m = 0 :=
by
  sorry

-- Part 2: Prove that if \( z \) is in the fourth quadrant, then \( 0 < m < 1 \)
theorem fourth_quadrant_condition (m : ℝ) :
  (m * (m + 2) > 0) ∧ (m^2 + m - 2 < 0) → (0 < m ∧ m < 1) :=
by
  sorry

end imaginary_number_condition_fourth_quadrant_condition_l305_305840


namespace simple_interest_years_l305_305501

variable (P R T : ℕ)
variable (deltaI : ℕ := 400)
variable (P_value : P = 800)

theorem simple_interest_years 
  (h : (800 * (R + 5) * T / 100) = (800 * R * T / 100) + 400) :
  T = 10 :=
by sorry

end simple_interest_years_l305_305501


namespace ball_bounce_height_l305_305630

theorem ball_bounce_height :
  ∃ k : ℕ, (20 * (3 / 4 : ℝ)^k < 2) ∧ ∀ n < k, ¬ (20 * (3 / 4 : ℝ)^n < 2) :=
sorry

end ball_bounce_height_l305_305630


namespace willie_stickers_l305_305795

theorem willie_stickers (initial_stickers : ℕ) (given_stickers : ℕ) (remaining_stickers : ℕ) :
  initial_stickers = 124 → given_stickers = 23 → remaining_stickers = initial_stickers - given_stickers → remaining_stickers = 101 :=
by
  intros h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining.trans rfl

end willie_stickers_l305_305795


namespace ice_cream_melt_time_l305_305739

theorem ice_cream_melt_time :
  let blocks := 16
  let block_length := 1.0/8.0 -- miles per block
  let distance := blocks * block_length -- in miles
  let speed := 12.0 -- miles per hour
  let time := distance / speed -- in hours
  let time_in_minutes := time * 60 -- converted to minutes
  time_in_minutes = 10 := by sorry

end ice_cream_melt_time_l305_305739


namespace first_day_revenue_l305_305583

theorem first_day_revenue :
  ∀ (S : ℕ), (12 * S + 90 = 246) → (4 * S + 3 * 9 = 79) :=
by
  intros S h1
  sorry

end first_day_revenue_l305_305583


namespace gratuities_correct_l305_305639

def cost_of_striploin : ℝ := 80
def cost_of_wine : ℝ := 10
def sales_tax_rate : ℝ := 0.10
def total_bill_with_gratuities : ℝ := 140

def total_bill_before_tax : ℝ := cost_of_striploin + cost_of_wine := by sorry

def sales_tax : ℝ := sales_tax_rate * total_bill_before_tax := by sorry

def total_bill_with_tax : ℝ := total_bill_before_tax + sales_tax := by sorry

def gratuities : ℝ := total_bill_with_gratuities - total_bill_with_tax := by sorry

theorem gratuities_correct : gratuities = 41 := by sorry

end gratuities_correct_l305_305639


namespace reciprocal_of_minus_one_half_l305_305167

theorem reciprocal_of_minus_one_half : (1 / (-1 / 2)) = -2 := 
by sorry

end reciprocal_of_minus_one_half_l305_305167


namespace product_of_integers_around_sqrt_50_l305_305311

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l305_305311


namespace certain_number_l305_305340

theorem certain_number (n : ℕ) : 
  (55 * 57) % n = 6 ∧ n = 1043 :=
by
  sorry

end certain_number_l305_305340


namespace real_part_of_product_l305_305845

open Complex

theorem real_part_of_product (α β : ℝ) :
  let z1 := Complex.mk (Real.cos α) (Real.sin α)
  let z2 := Complex.mk (Real.cos β) (Real.sin β)
  Complex.re (z1 * z2) = Real.cos (α + β) :=
by
  let z1 := Complex.mk (Real.cos α) (Real.sin α)
  let z2 := Complex.mk (Real.cos β) (Real.sin β)
  sorry

end real_part_of_product_l305_305845


namespace carrots_total_l305_305802

theorem carrots_total 
  (picked_1 : Nat) 
  (thrown_out : Nat) 
  (picked_2 : Nat) 
  (total_carrots : Nat) 
  (h_picked1 : picked_1 = 23) 
  (h_thrown_out : thrown_out = 10) 
  (h_picked2 : picked_2 = 47) : 
  total_carrots = 60 := 
by
  sorry

end carrots_total_l305_305802


namespace smallest_composite_no_prime_factors_less_than_20_l305_305681

def smallest_composite_no_prime_factors_less_than (n : ℕ) (k : ℕ) : ℕ :=
  if h1 : k > 1 ∧ ∀ p : ℕ, p.prime → p ∣ k → p ≥ n then k else 0

theorem smallest_composite_no_prime_factors_less_than_20 : smallest_composite_no_prime_factors_less_than 20 529 = 529 := by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l305_305681


namespace worksheets_graded_l305_305060

theorem worksheets_graded (w : ℕ) (h1 : ∀ (n : ℕ), n = 3) (h2 : ∀ (n : ℕ), n = 15) (h3 : ∀ (p : ℕ), p = 24)  :
  w = 7 :=
sorry

end worksheets_graded_l305_305060


namespace inequality_solution_l305_305909

theorem inequality_solution (x y : ℝ) : y - x < abs x ↔ y < 0 ∨ y < 2 * x :=
by sorry

end inequality_solution_l305_305909


namespace conditional_probability_l305_305217

variable (pA pB pAB : ℝ)
variable (h1 : pA = 0.2)
variable (h2 : pB = 0.18)
variable (h3 : pAB = 0.12)

theorem conditional_probability : (pAB / pB = 2 / 3) :=
by
  -- sorry is used to skip the proof
  sorry

end conditional_probability_l305_305217


namespace amanda_bought_30_candy_bars_l305_305219

noncomputable def candy_bars_bought (c1 c2 c3 c4 : ℕ) : ℕ :=
  let c5 := c4 * c2
  let c6 := c3 - c2
  let c7 := (c6 + c5) - c1
  c7

theorem amanda_bought_30_candy_bars :
  candy_bars_bought 7 3 22 4 = 30 :=
by
  sorry

end amanda_bought_30_candy_bars_l305_305219


namespace least_number_to_subtract_l305_305033

theorem least_number_to_subtract (n : ℕ) (p : ℕ) (hdiv : p = 47) (hn : n = 929) 
: ∃ k, n - 44 = k * p := by
  sorry

end least_number_to_subtract_l305_305033


namespace find_x_for_set_6_l305_305542

theorem find_x_for_set_6 (x : ℝ) (h : 6 ∈ ({2, 4, x^2 - x} : Set ℝ)) : x = 3 ∨ x = -2 := 
by 
  sorry

end find_x_for_set_6_l305_305542


namespace value_of_expression_l305_305248

variable {a : ℝ}

theorem value_of_expression (h : a^2 + 2 * a - 1 = 0) : 2 * a^2 + 4 * a - 2024 = -2022 :=
by
  sorry

end value_of_expression_l305_305248


namespace number_of_small_cubes_l305_305819

-- Definition of the conditions from the problem
def painted_cube (n : ℕ) :=
  6 * (n - 2) * (n - 2) = 54

-- The theorem we need to prove
theorem number_of_small_cubes (n : ℕ) (h : painted_cube n) : n^3 = 125 :=
by
  have h1 : 6 * (n - 2) * (n - 2) = 54 := h
  sorry

end number_of_small_cubes_l305_305819


namespace Jina_mascots_total_l305_305873

theorem Jina_mascots_total :
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  teddies + bunnies + koala + additional_teddies = 51 :=
by
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  show teddies + bunnies + koala + additional_teddies = 51
  sorry

end Jina_mascots_total_l305_305873


namespace find_removed_number_l305_305617

def list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

def target_average : ℝ := 8.2

theorem find_removed_number (n : ℕ) (h : n ∈ list) :
  (list.sum - n) / (list.length - 1) = target_average -> n = 5 := by
  sorry

end find_removed_number_l305_305617


namespace max_side_length_l305_305960

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l305_305960


namespace am_gm_inequality_l305_305839

theorem am_gm_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end am_gm_inequality_l305_305839


namespace g_zero_l305_305877

variable (f g h : Polynomial ℤ) -- Assume f, g, h are polynomials over the integers

-- Condition: h(x) = f(x) * g(x)
axiom h_def : h = f * g

-- Condition: The constant term of f(x) is 2
axiom f_const : f.coeff 0 = 2

-- Condition: The constant term of h(x) is -6
axiom h_const : h.coeff 0 = -6

-- Proof statement that g(0) = -3
theorem g_zero : g.coeff 0 = -3 := by
  sorry

end g_zero_l305_305877


namespace fewer_cucumbers_than_potatoes_l305_305051

theorem fewer_cucumbers_than_potatoes :
  ∃ C : ℕ, 237 + C + 2 * C = 768 ∧ 237 - C = 60 :=
by
  sorry

end fewer_cucumbers_than_potatoes_l305_305051


namespace scientific_notation_of_large_number_l305_305013

theorem scientific_notation_of_large_number :
  100000000000 = 1 * 10^11 :=
sorry

end scientific_notation_of_large_number_l305_305013


namespace number_101_in_pascals_triangle_l305_305415

/-- Prove that the number 101 appears in exactly one row of Pascal's Triangle, specifically the 101st row. -/
theorem number_101_in_pascals_triangle : 
  ∃! n : ℕ, (∃ k : ℕ, k ≤ n ∧ k ≠ 0 ∧ binom n k = 101) ∧ n = 101 :=
by sorry

end number_101_in_pascals_triangle_l305_305415


namespace find_k_l305_305247

def a : ℝ × ℝ := (2, 1)
def b (k : ℝ) : ℝ × ℝ := (-2, k)
def vec_op (a b : ℝ × ℝ) : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

noncomputable def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_k (k : ℝ) : dot_prod a (vec_op a (b k)) = 0 → k = 14 :=
by
  sorry

end find_k_l305_305247


namespace solve_for_A_l305_305910

theorem solve_for_A : ∃ (A : ℕ), A7 = 10 * A + 7 ∧ A7 + 30 = 77 ∧ A = 4 :=
by
  sorry

end solve_for_A_l305_305910


namespace find_15th_term_l305_305292

-- Define the initial terms and the sequence properties
def first_term := 4
def second_term := 13
def third_term := 22

-- Define the common difference
def common_difference := second_term - first_term

-- Define the nth term formula for arithmetic sequence
def nth_term (a d : ℕ) (n : ℕ) := a + (n - 1) * d

-- State the theorem
theorem find_15th_term : nth_term first_term common_difference 15 = 130 := by
  -- The proof will come here
  sorry

end find_15th_term_l305_305292


namespace find_50th_term_arithmetic_sequence_l305_305222

theorem find_50th_term_arithmetic_sequence :
  let a₁ := 3
  let d := 7
  let a₅₀ := a₁ + (50 - 1) * d
  a₅₀ = 346 :=
by
  let a₁ := 3
  let d := 7
  let a₅₀ := a₁ + (50 - 1) * d
  show a₅₀ = 346
  sorry

end find_50th_term_arithmetic_sequence_l305_305222


namespace descent_phase_duration_l305_305520

noncomputable def start_time_in_seconds : ℕ := 45 * 60 + 39
noncomputable def end_time_in_seconds : ℕ := 47 * 60 + 33

theorem descent_phase_duration :
  end_time_in_seconds - start_time_in_seconds = 114 := by
  sorry

end descent_phase_duration_l305_305520


namespace tan_alpha_eq_three_sin_cos_l305_305695

theorem tan_alpha_eq_three_sin_cos (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin α * Real.cos α = 3 / 10 :=
by 
  sorry

end tan_alpha_eq_three_sin_cos_l305_305695


namespace min_correct_answers_for_score_above_60_l305_305735

theorem min_correct_answers_for_score_above_60 :
  ∃ (x : ℕ), 6 * x - 2 * (15 - x) > 60 ∧ x = 12 :=
by
  sorry

end min_correct_answers_for_score_above_60_l305_305735


namespace initial_amount_liquid_A_l305_305194

theorem initial_amount_liquid_A (A B : ℝ) (h1 : A / B = 4)
    (h2 : (A / (B + 40)) = 2 / 3) : A = 32 := by
  sorry

end initial_amount_liquid_A_l305_305194


namespace ellipse_foci_distance_l305_305656

theorem ellipse_foci_distance (x y : ℝ) (h : 9 * x^2 + y^2 = 36) : 
  let a := 6
      b := 2
      c := Real.sqrt (a^2 - b^2)
  in 2 * c = 8 * Real.sqrt 2 :=
by
  sorry

end ellipse_foci_distance_l305_305656


namespace original_cost_price_l305_305496

theorem original_cost_price (C : ℝ) : 
  (0.89 * C * 1.20 = 54000) → C = 50561.80 :=
by
  sorry

end original_cost_price_l305_305496


namespace continuous_compound_interest_solution_l305_305611

noncomputable def continuous_compound_interest_rate 
  (A P: ℝ) (t: ℝ) (h_A_value: A = 760) (h_P_value: P = 600) (h_t_value: t = 4) : ℝ :=
  (Real.log (A / P)) / t

theorem continuous_compound_interest_solution :
  continuous_compound_interest_rate 760 600 4 (by norm_num) (by norm_num) (by norm_num) ≈ 0.05909725 :=
by
  unfold continuous_compound_interest_rate
  norm_num
  rw [← Real.log_div]
  sorry

end continuous_compound_interest_solution_l305_305611


namespace meaningful_expression_condition_l305_305619

theorem meaningful_expression_condition (x : ℝ) : (x > 1) ↔ (∃ y : ℝ, y = 2 / Real.sqrt (x - 1)) :=
by
  sorry

end meaningful_expression_condition_l305_305619


namespace initial_points_count_l305_305147

theorem initial_points_count (k : ℕ) (h : (4 * k - 3) = 101): k = 26 :=
by 
  sorry

end initial_points_count_l305_305147


namespace probability_all_digits_distinct_probability_all_digits_odd_l305_305356

-- Definitions to be used in the proof
def total_possibilities : ℕ := 10^5
def all_distinct_possibilities : ℕ := 10 * 9 * 8 * 7 * 6
def all_odd_possibilities : ℕ := 5^5

-- Probabilities
def prob_all_distinct : ℚ := all_distinct_possibilities / total_possibilities
def prob_all_odd : ℚ := all_odd_possibilities / total_possibilities

-- Lean 4 Statements to Prove
theorem probability_all_digits_distinct :
  prob_all_distinct = 30240 / 100000 := by
  sorry

theorem probability_all_digits_odd :
  prob_all_odd = 3125 / 100000 := by
  sorry

end probability_all_digits_distinct_probability_all_digits_odd_l305_305356


namespace teacher_already_graded_worksheets_l305_305643

-- Define the conditions
def num_worksheets : ℕ := 9
def problems_per_worksheet : ℕ := 4
def remaining_problems : ℕ := 16
def total_problems := num_worksheets * problems_per_worksheet

-- Define the required proof
theorem teacher_already_graded_worksheets :
  (total_problems - remaining_problems) / problems_per_worksheet = 5 :=
by sorry

end teacher_already_graded_worksheets_l305_305643


namespace women_attended_l305_305370

theorem women_attended (m w : ℕ) 
  (h_danced_with_4_women : ∀ (k : ℕ), k < m → k * 4 = 60)
  (h_danced_with_3_men : ∀ (k : ℕ), k < w → 3 * (k * (m / 3)) = 60)
  (h_men_count : m = 15) : 
  w = 20 := 
sorry

end women_attended_l305_305370


namespace increasing_function_inv_condition_l305_305341

-- Given a strictly increasing real-valued function f on ℝ with an inverse,
-- satisfying the condition f(x) + f⁻¹(x) = 2x for all x in ℝ,
-- prove that f(x) = x + b, where b is a real constant.

theorem increasing_function_inv_condition (f : ℝ → ℝ) (hf_strict_mono : StrictMono f)
  (hf_inv : ∀ x, f (f⁻¹ x) = x ∧ f⁻¹ (f x) = x)
  (hf_condition : ∀ x, f x + f⁻¹ x = 2 * x) :
  ∃ b : ℝ, ∀ x, f x = x + b :=
sorry

end increasing_function_inv_condition_l305_305341


namespace reciprocals_harmonic_progression_of_arithmetic_progression_l305_305459

open Real

theorem reciprocals_harmonic_progression_of_arithmetic_progression
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : b - a = c - b) :
  let a1 := 1 / a,
      b1 := 1 / b,
      c1 := 1 / c
  in 2 * c1 = a1 + b1 + b1 * c1 / a1 :=
by
  sorry

end reciprocals_harmonic_progression_of_arithmetic_progression_l305_305459


namespace extra_flowers_correct_l305_305648

variable (pickedTulips : ℕ) (pickedRoses : ℕ) (usedFlowers : ℕ)

def totalFlowers : ℕ := pickedTulips + pickedRoses
def extraFlowers : ℕ := totalFlowers pickedTulips pickedRoses - usedFlowers

theorem extra_flowers_correct : 
  pickedTulips = 39 → pickedRoses = 49 → usedFlowers = 81 → extraFlowers pickedTulips pickedRoses usedFlowers = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end extra_flowers_correct_l305_305648


namespace inequality_solution_sum_of_m_and_2n_l305_305568

-- Define the function f(x) = |x - a|
def f (x a : ℝ) : ℝ := abs (x - a)

-- Part (1): The inequality problem for a = 2
theorem inequality_solution (x : ℝ) :
  f x 2 ≥ 4 - abs (x - 1) → x ≤ 2 / 3 := sorry

-- Part (2): Given conditions with solution set [0, 2] and condition on m and n
theorem sum_of_m_and_2n (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : ∀ x, f x 1 ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) (h₄ : 1 / m + 1 / (2 * n) = 1) :
  m + 2 * n ≥ 4 := sorry

end inequality_solution_sum_of_m_and_2n_l305_305568


namespace Jina_has_51_mascots_l305_305875

def teddies := 5
def bunnies := 3 * teddies
def koala_bear := 1
def additional_teddies := 2 * bunnies
def total_mascots := teddies + bunnies + koala_bear + additional_teddies

theorem Jina_has_51_mascots : total_mascots = 51 := by
  sorry

end Jina_has_51_mascots_l305_305875


namespace predicted_temperature_l305_305515

def avg_x (x_vals : List ℕ) : ℕ := (x_vals.foldl (· + ·) 0) / x_vals.length
def avg_y (y_vals : List ℕ) : ℕ := (y_vals.foldl (· + ·) 0) / y_vals.length

theorem predicted_temperature (k : ℚ) (x_vals y_vals : List ℚ) (x : ℕ) (H : (avg_x x_vals = 40) ∧ (avg_y y_vals = 30) ∧ k = 20) :
  0.25 * 80 + k = 40 :=
by
  sorry

end predicted_temperature_l305_305515


namespace multiple_of_old_edition_l305_305497

theorem multiple_of_old_edition 
  (new_pages: ℕ) 
  (old_pages: ℕ) 
  (difference: ℕ) 
  (m: ℕ) 
  (h1: new_pages = 450) 
  (h2: old_pages = 340) 
  (h3: 450 = 340 * m - 230) : 
  m = 2 :=
sorry

end multiple_of_old_edition_l305_305497


namespace garden_roller_area_l305_305767

theorem garden_roller_area (D : ℝ) (A : ℝ) (π : ℝ) (L_new : ℝ) :
  D = 1.4 → A = 88 → π = 22/7 → L_new = 4 → A = 5 * (2 * π * (D / 2) * L_new) :=
by sorry

end garden_roller_area_l305_305767


namespace quadratic_eq_transformed_l305_305022

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2 * x - 7 = 0

-- Define the form to transform to using completing the square method
def transformed_eq (x : ℝ) : Prop := (x - 1)^2 = 8

-- The theorem to be proved
theorem quadratic_eq_transformed (x : ℝ) :
  quadratic_eq x → transformed_eq x :=
by
  intros h
  -- here we would use steps of completing the square to transform the equation
  sorry

end quadratic_eq_transformed_l305_305022


namespace lowest_possible_number_of_students_l305_305485

theorem lowest_possible_number_of_students :
  ∃ n : ℕ, (n % 12 = 0 ∧ n % 24 = 0) ∧ ∀ m : ℕ, ((m % 12 = 0 ∧ m % 24 = 0) → n ≤ m) :=
sorry

end lowest_possible_number_of_students_l305_305485


namespace graph_of_equation_l305_305189

theorem graph_of_equation :
  ∀ (x y : ℝ), (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (x + y + 2 = 0 ∨ x+y = 0 ∨ x-y = 0) ∧ 
  ¬(∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
    (x₁ + y₁ + 2 = 0 ∧ x₁ + y₁ = 0) ∧ 
    (x₂ + y₂ + 2 = 0 ∧ x₂ = -x₂) ∧ 
    (x₃ + y₃ + 2 = 0 ∧ x₃ - y₃ = 0)) := 
sorry

end graph_of_equation_l305_305189


namespace finding_a_of_geometric_sequence_l305_305783
noncomputable def geometric_sequence_a : Prop :=
  ∃ a : ℝ, (1, a, 2) = (1, a, 2) ∧ a^2 = 2

theorem finding_a_of_geometric_sequence :
  ∃ a : ℝ, (1, a, 2) = (1, a, 2) → a = Real.sqrt 2 ∨ a = -Real.sqrt 2 :=
by
  sorry

end finding_a_of_geometric_sequence_l305_305783


namespace max_triangle_side_l305_305995

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l305_305995


namespace gcd_of_16_and_12_l305_305792

theorem gcd_of_16_and_12 : Nat.gcd 16 12 = 4 := by
  sorry

end gcd_of_16_and_12_l305_305792


namespace negation_proposition_l305_305162

theorem negation_proposition :
  (∀ x : ℝ, |x - 2| + |x - 4| > 3) = ¬(∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) :=
  by sorry

end negation_proposition_l305_305162


namespace max_cables_191_l305_305645

/-- 
  There are 30 employees: 20 with brand A computers and 10 with brand B computers.
  Cables can only connect a brand A computer to a brand B computer.
  Employees can communicate with each other if their computers are directly connected by a cable 
  or by relaying messages through a series of connected computers.
  The maximum possible number of cables used to ensure every employee can communicate with each other
  is 191.
-/
theorem max_cables_191 (A B : ℕ) (hA : A = 20) (hB : B = 10) : 
  ∃ (max_cables : ℕ), max_cables = 191 ∧ 
  (∀ (i j : ℕ), (i ≤ A ∧ j ≤ B) → (i = A ∨ j = B) → i * j ≤ max_cables) := 
sorry

end max_cables_191_l305_305645


namespace coefficient_x5_in_expansion_of_3x_plus_2_power_7_l305_305612

theorem coefficient_x5_in_expansion_of_3x_plus_2_power_7 :
  (∑ k in Fin₇.finset, (7.choose k) * (3 * x) ^ (7 - k) * 2 ^ k) = 20412 * x^5 :=
by sorry

end coefficient_x5_in_expansion_of_3x_plus_2_power_7_l305_305612


namespace probability_alpha_in_interval_l305_305857

def vector_of_die_rolls_angle_probability : ℚ := 
  let total_outcomes := 36
  let favorable_pairs := 15
  favorable_pairs / total_outcomes

theorem probability_alpha_in_interval (m n : ℕ)
  (hm : 1 ≤ m ∧ m ≤ 6) (hn : 1 ≤ n ∧ n ≤ 6) :
  (vector_of_die_rolls_angle_probability = 5 / 12) := by
  sorry

end probability_alpha_in_interval_l305_305857


namespace blue_paint_gallons_l305_305351

-- Define the total gallons of paint used
def total_paint_gallons : ℕ := 6689

-- Define the gallons of white paint used
def white_paint_gallons : ℕ := 660

-- Define the corresponding proof problem
theorem blue_paint_gallons : 
  ∀ total white blue : ℕ, total = 6689 → white = 660 → blue = total - white → blue = 6029 := by
  sorry

end blue_paint_gallons_l305_305351


namespace toll_for_18_wheel_truck_l305_305601

-- Define the number of axles given the conditions
def num_axles (total_wheels rear_axle_wheels front_axle_wheels : ℕ) : ℕ :=
  let rear_axles := (total_wheels - front_axle_wheels) / rear_axle_wheels
  rear_axles + 1

-- Define the toll calculation given the number of axles
def toll (axles : ℕ) : ℝ :=
  1.50 + 0.50 * (axles - 2)

-- Constants specific to the problem
def total_wheels : ℕ := 18
def rear_axle_wheels : ℕ := 4
def front_axle_wheels : ℕ := 2

-- Calculate the number of axles for the given truck
def truck_axles : ℕ := num_axles total_wheels rear_axle_wheels front_axle_wheels

-- The actual statement to prove
theorem toll_for_18_wheel_truck : toll truck_axles = 3.00 :=
  by
    -- proof will go here
    sorry

end toll_for_18_wheel_truck_l305_305601


namespace select_4_people_arrangement_3_day_new_year_l305_305763

def select_4_people_arrangement (n k : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.factorial (n - 2) / Nat.factorial 2

theorem select_4_people_arrangement_3_day_new_year :
  select_4_people_arrangement 7 4 = 420 :=
by
  -- proof to be filled in
  sorry

end select_4_people_arrangement_3_day_new_year_l305_305763


namespace yield_percentage_l305_305628

theorem yield_percentage (d : ℝ) (q : ℝ) (f : ℝ) : d = 12 → q = 150 → f = 100 → (d * f / q) * 100 = 8 :=
by
  intros h_d h_q h_f
  rw [h_d, h_q, h_f]
  sorry

end yield_percentage_l305_305628


namespace right_triangle_leg_length_l305_305422

theorem right_triangle_leg_length (a b c : ℕ) (h_c : c = 13) (h_a : a = 12) (h_pythagorean : a^2 + b^2 = c^2) :
  b = 5 := 
by {
  -- Provide a placeholder for the proof
  sorry
}

end right_triangle_leg_length_l305_305422


namespace men_in_first_group_l305_305348

theorem men_in_first_group (M : ℕ) (h1 : 20 * 30 * (480 / (20 * 30)) = 480) (h2 : M * 15 * (120 / (M * 15)) = 120) :
  M = 10 :=
by sorry

end men_in_first_group_l305_305348


namespace line_through_two_points_l305_305728

theorem line_through_two_points (A B : ℝ × ℝ)
  (hA : A = (2, -3))
  (hB : B = (1, 4)) :
  ∃ (m b : ℝ), (∀ x y : ℝ, (y = m * x + b) ↔ ((x, y) = A ∨ (x, y) = B)) ∧ m = -7 ∧ b = 11 := by
  sorry

end line_through_two_points_l305_305728


namespace equal_sundays_tuesdays_l305_305812

theorem equal_sundays_tuesdays (days_in_month : ℕ) (week_days : ℕ) (extra_days : ℕ) :
  days_in_month = 30 → week_days = 7 → extra_days = 2 → 
  ∃ n, n = 3 ∧ ∀ start_day : ℕ, start_day = 3 ∨ start_day = 4 ∨ start_day = 5 :=
by sorry

end equal_sundays_tuesdays_l305_305812


namespace percentage_reduction_price_increase_for_profit_price_increase_max_profit_l305_305932

-- Define the conditions
def original_price : ℝ := 50
def final_price : ℝ := 32
def daily_sales : ℝ := 500
def profit_per_kg : ℝ := 10
def sales_decrease_per_yuan : ℝ := 20
def required_daily_profit : ℝ := 6000
def max_possible_profit : ℝ := 6125

-- Proving the percentage reduction each time
theorem percentage_reduction (a : ℝ) :
  (original_price * (1 - a) ^ 2 = final_price) → (a = 0.2) :=
sorry

-- Proving the price increase per kilogram to ensure a daily profit of 6000 yuan
theorem price_increase_for_profit (x : ℝ) :
  ((profit_per_kg + x) * (daily_sales - sales_decrease_per_yuan * x) = required_daily_profit) → (x = 5) :=
sorry

-- Proving the price increase per kilogram to maximize daily profit
theorem price_increase_max_profit (x : ℝ) :
  ((profit_per_kg + x) * (daily_sales - sales_decrease_per_yuan * x) = max_possible_profit) → (x = 7.5) :=
sorry

end percentage_reduction_price_increase_for_profit_price_increase_max_profit_l305_305932


namespace a_friend_gcd_l305_305000

theorem a_friend_gcd (a b : ℕ) (d : ℕ) (hab : a * b = d * d) (hd : d = Nat.gcd a b) : ∃ k : ℕ, a * d = k * k := by
  sorry

end a_friend_gcd_l305_305000


namespace find_point_A_coordinates_l305_305210

theorem find_point_A_coordinates :
  ∃ (A : ℝ × ℝ), (A.2 = 0) ∧ 
  (dist A (-3, 2) = dist A (4, -5)) →
  A = (2, 0) :=
by
-- We'll provide the explicit exact proof later
-- Proof steps would go here
sorry 

end find_point_A_coordinates_l305_305210


namespace find_side_length_l305_305424

theorem find_side_length (a b c : ℝ) (A : ℝ) 
  (h1 : Real.cos A = 7 / 8) 
  (h2 : c - a = 2) 
  (h3 : b = 3) : 
  a = 2 := by
  sorry

end find_side_length_l305_305424


namespace prime_p_satisfies_condition_l305_305831

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_p_satisfies_condition {p : ℕ} (hp : is_prime p) (hp2_8 : is_prime (p^2 + 8)) : p = 3 :=
sorry

end prime_p_satisfies_condition_l305_305831


namespace gcd_50420_35313_l305_305084

theorem gcd_50420_35313 : Int.gcd 50420 35313 = 19 := 
sorry

end gcd_50420_35313_l305_305084


namespace least_area_of_prime_dim_l305_305638

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem least_area_of_prime_dim (l w : ℕ) (h_perimeter : 2 * (l + w) = 120)
    (h_integer_dims : l > 0 ∧ w > 0) (h_prime_dim : is_prime l ∨ is_prime w) :
    l * w = 116 :=
sorry

end least_area_of_prime_dim_l305_305638


namespace find_triples_l305_305835

theorem find_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 = c^2) ∧ (a^3 + b^3 + 1 = (c-1)^3) ↔ (a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 8 ∧ b = 6 ∧ c = 10) :=
by
  sorry

end find_triples_l305_305835


namespace line_does_not_pass_second_quadrant_l305_305725

theorem line_does_not_pass_second_quadrant 
  (A B C x y : ℝ) 
  (h1 : A * C < 0) 
  (h2 : B * C > 0) 
  (h3 : A * x + B * y + C = 0) :
  ¬ (x < 0 ∧ y > 0) := 
sorry

end line_does_not_pass_second_quadrant_l305_305725


namespace average_reading_time_correct_l305_305180

-- We define total_reading_time as a parameter representing the sum of reading times
noncomputable def total_reading_time : ℝ := sorry

-- We define the number of students as a constant
def number_of_students : ℕ := 50

-- We define the average reading time per student based on the provided data
noncomputable def average_reading_time : ℝ :=
  total_reading_time / number_of_students

-- The theorem we need to prove: that the average reading time per student is correctly calculated
theorem average_reading_time_correct :
  ∃ (total_reading_time : ℝ), average_reading_time = total_reading_time / number_of_students :=
by
  -- since total_reading_time and number_of_students are already defined, we prove the theorem using them
  use total_reading_time
  exact rfl

end average_reading_time_correct_l305_305180


namespace Ara_height_in_inches_l305_305893

theorem Ara_height_in_inches (Shea_current_height : ℝ) (Shea_growth_percentage : ℝ) (Ara_growth_factor : ℝ) (Shea_growth_amount : ℝ) (Ara_current_height : ℝ) :
  Shea_current_height = 75 →
  Shea_growth_percentage = 0.25 →
  Ara_growth_factor = 1 / 3 →
  Shea_growth_amount = 75 * (1 / (1 + 0.25)) * 0.25 →
  Ara_current_height = 75 * (1 / (1 + 0.25)) + (75 * (1 / (1 + 0.25)) * 0.25) * (1 / 3) →
  Ara_current_height = 65 :=
by sorry

end Ara_height_in_inches_l305_305893


namespace distance_to_city_hall_l305_305127

variable (d : ℝ) (t : ℝ)

-- Conditions
def condition1 : Prop := d = 45 * (t + 1.5)
def condition2 : Prop := d - 45 = 65 * (t - 1.25)
def condition3 : Prop := t > 0

theorem distance_to_city_hall
  (h1 : condition1 d t)
  (h2 : condition2 d t)
  (h3 : condition3 t)
  : d = 300 := by
  sorry

end distance_to_city_hall_l305_305127


namespace integral_eq_exp_integral_eq_one_l305_305898

noncomputable
def y1 (τ : ℝ) (t : ℝ) (y : ℝ → ℝ) : Prop :=
  y τ = ∫ x in (0 : ℝ)..t, y x + 1

theorem integral_eq_exp (y : ℝ → ℝ) : 
  (∀ τ t, y1 τ t y) ↔ (∀ t, y t = Real.exp t) := 
  sorry

noncomputable
def y2 (t : ℝ) (y : ℝ → ℝ) : Prop :=
  ∫ x in (0 : ℝ)..t, y x * Real.sin (t - x) = 1 - Real.cos t

theorem integral_eq_one (y : ℝ → ℝ) : 
  (∀ t, y2 t y) ↔ (∀ t, y t = 1) :=
  sorry

end integral_eq_exp_integral_eq_one_l305_305898


namespace calc_g_3_l305_305390

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x - 1

theorem calc_g_3 : g (g (g (g 3))) = 1 := by
  sorry

end calc_g_3_l305_305390


namespace inequality_solution_set_l305_305540

noncomputable def f (x : ℝ) : ℝ := x + 1 / (2 * x) + 2

lemma f_increasing {x₁ x₂ : ℝ} (hx₁ : 1 ≤ x₁) (hx₂ : 1 ≤ x₂) (h : x₁ < x₂) : f x₁ < f x₂ := sorry

lemma solve_inequality (x : ℝ) (hx : 1 ≤ x) : (2 * x - 1 / 2 < x + 1007) → (f (2 * x - 1 / 2) < f (x + 1007)) := sorry

theorem inequality_solution_set {x : ℝ} : (1 ≤ x) → (2 * x - 1 / 2 < x + 1007) ↔ (3 / 4 ≤ x ∧ x < 2015 / 2) := sorry

end inequality_solution_set_l305_305540


namespace intersection_of_complements_l305_305776

open Set

theorem intersection_of_complements (U : Set ℕ) (A B : Set ℕ)
  (hU : U = {1,2,3,4,5,6,7,8})
  (hA : A = {3,4,5})
  (hB : B = {1,3,6}) :
  (U \ A) ∩ (U \ B) = {2,7,8} := by
  rw [hU, hA, hB]
  sorry

end intersection_of_complements_l305_305776


namespace smallest_prime_factor_in_setB_l305_305580

def setB : Set ℕ := {55, 57, 58, 59, 61}

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 2 then 2 else (Nat.minFac (Nat.pred n)).succ

theorem smallest_prime_factor_in_setB :
  ∃ n ∈ setB, smallest_prime_factor n = 2 := by
  sorry

end smallest_prime_factor_in_setB_l305_305580


namespace middle_joints_capacity_l305_305007

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def bamboo_tube_capacity (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 4.5 ∧ a 6 + a 7 + a 8 = 2.5 ∧ arithmetic_seq a (a 1 - a 0)

theorem middle_joints_capacity (a : ℕ → ℝ) (d : ℝ) (h : bamboo_tube_capacity a) : 
  a 3 + a 4 + a 5 = 3.5 :=
by
  sorry

end middle_joints_capacity_l305_305007


namespace intercept_sum_l305_305626

theorem intercept_sum (x0 y0 : ℕ) (h1 : x0 < 17) (h2 : y0 < 17)
  (hx : 7 * x0 ≡ 2 [MOD 17]) (hy : 3 * y0 ≡ 15 [MOD 17]) : x0 + y0 = 17 :=
sorry

end intercept_sum_l305_305626


namespace x_lt_y_l305_305441

variable {a b c d x y : ℝ}

theorem x_lt_y 
  (ha : a > 1) 
  (hb : b > 1) 
  (hc : c > 1) 
  (hd : d > 1)
  (h1 : a^x + b^y = (a^2 + b^2)^x)
  (h2 : c^x + d^y = 2^y * (cd)^(y/2)) : 
  x < y :=
by 
  sorry

end x_lt_y_l305_305441


namespace joe_total_paint_used_l305_305564

-- Define the initial amount of paint Joe buys.
def initial_paint : ℕ := 360

-- Define the fraction of paint used during the first week.
def first_week_fraction := 1 / 4

-- Define the fraction of remaining paint used during the second week.
def second_week_fraction := 1 / 2

-- Define the total paint used by Joe in the first week.
def paint_used_first_week := first_week_fraction * initial_paint

-- Define the remaining paint after the first week.
def remaining_paint_after_first_week := initial_paint - paint_used_first_week

-- Define the total paint used by Joe in the second week.
def paint_used_second_week := second_week_fraction * remaining_paint_after_first_week

-- Define the total paint used by Joe.
def total_paint_used := paint_used_first_week + paint_used_second_week

-- The theorem to be proven: the total amount of paint Joe has used is 225 gallons.
theorem joe_total_paint_used : total_paint_used = 225 := by
  sorry

end joe_total_paint_used_l305_305564


namespace cos_angle_correct_l305_305409

noncomputable def cos_angle_F1PF2 : ℝ :=
  let F₁ := (-2 : ℝ, 0 : ℝ)
  let F₂ := (2 : ℝ, 0 : ℝ)
  let P := (3 * Real.sqrt 2 / 2, Real.sqrt 2 / 2)

  let PF₁ := (F₁.1 - P.1, F₁.2 - P.2)
  let PF₂ := (F₂.1 - P.1, F₂.2 - P.2)

  let dot_product := PF₁.1 * PF₂.1 + PF₁.2 * PF₂.2
  let norm_PF₁ := Real.sqrt ((PF₁.1) * (PF₁.1) + (PF₁.2) * (PF₁.2))
  let norm_PF₂ := Real.sqrt ((PF₁.1) * (PF₂.1) + (PF₁.2) * (PF₂.2))

  dot_product / (norm_PF₁ * norm_PF₂)

theorem cos_angle_correct :
  cos_angle_F1PF2 = 1 / 3 :=
by
  sorry

end cos_angle_correct_l305_305409


namespace reciprocal_neg_half_l305_305164

theorem reciprocal_neg_half : (1 / (- (1 / 2) : ℚ) = -2) :=
by
  sorry

end reciprocal_neg_half_l305_305164


namespace sum_of_perimeters_l305_305644

theorem sum_of_perimeters (a : ℕ → ℝ) (h₁ : a 0 = 180) (h₂ : ∀ n, a (n + 1) = 1 / 2 * a n) :
  (∑' n, a n) = 360 :=
by
  sorry

end sum_of_perimeters_l305_305644


namespace value_of_f_neg1_l305_305791

def f (x : ℝ) : ℝ := x^3 + x^2 - 2 * x

theorem value_of_f_neg1 : f (-1) = 2 := by
  sorry

end value_of_f_neg1_l305_305791


namespace find_max_side_length_l305_305947

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l305_305947


namespace half_angle_quadrant_l305_305726

theorem half_angle_quadrant
  (α : ℝ) (k : ℤ)
  (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (∃ m : ℤ, m * π < α / 2 ∧ α / 2 < m * π + π / 2) :=
by
  sorry

end half_angle_quadrant_l305_305726


namespace midpoint_sum_l305_305465

theorem midpoint_sum (x1 y1 x2 y2 : ℕ) (h₁ : x1 = 4) (h₂ : y1 = 7) (h₃ : x2 = 12) (h₄ : y2 = 19) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 21 :=
by
  sorry

end midpoint_sum_l305_305465


namespace range_of_y0_l305_305502

theorem range_of_y0
  (y0 : ℝ)
  (h_tangent : ∃ N : ℝ × ℝ, (N.1^2 + N.2^2 = 1) ∧ ((↑(Real.sqrt 3 - N.1)^2 + (y0 - N.2)^2) = 1))
  (h_angle : ∀ N : ℝ × ℝ, (N.1^2 + N.2^2 = 1) ∧ ((↑(Real.sqrt 3 - N.1)^2 + (y0 - N.2)^2 = 1)) → (Real.arccos ((Real.sqrt 3 - N.1)/Real.sqrt ((3 - 2 * N.1 * Real.sqrt 3 + N.1^2) + (y0 - N.2)^2)) ≥ π / 6)) :
  -1 ≤ y0 ∧ y0 ≤ 1 :=
by
  sorry

end range_of_y0_l305_305502


namespace average_age_of_group_l305_305587

theorem average_age_of_group :
  let n_graders := 40
  let n_parents := 50
  let n_teachers := 10
  let avg_age_graders := 12
  let avg_age_parents := 35
  let avg_age_teachers := 45
  let total_individuals := n_graders + n_parents + n_teachers
  let total_age := n_graders * avg_age_graders + n_parents * avg_age_parents + n_teachers * avg_age_teachers
  (total_age : ℚ) / total_individuals = 26.8 :=
by
  sorry

end average_age_of_group_l305_305587


namespace smallest_number_diminished_by_8_divisible_by_9_6_12_18_l305_305338

theorem smallest_number_diminished_by_8_divisible_by_9_6_12_18 :
  ∃ x : ℕ, (x - 8) % Nat.lcm (Nat.lcm 9 6) (Nat.lcm 12 18) = 0 ∧ ∀ y : ℕ, (y - 8) % Nat.lcm (Nat.lcm 9 6) (Nat.lcm 12 18) = 0 → x ≤ y → x = 44 :=
by
  sorry

end smallest_number_diminished_by_8_divisible_by_9_6_12_18_l305_305338


namespace sqrt_50_product_consecutive_integers_l305_305306

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l305_305306


namespace total_payment_divisible_by_25_l305_305366

theorem total_payment_divisible_by_25 (B : ℕ) (h1 : 0 ≤ B ∧ B ≤ 9) : 
  (2005 + B * 1000) % 25 = 0 :=
by
  sorry

end total_payment_divisible_by_25_l305_305366


namespace number_of_pairs_l305_305238

theorem number_of_pairs (h : ∀ (a : ℝ) (b : ℕ), 0 < a → 2 ≤ b ∧ b ≤ 200 → (Real.log a / Real.log b) ^ 2017 = Real.log (a ^ 2017) / Real.log b) :
  ∃ n, n = 597 ∧ ∀ b : ℕ, 2 ≤ b ∧ b ≤ 200 → 
    ∃ a1 a2 a3 : ℝ, 0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 
      (Real.log a1 / Real.log b) = 0 ∧ 
      (Real.log a2 / Real.log b) = 2017^((1:ℝ)/2016) ∧ 
      (Real.log a3 / Real.log b) = -2017^((1:ℝ)/2016) :=
sorry

end number_of_pairs_l305_305238


namespace total_strawberries_weight_is_72_l305_305447

-- Define the weights
def Marco_strawberries_weight := 19
def dad_strawberries_weight := Marco_strawberries_weight + 34 

-- The total weight of their strawberries
def total_strawberries_weight := Marco_strawberries_weight + dad_strawberries_weight

-- Prove that the total weight is 72 pounds
theorem total_strawberries_weight_is_72 : total_strawberries_weight = 72 := by
  sorry

end total_strawberries_weight_is_72_l305_305447


namespace product_of_consecutive_integers_sqrt_50_l305_305329

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l305_305329


namespace equal_sundays_tuesdays_l305_305813

theorem equal_sundays_tuesdays (days_in_month : ℕ) (week_days : ℕ) (extra_days : ℕ) :
  days_in_month = 30 → week_days = 7 → extra_days = 2 → 
  ∃ n, n = 3 ∧ ∀ start_day : ℕ, start_day = 3 ∨ start_day = 4 ∨ start_day = 5 :=
by sorry

end equal_sundays_tuesdays_l305_305813


namespace total_paintings_is_correct_l305_305458

-- Definitions for Philip's schedule and starting number of paintings
def philip_paintings_monday_and_tuesday := 3
def philip_paintings_wednesday := 2
def philip_paintings_thursday_and_friday := 5
def philip_initial_paintings := 20

-- Definitions for Amelia's schedule and starting number of paintings
def amelia_paintings_every_day := 2
def amelia_initial_paintings := 45

-- Calculation of total paintings after 5 weeks
def philip_weekly_paintings := 
  (2 * philip_paintings_monday_and_tuesday) + 
  philip_paintings_wednesday + 
  (2 * philip_paintings_thursday_and_friday)

def amelia_weekly_paintings := 
  7 * amelia_paintings_every_day

def total_paintings_after_5_weeks := 5 * philip_weekly_paintings + philip_initial_paintings + 5 * amelia_weekly_paintings + amelia_initial_paintings

-- Proof statement
theorem total_paintings_is_correct :
  total_paintings_after_5_weeks = 225 :=
  by sorry

end total_paintings_is_correct_l305_305458


namespace variable_cost_per_book_l305_305943

theorem variable_cost_per_book
  (F : ℝ) (S : ℝ) (N : ℕ) (V : ℝ)
  (fixed_cost : F = 56430) 
  (selling_price_per_book : S = 21.75) 
  (num_books : N = 4180) 
  (production_eq_sales : S * N = F + V * N) :
  V = 8.25 :=
by sorry

end variable_cost_per_book_l305_305943


namespace monthly_installments_l305_305902

theorem monthly_installments (cash_price deposit installment saving : ℕ) (total_paid installments_made : ℕ) :
  cash_price = 8000 →
  deposit = 3000 →
  installment = 300 →
  saving = 4000 →
  total_paid = cash_price + saving →
  installments_made = (total_paid - deposit) / installment →
  installments_made = 30 :=
by
  intros h_cash_price h_deposit h_installment h_saving h_total_paid h_installments_made
  sorry

end monthly_installments_l305_305902


namespace max_side_of_triangle_exists_max_side_of_elevent_l305_305978

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l305_305978


namespace problem_l305_305701

theorem problem {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h : 3 * a * b = a + 3 * b) :
  (3 * a + b >= 16/3) ∧
  (a * b >= 4/3) ∧
  (a^2 + 9 * b^2 >= 8) ∧
  (¬ (b > 1/2)) :=
by
  sorry

end problem_l305_305701


namespace response_rate_percentage_50_l305_305205

def questionnaire_response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ) : ℕ :=
  (responses_needed * 100) / questionnaires_mailed

theorem response_rate_percentage_50 
  (responses_needed : ℕ) 
  (questionnaires_mailed : ℕ) 
  (h1 : responses_needed = 300) 
  (h2 : questionnaires_mailed = 600) : 
  questionnaire_response_rate_percentage responses_needed questionnaires_mailed = 50 :=
by 
  rw [h1, h2]
  norm_num
  sorry

end response_rate_percentage_50_l305_305205


namespace largest_number_less_than_2_l305_305176

theorem largest_number_less_than_2 (a b c : ℝ) (h_a : a = 0.8) (h_b : b = 1/2) (h_c : c = 0.5) : 
  a < 2 ∧ b < 2 ∧ c < 2 ∧ (∀ x, (x = a ∨ x = b ∨ x = c) → x < 2) → 
  a = 0.8 ∧ 
  (a > b ∧ a > c) ∧ 
  (a < 2) :=
by sorry

end largest_number_less_than_2_l305_305176


namespace hundreds_digit_of_factorial_subtraction_l305_305788

theorem hundreds_digit_of_factorial_subtraction : (30.factorial - 25.factorial) % 1000 / 100 % 10 = 0 :=
by
  sorry

end hundreds_digit_of_factorial_subtraction_l305_305788


namespace ice_formation_l305_305606

-- Definitions for problem setup
variable {T_in T_out T_critical : ℝ}
variable T_inside_surface : ℝ -- The temperature on the inside surface of the trolleybus windows

-- Given conditions
def is_humid (T_in : ℝ) : Prop := true -- 100% humidity condition (always true for this problem)
def condensation_point (T : ℝ) : Prop := T < 0 -- Temperature below which condensation occurs

-- Condition that the inner surface temperature is determined by both inside and outside temperatures
def inner_surface_temp (T_in T_out : ℝ) : ℝ := (T_in + T_out) / 2  -- Simplified model

theorem ice_formation 
  (h_humid : is_humid T_in)
  (h_T_in : T_in < T_critical)
  : condensation_point (inner_surface_temp T_in T_out) :=
sorry

end ice_formation_l305_305606


namespace remainder_3_101_add_5_mod_11_l305_305034

theorem remainder_3_101_add_5_mod_11 : (3 ^ 101 + 5) % 11 = 8 := 
by sorry

end remainder_3_101_add_5_mod_11_l305_305034


namespace ratio_of_parts_l305_305280

theorem ratio_of_parts (N : ℝ) (h1 : (1/4) * (2/5) * N = 14) (h2 : 0.40 * N = 168) : (2/5) * N / N = 1 / 2.5 :=
by
  sorry

end ratio_of_parts_l305_305280


namespace coeff_a_neg_one_in_expansion_l305_305517

open Finset

noncomputable def binom (n k : ℕ) : ℕ := (n.choose k)

theorem coeff_a_neg_one_in_expansion : 
  ∀ (a : ℝ), (∃ c : ℝ, c = -448 ∧ (∀ x : ℝ, (x + (-((1 + 2 * real.sqrt x) / x)))^8 = 
  ∑ k in range (8 + 1), binom 8 k * x^(8 - k) * (-(1 + 2 * real.sqrt x) / x)^k) → 
  coeff_a_neg_one_in_expansion):
  sorry

end coeff_a_neg_one_in_expansion_l305_305517


namespace parallelepiped_side_lengths_l305_305481

theorem parallelepiped_side_lengths (x y z : ℕ) 
  (h1 : x + y + z = 17) 
  (h2 : 2 * x * y + 2 * y * z + 2 * z * x = 180) 
  (h3 : x^2 + y^2 = 100) :
  x = 8 ∧ y = 6 ∧ z = 3 :=
by {
  sorry
}

end parallelepiped_side_lengths_l305_305481


namespace cricket_player_average_increase_l305_305154

theorem cricket_player_average_increase
  (average : ℕ) (n : ℕ) (next_innings_runs : ℕ) 
  (x : ℕ) 
  (h1 : average = 32)
  (h2 : n = 20)
  (h3 : next_innings_runs = 200)
  (total_runs := average * n)
  (new_total_runs := total_runs + next_innings_runs)
  (new_average := (average + x))
  (new_total := new_average * (n + 1)):
  new_total_runs = 840 →
  new_total = 840 →
  x = 8 :=
by
  sorry

end cricket_player_average_increase_l305_305154


namespace pascals_triangle_101_rows_pascals_triangle_only_101_l305_305416

theorem pascals_triangle_101_rows (n : ℕ) :
  (∃ k, (0 ≤ k) ∧ (k ≤ n) ∧ (Nat.choose n k = 101)) → n = 101 :=
begin
  -- assume that there exists some row n where 101 appears in Pascal's Triangle
  intro h,
  cases h with k hk,
  cases hk with hk0 hk1,
  cases hk1 with hk1 hl,
  
  -- we need to show that n = 101
  have h_prime := Nat.prime_101,
  
  -- use the properties of 101 being a prime number and Pascal's Triangle.
  sorry
end

theorem pascals_triangle_only_101 :
  ∀ n : ℕ, (∀ k, (0 ≤ k) ∧ (k ≤ n) → (Nat.choose n k = 101) → n = 101) :=
begin
  intros n k hkn h,
  have h_prime := Nat.prime_101,
  -- use the properties of 101 being a prime number and Pascal's Triangle.
  sorry
end

end pascals_triangle_101_rows_pascals_triangle_only_101_l305_305416


namespace number_of_movies_l305_305473

theorem number_of_movies (B M : ℕ)
  (h1 : B = 15)
  (h2 : B = M + 1) : M = 14 :=
by sorry

end number_of_movies_l305_305473


namespace simple_interest_rate_l305_305490

-- Define the conditions
def S : ℚ := 2500
def P : ℚ := 5000
def T : ℚ := 5

-- Define the proof problem
theorem simple_interest_rate (R : ℚ) (h : S = P * R * T / 100) : R = 10 := by
  sorry

end simple_interest_rate_l305_305490


namespace max_side_of_triangle_exists_max_side_of_elevent_l305_305984

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l305_305984


namespace find_certain_number_l305_305349

theorem find_certain_number (x : ℝ) (h : x + 12.952 - 47.95000000000027 = 3854.002) : x = 3889.000 :=
sorry

end find_certain_number_l305_305349


namespace solve_for_b_l305_305708

theorem solve_for_b (a b : ℚ) 
  (h1 : 8 * a + 3 * b = -1) 
  (h2 : a = b - 3 ) : 
  5 * b = 115 / 11 := 
by 
  sorry

end solve_for_b_l305_305708


namespace sum_prime_factors_2_10_minus_1_l305_305916

theorem sum_prime_factors_2_10_minus_1 : 
  let n := 10 
  let number := 2^n - 1 
  let factors := [3, 5, 7, 11] 
  number.prime_factors.sum = 26 :=
by
  sorry

end sum_prime_factors_2_10_minus_1_l305_305916


namespace polynomial_roots_expression_l305_305565

theorem polynomial_roots_expression 
  (a b α β γ δ : ℝ)
  (h1 : α^2 - a*α - 1 = 0)
  (h2 : β^2 - a*β - 1 = 0)
  (h3 : γ^2 - b*γ - 1 = 0)
  (h4 : δ^2 - b*δ - 1 = 0) :
  ((α - γ)^2 * (β - γ)^2 * (α + δ)^2 * (β + δ)^2) = (b^2 - a^2)^2 :=
sorry

end polynomial_roots_expression_l305_305565


namespace evaluate_expression_l305_305833

-- Given conditions 
def x := 3
def y := 2

-- Prove that y + y(y^x + x!) evaluates to 30.
theorem evaluate_expression : y + y * (y^x + Nat.factorial x) = 30 := by
  sorry

end evaluate_expression_l305_305833


namespace product_of_consecutive_integers_between_sqrt_50_l305_305325

theorem product_of_consecutive_integers_between_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (sqrt 50 ∈ set.Icc (m : ℝ) (n : ℝ)) ∧ (m * n = 56) := by
  sorry

end product_of_consecutive_integers_between_sqrt_50_l305_305325


namespace cosine_expression_rewrite_l305_305290

theorem cosine_expression_rewrite (x : ℝ) :
  ∃ a b c d : ℕ, 
    a * (Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x)) = 
    Real.cos (2 * x) + Real.cos (6 * x) + Real.cos (14 * x) + Real.cos (18 * x) 
    ∧ a + b + c + d = 22 := sorry

end cosine_expression_rewrite_l305_305290


namespace remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one_l305_305477

theorem remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one :
  ((x - 1) ^ 2028) % (x^2 - x + 1) = 1 :=
by
  sorry

end remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one_l305_305477


namespace forest_enclosure_l305_305733

theorem forest_enclosure
  (n : ℕ)
  (a : Fin n → ℝ)
  (h_a_lt_100 : ∀ i, a i < 100)
  (d : Fin n → Fin n → ℝ)
  (h_dist : ∀ i j, i < j → d i j ≤ (a i) - (a j)) :
  ∃ f : ℝ, f = 200 :=
by
  -- The proof goes here
  sorry

end forest_enclosure_l305_305733


namespace find_two_primes_l305_305090

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m ≠ n → n % m ≠ 0

-- Prove the existence of two specific prime numbers with the desired properties
theorem find_two_primes :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p = 2 ∧ q = 5 ∧ is_prime (p + q) ∧ is_prime (q - p) :=
by
  exists 2
  exists 5
  repeat {split}
  sorry

end find_two_primes_l305_305090


namespace parabola_focus_to_directrix_distance_correct_l305_305157

def parabola_focus_to_directrix_distance (a : ℕ) (y x : ℝ) : Prop :=
  y^2 = 2 * x → a = 2 →  1 = 1

theorem parabola_focus_to_directrix_distance_correct :
  ∀ (a : ℕ) (y x : ℝ), parabola_focus_to_directrix_distance a y x :=
by
  unfold parabola_focus_to_directrix_distance
  intros
  sorry

end parabola_focus_to_directrix_distance_correct_l305_305157


namespace sequence_a_5_l305_305700

noncomputable section

-- Definition of the sequence
def a : ℕ → ℕ
| 0       => 1
| 1       => 2
| (n + 2) => a (n + 1) + a n

-- Statement to prove that a 4 = 8 (in Lean, the sequence is zero-indexed, so a 4 is a_5)
theorem sequence_a_5 : a 4 = 8 :=
  by
    sorry

end sequence_a_5_l305_305700


namespace prime_square_mod_12_l305_305759

theorem prime_square_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_ne2 : p ≠ 2) (h_ne3 : p ≠ 3) :
    (∃ n : ℤ, p = 6 * n + 1 ∨ p = 6 * n + 5) → (p^2 % 12 = 1) := by
  sorry

end prime_square_mod_12_l305_305759


namespace reciprocal_of_minus_one_half_l305_305168

theorem reciprocal_of_minus_one_half : (1 / (-1 / 2)) = -2 := 
by sorry

end reciprocal_of_minus_one_half_l305_305168


namespace company_fund_initial_amount_l305_305467

theorem company_fund_initial_amount (n : ℕ) (fund_initial : ℤ) 
  (h1 : ∃ n, fund_initial = 60 * n - 10)
  (h2 : ∃ n, 55 * n + 120 = fund_initial + 130)
  : fund_initial = 1550 := 
sorry

end company_fund_initial_amount_l305_305467


namespace range_of_ω_l305_305854

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ :=
  Real.cos (ω * x + ϕ)

theorem range_of_ω :
  ∀ (ω : ℝ) (ϕ : ℝ),
    (0 < ω) →
    (-π ≤ ϕ) →
    (ϕ ≤ 0) →
    (∀ x, f x ω ϕ = -f (-x) ω ϕ) →
    (∀ x1 x2, (x1 < x2) → (-π/4 ≤ x1 ∧ x1 ≤ 3*π/16) ∧ (-π/4 ≤ x2 ∧ x2 ≤ 3*π/16) → f x1 ω ϕ ≤ f x2 ω ϕ) →
    (0 < ω ∧ ω ≤ 2) :=
by
  sorry

end range_of_ω_l305_305854


namespace costume_total_cost_l305_305495

variable (friends : ℕ) (cost_per_costume : ℕ) 

theorem costume_total_cost (h1 : friends = 8) (h2 : cost_per_costume = 5) : friends * cost_per_costume = 40 :=
by {
  sorry -- We omit the proof, as instructed.
}

end costume_total_cost_l305_305495


namespace isosceles_triangle_perimeter_l305_305602

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 6) (h2 : b = 13) 
  (triangle_inequality : b + b > a) : 
  (2 * b + a) = 32 := by
  sorry

end isosceles_triangle_perimeter_l305_305602


namespace no_solution_l305_305691

theorem no_solution (x : ℝ) : ¬ (3 * x^2 + 9 * x ≤ -12) :=
sorry

end no_solution_l305_305691


namespace sqrt_ab_is_integer_l305_305890

theorem sqrt_ab_is_integer
  (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n)
  (h_eq : a * (b^2 + n^2) = b * (a^2 + n^2)) :
  ∃ k : ℕ, k * k = a * b :=
by
  sorry

end sqrt_ab_is_integer_l305_305890


namespace max_side_of_triangle_exists_max_side_of_elevent_l305_305973

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l305_305973


namespace smallest_composite_no_prime_factors_below_20_l305_305683

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l305_305683


namespace max_side_length_l305_305962

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l305_305962


namespace sum_of_squared_residuals_l305_305730

theorem sum_of_squared_residuals (S : ℝ) (r : ℝ) (hS : S = 100) (hr : r = 0.818) : 
    S * (1 - r^2) = 33.0876 :=
by
  rw [hS, hr]
  sorry

end sum_of_squared_residuals_l305_305730


namespace train_crossing_time_l305_305946

def train_length : ℕ := 320
def train_speed_kmh : ℕ := 72
def kmh_to_ms (v : ℕ) : ℕ := v * 1000 / 3600
def train_speed_ms : ℕ := kmh_to_ms train_speed_kmh
def crossing_time (length : ℕ) (speed : ℕ) : ℕ := length / speed

theorem train_crossing_time : crossing_time train_length train_speed_ms = 16 := 
by {
  sorry
}

end train_crossing_time_l305_305946


namespace total_investment_is_correct_l305_305365

-- Define principal, rate, and number of years
def principal : ℝ := 8000
def rate : ℝ := 0.04
def years : ℕ := 10

-- Define the formula for compound interest
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- State the theorem
theorem total_investment_is_correct :
  compound_interest principal rate years = 11842 :=
by
  sorry

end total_investment_is_correct_l305_305365


namespace unique_positive_integer_k_for_rational_solutions_l305_305518

theorem unique_positive_integer_k_for_rational_solutions :
  ∃ (k : ℕ), (k > 0) ∧ (∀ (x : ℤ), x * x = 256 - 4 * k * k → x = 8) ∧ (k = 7) :=
by
  sorry

end unique_positive_integer_k_for_rational_solutions_l305_305518


namespace study_tour_part1_l305_305608

theorem study_tour_part1 (x y : ℕ) 
  (h1 : 45 * y + 15 = x) 
  (h2 : 60 * (y - 3) = x) : 
  x = 600 ∧ y = 13 :=
by sorry

end study_tour_part1_l305_305608


namespace stocks_worth_at_year_end_l305_305722

-- Definitions for initial investments
def initial_bonus : ℝ := 900
def investment_A : ℝ := initial_bonus / 3
def investment_B : ℝ := initial_bonus / 3
def investment_C : ℝ := initial_bonus / 3

-- Definitions for the value changes after one year
def value_A_after_one_year : ℝ := 2 * investment_A
def value_B_after_one_year : ℝ := 2 * investment_B
def value_C_after_one_year : ℝ := investment_C / 2

-- Total value after one year
def total_value_after_one_year : ℝ := value_A_after_one_year + value_B_after_one_year + value_C_after_one_year

-- Theorem to prove the total value of stocks at the end of the year
theorem stocks_worth_at_year_end : total_value_after_one_year = 1350 := by
  sorry

end stocks_worth_at_year_end_l305_305722


namespace two_times_koi_minus_X_is_64_l305_305476

-- Definitions based on the conditions
def n : ℕ := 39
def X : ℕ := 14

-- Main proof statement
theorem two_times_koi_minus_X_is_64 : 2 * n - X = 64 :=
by
  sorry

end two_times_koi_minus_X_is_64_l305_305476


namespace smallest_composite_no_prime_factors_less_than_20_l305_305661

def isComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n
def hasNoPrimeFactorsLessThan (n minPrime : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p < minPrime → ¬(p ∣ n)

theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, isComposite n ∧ hasNoPrimeFactorsLessThan n 20 ∧ ∀ m : ℕ, isComposite m ∧ hasNoPrimeFactorsLessThan m 20 → 529 ≤ m :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l305_305661


namespace find_x_l305_305123

-- Defining the conditions
def angle_PQR : ℝ := 180
def angle_PQS : ℝ := 125
def angle_QSR (x : ℝ) : ℝ := x
def SQ_eq_SR : Prop := true -- Assuming an isosceles triangle where SQ = SR.

-- The theorem to be proved
theorem find_x (x : ℝ) :
  angle_PQR = 180 → angle_PQS = 125 → SQ_eq_SR → angle_QSR x = 70 :=
by
  intros _ _ _
  sorry

end find_x_l305_305123


namespace three_digit_number_proof_l305_305122

noncomputable def is_prime (n : ℕ) : Prop := (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem three_digit_number_proof (H T U : ℕ) (h1 : H = 2 * T)
  (h2 : U = 2 * T^3)
  (h3 : is_prime (H + T + U))
  (h_digits : H < 10 ∧ T < 10 ∧ U < 10)
  (h_nonzero : T > 0) : H * 100 + T * 10 + U = 212 := 
by
  sorry

end three_digit_number_proof_l305_305122


namespace smallest_composite_no_prime_factors_less_than_20_l305_305685

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l305_305685


namespace train_stops_one_minute_per_hour_l305_305796

theorem train_stops_one_minute_per_hour (D : ℝ) (h1 : D / 400 = T₁) (h2 : D / 360 = T₂) : 
  (T₂ - T₁) * 60 = 1 :=
by
  sorry

end train_stops_one_minute_per_hour_l305_305796


namespace james_points_l305_305437

theorem james_points (x : ℕ) :
  13 * 3 + 20 * x = 79 → x = 2 :=
by
  sorry

end james_points_l305_305437


namespace average_age_of_guardians_and_fourth_graders_l305_305765

theorem average_age_of_guardians_and_fourth_graders (num_fourth_graders num_guardians : ℕ)
  (avg_age_fourth_graders avg_age_guardians : ℕ)
  (h1 : num_fourth_graders = 40)
  (h2 : avg_age_fourth_graders = 10)
  (h3 : num_guardians = 60)
  (h4 : avg_age_guardians = 35)
  : (num_fourth_graders * avg_age_fourth_graders + num_guardians * avg_age_guardians) / (num_fourth_graders + num_guardians) = 25 :=
by
  sorry

end average_age_of_guardians_and_fourth_graders_l305_305765


namespace max_side_of_triangle_exists_max_side_of_elevent_l305_305982

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l305_305982


namespace product_of_consecutive_integers_sqrt_50_l305_305319

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (√50 ≥ m) ∧ (√50 < n) ∧ (m * n = 56) :=
by
  use 7, 8
  split
  exact Nat.lt_succ_self 7
  split
  norm_num
  split
  norm_num
  norm_num

end product_of_consecutive_integers_sqrt_50_l305_305319


namespace goose_eggs_calculation_l305_305136

noncomputable def goose_eggs_total (E : ℕ) : Prop :=
  let hatched := (2/3) * E
  let survived_first_month := (3/4) * hatched
  let survived_first_year := (2/5) * survived_first_month
  survived_first_year = 110

theorem goose_eggs_calculation :
  goose_eggs_total 3300 :=
by
  have h1 : (2 : ℝ) / (3 : ℝ) ≠ 0 := by norm_num
  have h2 : (3 : ℝ) / (4 : ℝ) ≠ 0 := by norm_num
  have h3 : (2 : ℝ) / (5 : ℝ) ≠ 0 := by norm_num
  sorry

end goose_eggs_calculation_l305_305136


namespace arc_length_150_deg_max_area_sector_l305_305102

noncomputable def alpha := 150 * (Real.pi / 180)
noncomputable def r := 6
noncomputable def perimeter := 24

-- 1. Proving the arc length when α = 150° and r = 6
theorem arc_length_150_deg : alpha * r = 5 * Real.pi := by
  sorry

-- 2. Proving the maximum area and corresponding alpha given the perimeter of 24
theorem max_area_sector : ∃ (α : ℝ), α = 2 ∧ (1 / 2) * ((perimeter - 2 * r) * r) = 36 := by
  sorry

end arc_length_150_deg_max_area_sector_l305_305102


namespace find_speed_of_man_in_still_water_l305_305195

noncomputable def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
  (v_m + v_s) * 3 = 42 ∧ (v_m - v_s) * 3 = 18

theorem find_speed_of_man_in_still_water (v_s : ℝ) : ∃ v_m : ℝ, speed_of_man_in_still_water v_m v_s ∧ v_m = 10 :=
by
  sorry

end find_speed_of_man_in_still_water_l305_305195


namespace smallest_fraction_l305_305450

theorem smallest_fraction (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100) (eqn : y^2 - 1 = a^2 * (x^2 - 1)) :
  2 ≤ a / x :=
sorry

end smallest_fraction_l305_305450


namespace differential_savings_l305_305799

def annual_income_before_tax : ℝ := 42400
def initial_tax_rate : ℝ := 0.42
def new_tax_rate : ℝ := 0.32

theorem differential_savings :
  annual_income_before_tax * initial_tax_rate - annual_income_before_tax * new_tax_rate = 4240 :=
by
  sorry

end differential_savings_l305_305799


namespace profit_in_2004_correct_l305_305931

-- We define the conditions as given in the problem
def annual_profit_2002 : ℝ := 10
def annual_growth_rate (p : ℝ) : ℝ := p

-- The expression for the annual profit in 2004 given the above conditions
def annual_profit_2004 (p : ℝ) : ℝ := annual_profit_2002 * (1 + p) * (1 + p)

-- The theorem to prove that the computed annual profit in 2004 matches the expected answer
theorem profit_in_2004_correct (p : ℝ) :
  annual_profit_2004 p = 10 * (1 + p)^2 := 
by 
  sorry

end profit_in_2004_correct_l305_305931


namespace find_x_plus_y_l305_305243

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2010) (h2 : x + 2010 * Real.sin y = 2009) (h3 : Real.pi / 2 ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2011 + Real.pi :=
sorry

end find_x_plus_y_l305_305243


namespace smallest_composite_no_prime_factors_less_20_l305_305664

def is_prime (n : ℕ) : Prop := nat.prime n

def has_prime_factors_greater_than (n : ℕ) (k : ℕ) : Prop :=
  ∀ p, is_prime p → p ∣ n → p > k

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

theorem smallest_composite_no_prime_factors_less_20 :
  ∃ n : ℕ,
    is_composite n ∧ 
    has_prime_factors_greater_than n 19 ∧ 
    (∀ m : ℕ, is_composite m ∧ has_prime_factors_greater_than m 19 → n ≤ m) ∧ 
    n = 667 :=
sorry

end smallest_composite_no_prime_factors_less_20_l305_305664


namespace percentage_increase_second_year_l305_305182

theorem percentage_increase_second_year :
  let initial_deposit : ℤ := 1000
  let balance_first_year : ℤ := 1100
  let total_balance_two_years : ℤ := 1320
  let percent_increase_first_year : ℚ := ((balance_first_year - initial_deposit) / initial_deposit) * 100
  let percent_increase_total : ℚ := ((total_balance_two_years - initial_deposit) / initial_deposit) * 100
  let increase_second_year : ℤ := total_balance_two_years - balance_first_year
  let percent_increase_second_year : ℚ := (increase_second_year / balance_first_year) * 100
  percent_increase_first_year = 10 ∧
  percent_increase_total = 32 ∧
  increase_second_year = 220 → 
  percent_increase_second_year = 20 := by
  intros initial_deposit balance_first_year total_balance_two_years percent_increase_first_year
         percent_increase_total increase_second_year percent_increase_second_year
  sorry

end percentage_increase_second_year_l305_305182


namespace alex_score_correct_l305_305278

-- Conditions of the problem
def num_students := 20
def average_first_19 := 78
def new_average := 79

-- Alex's score calculation
def alex_score : ℕ :=
  let total_score_first_19 := 19 * average_first_19
  let total_score_all := num_students * new_average
  total_score_all - total_score_first_19

-- Problem statement: Prove Alex's score is 98
theorem alex_score_correct : alex_score = 98 := by
  sorry

end alex_score_correct_l305_305278


namespace cars_to_sell_l305_305055

theorem cars_to_sell (n : ℕ) 
  (h1 : ∀ c, c ∈ {c' : ℕ | c' ≤ n} → ∃ m, m = 3)
  (h2 : ∀ c, c ∈ {c' : ℕ | c' ≤ n} → c ∈ {c' : ℕ | c' < 3})
  (h3 : 15 * 3 = 45)
  (h4 : ∀ n, n * 3 = 45 → n = 15):
  n = 15 := 
  by
    have n_eq: n * 3 = 45 := sorry
    exact h4 n n_eq

end cars_to_sell_l305_305055


namespace distance_between_points_l305_305255

theorem distance_between_points (x : ℝ) :
  let M := (-1, 4)
  let N := (x, 4)
  dist (M, N) = 5 →
  (x = -6 ∨ x = 4) := sorry

end distance_between_points_l305_305255


namespace income_recording_l305_305435

theorem income_recording (exp_200 : Int := -200) (income_60 : Int := 60) : exp_200 = -200 → income_60 = 60 →
  (income_60 > 0) :=
by
  intro h_exp h_income
  sorry

end income_recording_l305_305435


namespace average_without_ivan_l305_305230

theorem average_without_ivan
  (total_friends : ℕ := 5)
  (avg_all : ℝ := 55)
  (ivan_amount : ℝ := 43)
  (remaining_friends : ℕ := total_friends - 1)
  (total_amount : ℝ := total_friends * avg_all)
  (remaining_amount : ℝ := total_amount - ivan_amount)
  (new_avg : ℝ := remaining_amount / remaining_friends) :
  new_avg = 58 := 
sorry

end average_without_ivan_l305_305230


namespace sum_of_prime_factors_l305_305918

theorem sum_of_prime_factors (x : ℕ) (h1 : x = 2^10 - 1) 
  (h2 : 2^10 - 1 = (2^5 + 1) * (2^5 - 1)) 
  (h3 : 2^5 - 1 = 31) 
  (h4 : 2^5 + 1 = 33) 
  (h5 : 33 = 3 * 11) : 
  (31 + 3 + 11 = 45) := 
  sorry

end sum_of_prime_factors_l305_305918


namespace complete_the_square_k_l305_305115

theorem complete_the_square_k (x : ℝ) : ∃ k : ℝ, (∃ a h: ℝ, (a = 1) ∧ (h = 7 / 2) ∧ (x^2 - 7*x = a * (x - h)^2 + k)) → k = -49 / 4 :=
by
  sorry

end complete_the_square_k_l305_305115


namespace sum_of_g_is_zero_l305_305269

def g (x : ℝ) : ℝ := x^3 * (1 - x)^3

theorem sum_of_g_is_zero :
  (Finset.range 2022).sum (λ k => (-1)^(k + 1) * g ((k + 1 : ℝ) / 2023)) = 0 :=
by
  sorry

end sum_of_g_is_zero_l305_305269


namespace no_negative_roots_l305_305391

theorem no_negative_roots (x : ℝ) (h : x < 0) : x^4 - 4*x^3 - 6*x^2 - 3*x + 9 ≠ 0 :=
by sorry

end no_negative_roots_l305_305391


namespace maximize_prob_l305_305288

-- Define the probability of correctly answering each question
def prob_A : ℝ := 0.6
def prob_B : ℝ := 0.8
def prob_C : ℝ := 0.5

-- Define the probability of getting two questions correct in a row for each order
def prob_A_first : ℝ := (prob_A * prob_B * (1 - prob_C) + (1 - prob_A) * prob_B * prob_C) +
                        (prob_A * prob_C * (1 - prob_B) + (1 - prob_A) * prob_C * prob_B)
def prob_B_first : ℝ := (prob_B * prob_A * (1 - prob_C) + (1 - prob_B) * prob_A * prob_C) +
                        (prob_B * prob_C * (1 - prob_A) + (1 - prob_B) * prob_C * prob_A)
def prob_C_first : ℝ := (prob_C * prob_A * (1 - prob_B) + (1 - prob_C) * prob_A * prob_B) +
                        (prob_C * prob_B * (1 - prob_A) + (1 - prob_C) * prob_B * prob_A)

-- Prove that the maximum probability is obtained when question C is answered first
theorem maximize_prob : prob_C_first > prob_A_first ∧ prob_C_first > prob_B_first :=
by
  -- Add the proof details here
  sorry

end maximize_prob_l305_305288


namespace shaniqua_earnings_l305_305892

noncomputable def shaniqua_total_earnings : ℕ :=
  let haircut_rate := 12
  let style_rate := 25
  let coloring_rate := 35
  let treatment_rate := 50
  let haircuts := 8
  let styles := 5
  let colorings := 10
  let treatments := 6
  (haircuts * haircut_rate) +
  (styles * style_rate) +
  (colorings * coloring_rate) +
  (treatments * treatment_rate)

theorem shaniqua_earnings : shaniqua_total_earnings = 871 := by
  sorry

end shaniqua_earnings_l305_305892


namespace checkered_rectangles_containing_one_gray_cell_l305_305859

theorem checkered_rectangles_containing_one_gray_cell 
  (num_gray_cells : ℕ) 
  (num_blue_cells : ℕ) 
  (num_red_cells : ℕ)
  (blue_containing_rectangles : ℕ) 
  (red_containing_rectangles : ℕ) :
  num_gray_cells = 40 →
  num_blue_cells = 36 →
  num_red_cells = 4 →
  blue_containing_rectangles = 4 →
  red_containing_rectangles = 8 →
  num_blue_cells * blue_containing_rectangles + num_red_cells * red_containing_rectangles = 176 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end checkered_rectangles_containing_one_gray_cell_l305_305859


namespace M_diff_N_l305_305442

def A : Set ℝ := sorry
def B : Set ℝ := sorry

def M := {x : ℝ | -3 ≤ x ∧ x ≤ 1}
def N := {y : ℝ | ∃ x : ℝ, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

-- Definition of set subtraction
def set_diff (A B : Set ℝ) := {x : ℝ | x ∈ A ∧ x ∉ B}

-- Given problem statement
theorem M_diff_N : set_diff M N = {x : ℝ | -3 ≤ x ∧ x < 0} := 
by
  sorry

end M_diff_N_l305_305442


namespace no_int_solutions_except_zero_l305_305143

theorem no_int_solutions_except_zero 
  (a b c n : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
by
  sorry

end no_int_solutions_except_zero_l305_305143


namespace complete_square_k_value_l305_305117

noncomputable def complete_square_form (x : ℝ) : ℝ := x^2 - 7 * x

theorem complete_square_k_value : ∃ a h k : ℝ, complete_square_form x = a * (x - h)^2 + k ∧ k = -49 / 4 :=
by
  use [1, 7/2, -49/4]
  -- This proof step will establish the relationships and the equality
  sorry

end complete_square_k_value_l305_305117


namespace maxwell_walking_speed_l305_305135

variable (distance : ℕ) (brad_speed : ℕ) (maxwell_time : ℕ) (brad_time : ℕ) (maxwell_speed : ℕ)

-- Given conditions
def conditions := distance = 54 ∧ brad_speed = 6 ∧ maxwell_time = 6 ∧ brad_time = 5

-- Problem statement
theorem maxwell_walking_speed (h : conditions distance brad_speed maxwell_time brad_time) : maxwell_speed = 4 := sorry

end maxwell_walking_speed_l305_305135


namespace polynomial_div_6_l305_305758

theorem polynomial_div_6 (n : ℕ) : 6 ∣ (2 * n ^ 3 + 9 * n ^ 2 + 13 * n) := 
sorry

end polynomial_div_6_l305_305758


namespace digit_to_make_multiple_of_5_l305_305922

theorem digit_to_make_multiple_of_5 (d : ℕ) (h : 0 ≤ d ∧ d ≤ 9) 
  (N := 71360 + d) : (N % 5 = 0) → (d = 0 ∨ d = 5) :=
by
  sorry

end digit_to_make_multiple_of_5_l305_305922


namespace difference_of_sides_l305_305772

-- Definitions based on conditions
def smaller_square_side (s : ℝ) := s
def larger_square_side (S s : ℝ) (h : (S^2 : ℝ) = 4 * s^2) := S

-- Theorem statement based on the proof problem
theorem difference_of_sides (s S : ℝ) (h : (S^2 : ℝ) = 4 * s^2) : S - s = s := 
by
  sorry

end difference_of_sides_l305_305772


namespace children_ages_l305_305936

-- Define the ages of the four children
variable (a b c d : ℕ)

-- Define the conditions
axiom h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
axiom h2 : a + b + c + d = 31
axiom h3 : (a - 4) + (b - 4) + (c - 4) + (d - 4) = 16
axiom h4 : (a - 7) + (b - 7) + (c - 7) + (d - 7) = 8
axiom h5 : (a - 11) + (b - 11) + (c - 11) + (d - 11) = 1
noncomputable def ages : ℕ × ℕ × ℕ × ℕ := (12, 10, 6, 3)

-- The theorem to prove
theorem children_ages (h1 : a = 12) (h2 : b = 10) (h3 : c = 6) (h4 : d = 3) : a = 12 ∧ b = 10 ∧ c = 6 ∧ d = 3 :=
by sorry

end children_ages_l305_305936


namespace carol_ate_12_cakes_l305_305834

-- Definitions for conditions
def cakes_per_day : ℕ := 10
def days_baking : ℕ := 5
def cans_per_cake : ℕ := 2
def cans_for_remaining_cakes : ℕ := 76

-- Total cakes baked by Sara
def total_cakes_baked (cakes_per_day days_baking : ℕ) : ℕ :=
  cakes_per_day * days_baking

-- Remaining cakes based on frosting cans needed
def remaining_cakes (cans_for_remaining_cakes cans_per_cake : ℕ) : ℕ :=
  cans_for_remaining_cakes / cans_per_cake

-- Cakes Carol ate
def cakes_carol_ate (total_cakes remaining_cakes : ℕ) : ℕ :=
  total_cakes - remaining_cakes

-- Theorem statement
theorem carol_ate_12_cakes :
  cakes_carol_ate (total_cakes_baked cakes_per_day days_baking) (remaining_cakes cans_for_remaining_cakes cans_per_cake) = 12 :=
by
  sorry

end carol_ate_12_cakes_l305_305834


namespace sum_of_squares_geometric_progression_theorem_l305_305029

noncomputable def sum_of_squares_geometric_progression (a₁ q : ℝ) (S₁ S₂ : ℝ)
  (h_q : abs q < 1)
  (h_S₁ : S₁ = a₁ / (1 - q))
  (h_S₂ : S₂ = a₁ / (1 + q)) : ℝ :=
  S₁ * S₂

theorem sum_of_squares_geometric_progression_theorem
  (a₁ q S₁ S₂ : ℝ)
  (h_q : abs q < 1)
  (h_S₁ : S₁ = a₁ / (1 - q))
  (h_S₂ : S₂ = a₁ / (1 + q)) :
  sum_of_squares_geometric_progression a₁ q S₁ S₂ h_q h_S₁ h_S₂ = S₁ * S₂ := sorry

end sum_of_squares_geometric_progression_theorem_l305_305029


namespace sum_of_variables_l305_305627

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_of_variables (x y z : ℝ) :
  log 2 (log 3 (log 4 x)) = 0 ∧ log 3 (log 4 (log 2 y)) = 0 ∧ log 4 (log 2 (log 3 z)) = 0 →
  x + y + z = 89 :=
by
  sorry

end sum_of_variables_l305_305627


namespace find_a2_l305_305531

noncomputable def geometric_sequence (n : ℕ) : ℝ := sorry -- Define the geometric sequence

variable (a1 : ℝ) (a3a5_eq : ℝ) -- Variables for given conditions

-- Main theorem statement
theorem find_a2 (h_geo : ∀ n, geometric_sequence n = a1 * (2 : ℝ) ^ (n - 1))
  (h_a1 : a1 = 1 / 4)
  (h_a3a5 : (geometric_sequence 3) * (geometric_sequence 5) = 4 * (geometric_sequence 4 - 1)) :
  geometric_sequence 2 = 1 / 2 :=
sorry  -- Proof is omitted

end find_a2_l305_305531


namespace cost_of_pears_l305_305646

theorem cost_of_pears (P : ℕ)
  (apples_cost : ℕ := 40)
  (dozens : ℕ := 14)
  (total_cost : ℕ := 1260)
  (h_p : dozens * P + dozens * apples_cost = total_cost) :
  P = 50 :=
by
  sorry

end cost_of_pears_l305_305646


namespace smallest_composite_no_prime_lt_20_l305_305676

theorem smallest_composite_no_prime_lt_20 :
  ∀ (n : ℕ), (prime_factors n ∩ { p | p < 20 } = ∅) ∧ ¬prime n → n ≥ 529 := 
by
  sorry

end smallest_composite_no_prime_lt_20_l305_305676


namespace solve_equation_l305_305232

noncomputable def min (a b : ℚ) : ℚ := if a ≤ b then a else b

theorem solve_equation : (∃ x : ℚ, (min x (-x) = 3 * x + 4) ∧ x = -2) :=
by
  use -2
  sorry

end solve_equation_l305_305232


namespace B_Bons_wins_probability_l305_305158

theorem B_Bons_wins_probability :
  let roll_six := (1 : ℚ) / 6
  let not_roll_six := (5 : ℚ) / 6
  let p := (5 : ℚ) / 11
  p = (5 / 36) + (25 / 36) * p :=
by
  sorry

end B_Bons_wins_probability_l305_305158


namespace solve_quadratic_l305_305582

theorem solve_quadratic : ∀ x : ℝ, 3 * x^2 - 2 * x - 1 = 0 ↔ (x = -1/3 ∨ x = 1) := by
  intro x
  sorry

end solve_quadratic_l305_305582


namespace scientific_notation_of_274M_l305_305392

theorem scientific_notation_of_274M :
  274000000 = 2.74 * 10^8 := 
by 
  sorry

end scientific_notation_of_274M_l305_305392


namespace find_max_side_length_l305_305956

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l305_305956


namespace point_on_xaxis_equidistant_l305_305209

theorem point_on_xaxis_equidistant :
  ∃ (A : ℝ × ℝ), A.2 = 0 ∧ 
                  dist A (-3, 2) = dist A (4, -5) ∧ 
                  A = (2, 0) :=
by
  sorry

end point_on_xaxis_equidistant_l305_305209


namespace find_x_l305_305014

theorem find_x : ∃ x : ℤ, (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 4 ∧ x = 28 := 
by sorry

end find_x_l305_305014


namespace max_set_size_divisible_diff_l305_305844

theorem max_set_size_divisible_diff (S : Finset ℕ) (h1 : ∀ x ∈ S, ∀ y ∈ S, x ≠ y → (5 ∣ (x - y) ∨ 25 ∣ (x - y))) : S.card ≤ 25 :=
sorry

end max_set_size_divisible_diff_l305_305844


namespace remainder_when_4x_div_7_l305_305043

theorem remainder_when_4x_div_7 (x : ℤ) (h : x % 7 = 5) : (4 * x) % 7 = 6 :=
by
  sorry

end remainder_when_4x_div_7_l305_305043


namespace graph_not_pass_through_second_quadrant_l305_305293

theorem graph_not_pass_through_second_quadrant 
    (k : ℝ) (b : ℝ) (h1 : k = 1) (h2 : b = -2) : 
    ¬ ∃ (x y : ℝ), y = k * x + b ∧ x < 0 ∧ y > 0 := 
by
  sorry

end graph_not_pass_through_second_quadrant_l305_305293


namespace fourth_root_12960000_eq_60_l305_305512

theorem fourth_root_12960000_eq_60 :
  (6^4 = 1296) →
  (10^4 = 10000) →
  (60^4 = 12960000) →
  (Real.sqrt (Real.sqrt 12960000) = 60) := 
by
  intros h1 h2 h3
  sorry

end fourth_root_12960000_eq_60_l305_305512


namespace probability_joint_independent_events_conditional_probability_independent_events_l305_305408

theorem probability_joint_independent_events
  (pa pb pc : ℝ)
  (habc : a ∧ b ∧ c)
  (ind : independent a b c)
  (hpa : pa = 5/7)
  (hpb : pb = 2/5)
  (hpc : pc = 3/4) :
  Prob (a ∧ b ∧ c) = 3/14 :=
by
  sorry

theorem conditional_probability_independent_events
  (pa pb pc : ℝ)
  (habc : a ∧ b ∧ c)
  (ind : independent a b c)
  (hpa : pa = 5/7)
  (hpb : pb = 2/5)
  (hpc : pc = 3/4) :
  Prob (a ∧ b | c) = 2/7 :=
by
  sorry

end probability_joint_independent_events_conditional_probability_independent_events_l305_305408


namespace bridge_length_l305_305930

theorem bridge_length (train_length : ℕ) (crossing_time : ℕ) (train_speed_kmh : ℕ) :
  train_length = 500 → crossing_time = 45 → train_speed_kmh = 64 → 
  ∃ (bridge_length : ℝ), bridge_length = 300.1 :=
by
  intros h1 h2 h3
  have speed_mps := (train_speed_kmh * 1000) / 3600
  have total_distance := speed_mps * crossing_time
  have bridge_length_calculated := total_distance - train_length
  use bridge_length_calculated
  sorry

end bridge_length_l305_305930


namespace days_with_equal_sun_tue_l305_305808

theorem days_with_equal_sun_tue (days_in_month : ℕ) (weekdays : ℕ) (d1 d2 : ℕ) (h1 : days_in_month = 30)
  (h2 : weekdays = 7) (h3 : d1 = 4) (h4 : d2 = 2) :
  ∃ count, count = 3 := by
  sorry

end days_with_equal_sun_tue_l305_305808


namespace max_side_length_l305_305964

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l305_305964


namespace problem_I_problem_II_l305_305244

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x + (4 / x) - m| + m

-- Proof problem (I): When m = 0, find the minimum value of the function f(x).
theorem problem_I : ∀ x : ℝ, (f x 0) ≥ 4 := by
  sorry

-- Proof problem (II): If the function f(x) ≤ 5 for all x ∈ [1, 4], find the range of m.
theorem problem_II (m : ℝ) : (∀ x : ℝ, 1 ≤ x → x ≤ 4 → f x m ≤ 5) ↔ m ≤ 9 / 2 := by
  sorry

end problem_I_problem_II_l305_305244


namespace roommates_condition_l305_305261

def f (x : ℝ) := 3 * x ^ 2 + 5 * x - 1
def g (x : ℝ) := 2 * x ^ 2 - 3 * x + 5

theorem roommates_condition : f 3 = 2 * g 3 + 5 := 
by {
  sorry
}

end roommates_condition_l305_305261


namespace actual_distance_traveled_l305_305488

theorem actual_distance_traveled (D : ℕ) (h : D / 10 = (D + 20) / 15) : D = 40 := 
sorry

end actual_distance_traveled_l305_305488


namespace total_cost_l305_305939

-- Define the cost of a neutral pen and a pencil
variables (x y : ℝ)

-- The total cost of buying 5 neutral pens and 3 pencils
theorem total_cost (x y : ℝ) : 5 * x + 3 * y = 5 * x + 3 * y :=
by
  -- The statement is self-evident, hence can be written directly
  sorry

end total_cost_l305_305939


namespace Elle_practice_time_l305_305838

variable (x : ℕ)

theorem Elle_practice_time : 
  (5 * x) + (3 * x) = 240 → x = 30 :=
by
  intro h
  sorry

end Elle_practice_time_l305_305838


namespace inverse_variation_l305_305044

theorem inverse_variation (k : ℝ) : 
  (∀ (x y : ℝ), x * y^2 = k) → 
  (∀ (x y : ℝ), x = 1 → y = 2 → k = 4) → 
  (x = 0.1111111111111111) → 
  (y = 6) :=
by 
  -- Assume the given conditions
  intros h h0 hx
  -- Proof goes here...
  sorry

end inverse_variation_l305_305044


namespace range_of_x_l305_305185

theorem range_of_x (a : ℝ) (x : ℝ) (h0 : 0 ≤ a) (h1 : a ≤ 2) :
  a * x^2 + (a + 1) * x + 1 - (3 / 2) * a < 0 → -2 < x ∧ x < -1 :=
by
  sorry

end range_of_x_l305_305185


namespace pieces_given_by_brother_l305_305145

-- Given conditions
def original_pieces : ℕ := 18
def total_pieces_now : ℕ := 62

-- The statement to prove
theorem pieces_given_by_brother : total_pieces_now - original_pieces = 44 := by
  -- Starting with the given conditions
  unfold original_pieces total_pieces_now
  -- Place to insert the proof
  sorry

end pieces_given_by_brother_l305_305145


namespace min_valid_subset_card_eq_l305_305914

open Finset

def pairs (n : ℕ) : Finset (ℕ × ℕ) := 
  (range n).product (range n)

def valid_subset (X : Finset (ℕ × ℕ)) (n : ℕ) : Prop :=
  ∀ (seq : ℕ → ℕ), ∃ k, (seq k, seq (k+1)) ∈ X

theorem min_valid_subset_card_eq (n : ℕ) (h : n = 10) : 
  ∃ X : Finset (ℕ × ℕ), valid_subset X n ∧ X.card = 55 := 
by 
  sorry

end min_valid_subset_card_eq_l305_305914


namespace set_inter_compl_eq_l305_305274

def U := ℝ
def M : Set ℝ := { x | abs (x - 1/2) ≤ 5/2 }
def P : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def complement_U_M : Set ℝ := { x | x < -2 ∨ x > 3 }

theorem set_inter_compl_eq :
  (complement_U_M ∩ P) = { x | 3 < x ∧ x ≤ 4 } :=
sorry

end set_inter_compl_eq_l305_305274


namespace convex_polygon_sides_l305_305536

theorem convex_polygon_sides (n : ℕ) (h : ∀ angle, angle = 45 → angle * n = 360) : n = 8 :=
  sorry

end convex_polygon_sides_l305_305536


namespace solution_existence_l305_305618

def problem_statement : Prop :=
  ∃ x : ℝ, (0.38 * 80) - (0.12 * x) = 11.2 ∧ x = 160

theorem solution_existence : problem_statement :=
  sorry

end solution_existence_l305_305618


namespace height_difference_of_packings_l305_305920

theorem height_difference_of_packings :
  (let d := 12
   let n := 180
   let rowsA := n / 10
   let heightA := rowsA * d
   let height_of_hex_gap := (6 * Real.sqrt 3 : ℝ)
   let gaps := rowsA - 1
   let heightB := gaps * height_of_hex_gap + 2 * (d / 2)
   heightA - heightB) = 204 - 102 * Real.sqrt 3 :=
  sorry

end height_difference_of_packings_l305_305920


namespace larger_number_is_1617_l305_305591

-- Given conditions
variables (L S : ℤ)
axiom condition1 : L - S = 1515
axiom condition2 : L = 16 * S + 15

-- To prove
theorem larger_number_is_1617 : L = 1617 := by
  sorry

end larger_number_is_1617_l305_305591


namespace value_after_increase_l305_305940

def original_number : ℝ := 400
def percentage_increase : ℝ := 0.20

theorem value_after_increase : original_number * (1 + percentage_increase) = 480 := by
  sorry

end value_after_increase_l305_305940


namespace product_mod_10_l305_305924

theorem product_mod_10 (a b c : ℕ) (ha : a % 10 = 4) (hb : b % 10 = 5) (hc : c % 10 = 5) :
  (a * b * c) % 10 = 0 :=
sorry

end product_mod_10_l305_305924


namespace art_club_students_l305_305177

theorem art_club_students 
    (students artworks_per_student_per_quarter quarters_per_year artworks_in_two_years : ℕ) 
    (h1 : artworks_per_student_per_quarter = 2)
    (h2 : quarters_per_year = 4) 
    (h3 : artworks_in_two_years = 240) 
    (h4 : students * (artworks_per_student_per_quarter * quarters_per_year) * 2 = artworks_in_two_years) :
    students = 15 := 
by
    -- Given conditions for the problem
    sorry

end art_club_students_l305_305177


namespace kiki_scarves_count_l305_305262

variable (money : ℝ) (scarf_cost : ℝ) (hat_spending_ratio : ℝ) (scarves : ℕ) (hats : ℕ)

-- Condition: Kiki has $90.
axiom kiki_money : money = 90

-- Condition: Kiki spends 60% of her money on hats.
axiom kiki_hat_spending_ratio : hat_spending_ratio = 0.60

-- Condition: Each scarf costs $2.
axiom scarf_price : scarf_cost = 2

-- Condition: Kiki buys twice as many hats as scarves.
axiom hat_scarf_relationship : hats = 2 * scarves

theorem kiki_scarves_count 
  (kiki_money : money = 90)
  (kiki_hat_spending_ratio : hat_spending_ratio = 0.60)
  (scarf_price : scarf_cost = 2)
  (hat_scarf_relationship : hats = 2 * scarves)
  : scarves = 18 := 
sorry

end kiki_scarves_count_l305_305262


namespace arithmetic_sequence_geometric_sequence_sum_of_first_n_terms_l305_305242

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℤ := 3^n
noncomputable def c_n (n : ℕ) : ℤ := (-1 : ℤ)^n * a_n n * b_n n
noncomputable def S_n (n : ℕ) : ℝ := ∑ i in range (n + 1), c_n i

theorem arithmetic_sequence {d : ℤ} : (a_n 1 = 1) → (a_n 3 + a_n 4 = 12) → d = 2 → a_n n = 2n - 1 := sorry
theorem geometric_sequence {q : ℤ} : (b_n 1 = a_n 2) → (b_n 2 = a_n 5) → q = 3 → b_n n = 3^n := sorry
theorem sum_of_first_n_terms {n : ℕ} : S_n n = (3/8 : ℝ) - (8n - 1)/8 * (-3 : ℝ)^(n + 1) := sorry

end arithmetic_sequence_geometric_sequence_sum_of_first_n_terms_l305_305242


namespace min_value_expr_l305_305535

theorem min_value_expr (a d : ℝ) (b c : ℝ) (h_a : 0 ≤ a) (h_d : 0 ≤ d) (h_b : 0 < b) (h_c : 0 < c) (h : b + c ≥ a + d) :
  (b / (c + d) + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end min_value_expr_l305_305535


namespace hieu_catches_up_beatrice_in_5_minutes_probability_beatrice_hieu_same_place_range_of_x_for_meeting_probability_l305_305641

namespace CatchUpProblem

-- Part (a)
theorem hieu_catches_up_beatrice_in_5_minutes :
  ∀ (d_b_walked : ℕ) (relative_speed : ℕ) (catch_up_time : ℕ),
  d_b_walked = 5 / 6 ∧ relative_speed = 10 ∧ catch_up_time = 5 :=
sorry

-- Part (b)(i)
theorem probability_beatrice_hieu_same_place :
  ∀ (total_pairs : ℕ) (valid_pairs : ℕ) (probability : Rat),
  total_pairs = 3600 ∧ valid_pairs = 884 ∧ probability = 221 / 900 :=
sorry

-- Part (b)(ii)
theorem range_of_x_for_meeting_probability :
  ∀ (probability : Rat) (valid_pairs : ℕ) (total_pairs : ℕ) (lower_bound : ℕ) (upper_bound : ℕ),
  probability = 13 / 200 ∧ valid_pairs = 234 ∧ total_pairs = 3600 ∧ 
  lower_bound = 10 ∧ upper_bound = 120 / 11 :=
sorry

end CatchUpProblem

end hieu_catches_up_beatrice_in_5_minutes_probability_beatrice_hieu_same_place_range_of_x_for_meeting_probability_l305_305641


namespace frogs_moving_l305_305937

theorem frogs_moving (initial_frogs tadpoles mature_frogs pond_capacity frogs_to_move : ℕ)
  (h1 : initial_frogs = 5)
  (h2 : tadpoles = 3 * initial_frogs)
  (h3 : mature_frogs = (2 * tadpoles) / 3)
  (h4 : pond_capacity = 8)
  (h5 : frogs_to_move = (initial_frogs + mature_frogs) - pond_capacity) :
  frogs_to_move = 7 :=
by {
  sorry
}

end frogs_moving_l305_305937


namespace solution_to_equation_l305_305387

def star_operation (a b : ℝ) : ℝ := a^2 - 2 * a * b + b^2

theorem solution_to_equation : ∀ (x : ℝ), star_operation (x - 4) 1 = 0 → x = 5 :=
by
  intro x
  assume h
  -- Skipping the proof steps with sorry
  sorry

end solution_to_equation_l305_305387


namespace partition_sum_equal_l305_305827

open Set Finset

def isPermutationOfDigits (n : ℕ) : Prop :=
  ∀ {a b c d e : ℕ}, List.Perm [a, b, c, d, e] [1, 2, 3, 4, 5] → 
  (n = a * 10000 + b * 1000 + c * 100 + d * 10 + e)

theorem partition_sum_equal (S : Finset ℕ) :
  (∀ n ∈ S, isPermutationOfDigits n) →
  ∃ (A B : Finset ℕ), A ∪ B = S ∧ A ∩ B = ∅ ∧ 
  (∑ x in A, x^2) = (∑ x in B, x^2) :=
by
  sorry

end partition_sum_equal_l305_305827


namespace tan_alpha_sq_two_trigonometric_identity_l305_305393

-- Problem 1
theorem tan_alpha_sq_two (α : ℝ) (h : Real.tan α = Real.sqrt 2) :
  1 + Real.sin (2 * α) + (Real.cos α) ^ 2 = (4 + Real.sqrt 2) / 3 :=
by sorry

-- Problem 2
theorem trigonometric_identity : 
  (2 * Real.sin (50 * Real.pi / 180) + Real.sin (80 * Real.pi / 180) *
    (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180))) / 
  Real.sqrt (1 + Real.sin (100 * Real.pi / 180)) = 2 :=
by sorry

end tan_alpha_sq_two_trigonometric_identity_l305_305393


namespace geometric_sequence_expression_l305_305256

variable {a : ℕ → ℝ}

-- Define the geometric sequence property
def is_geometric (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_expression :
  is_geometric a q →
  a 3 = 2 →
  a 6 = 16 →
  ∀ n, a n = 2^(n-2) := by
  intros h_geom h_a3 h_a6
  sorry

end geometric_sequence_expression_l305_305256


namespace proof_statements_l305_305543

theorem proof_statements (m : ℝ) (x y : ℝ)
  (h1 : 2 * x + y = 4 - m)
  (h2 : x - 2 * y = 3 * m) :
  (m = 1 → (x = 9 / 5 ∧ y = -3 / 5)) ∧
  (3 * x - y = 4 + 2 * m) ∧
  ¬(∃ (m' : ℝ), (8 + m') / 5 < 0 ∧ (4 - 7 * m') / 5 < 0) :=
sorry

end proof_statements_l305_305543


namespace percentage_difference_l305_305923

theorem percentage_difference :
    let A := (40 / 100) * ((50 / 100) * 60)
    let B := (50 / 100) * ((60 / 100) * 70)
    (B - A) = 9 :=
by
    sorry

end percentage_difference_l305_305923


namespace fifth_term_sequence_l305_305650

theorem fifth_term_sequence : 
  (4 + 8 + 16 + 32 + 64) = 124 := 
by 
  sorry

end fifth_term_sequence_l305_305650


namespace recycling_program_earnings_l305_305439

-- Define conditions
def signup_earning : ℝ := 5.00
def referral_earning_tier1 : ℝ := 8.00
def referral_earning_tier2 : ℝ := 1.50
def friend_earning_signup : ℝ := 5.00
def friend_earning_tier2 : ℝ := 2.00

def initial_friend_count : ℕ := 5
def initial_friend_tier1_referrals_day1 : ℕ := 3
def initial_friend_tier1_referrals_week : ℕ := 2

def additional_friend_count : ℕ := 2
def additional_friend_tier1_referrals : ℕ := 1

-- Calculate Katrina's total earnings
def katrina_earnings : ℝ :=
  signup_earning +
  (initial_friend_count * referral_earning_tier1) +
  (initial_friend_count * initial_friend_tier1_referrals_day1 * referral_earning_tier2) +
  (initial_friend_count * initial_friend_tier1_referrals_week * referral_earning_tier2) +
  (additional_friend_count * referral_earning_tier1) +
  (additional_friend_count * additional_friend_tier1_referrals * referral_earning_tier2)

-- Calculate friends' total earnings
def friends_earnings : ℝ :=
  (initial_friend_count * friend_earning_signup) +
  (initial_friend_count * initial_friend_tier1_referrals_day1 * friend_earning_tier2) +
  (initial_friend_count * initial_friend_tier1_referrals_week * friend_earning_tier2) +
  (additional_friend_count * friend_earning_signup) +
  (additional_friend_count * additional_friend_tier1_referrals * friend_earning_tier2)

-- Calculate combined total earnings
def combined_earnings : ℝ := katrina_earnings + friends_earnings

-- The proof assertion
theorem recycling_program_earnings : combined_earnings = 190.50 :=
by sorry

end recycling_program_earnings_l305_305439


namespace value_of_I_l305_305493

variables (T H I S : ℤ)

theorem value_of_I :
  H = 10 →
  T + H + I + S = 50 →
  H + I + T = 35 →
  S + I + T = 40 →
  I = 15 :=
  by
  sorry

end value_of_I_l305_305493


namespace cost_of_eraser_carton_l305_305888

/-- An order consists of 100 cartons, including 20 cartons of pencils costing 6 dollars per carton. 
     The total cost of the order is 360 dollars. Prove that the cost of a carton of erasers is 3 dollars. -/
theorem cost_of_eraser_carton 
  (total_cartons : ℕ) 
  (pencil_cartons : ℕ) 
  (pencil_cost_per_carton : ℚ)
  (eraser_cost_per_carton : ℚ)
  (total_cost : ℚ)
  (total_pencil_cost : ℚ)
  (eraser_cartons : ℕ) 
  (remaining_cost : ℚ)
  (cost_per_eraser_carton : ℚ) 
  (h1 : total_cartons = 100)
  (h2 : pencil_cartons = 20)
  (h3 : pencil_cost_per_carton = 6)
  (h4 : total_cost = 360)
  (h5 : total_pencil_cost = (pencil_cartons * pencil_cost_per_carton))
  (h6 : eraser_cartons = (total_cartons - pencil_cartons))
  (h7 : remaining_cost = (total_cost - total_pencil_cost))
  (h8 : cost_per_eraser_carton = (remaining_cost / eraser_cartons)) 
  : eraser_cost_per_carton = 3 :=
by 
  -- Definition of each step is stated as hypotheses
  have h_total_pencil_cost : total_pencil_cost = (20 * 6) := by rw [h2, h3]; rfl
  have h_eraser_cartons : eraser_cartons = (100 - 20) := by rw [h1, h2]; rfl
  have h_remaining_cost : remaining_cost = (360 - 120) := by rw [h4, h_total_pencil_cost]; rfl
  have h_cost_per_eraser : cost_per_eraser_carton = (240 / 80) := 
    by rw [h8]; rw [h_remaining_cost, h_eraser_cartons]; rfl
  
  -- Final proof by asserting eraser_cost_per_carton is 3 
  have h_final : eraser_cost_per_carton = 3 := by 
    rw [h5, h6, h7, h8, h_total_pencil_cost, h_eraser_cartons, h_remaining_cost, h_cost_per_eraser]; exact rfl
  exact h_final

end cost_of_eraser_carton_l305_305888


namespace values_of_x_and_y_l305_305108

theorem values_of_x_and_y (x y : ℝ) (h1 : x - y > x + 1) (h2 : x + y < y - 2) : x < -2 ∧ y < -1 :=
by
  -- Proof goes here
  sorry

end values_of_x_and_y_l305_305108


namespace sqrt_50_between_consecutive_integers_product_l305_305301

theorem sqrt_50_between_consecutive_integers_product :
  ∃ (m n : ℕ), (m + 1 = n) ∧ (m * m < 50) ∧ (50 < n * n) ∧ (m * n = 56) :=
begin
  sorry
end

end sqrt_50_between_consecutive_integers_product_l305_305301


namespace problem_statement_l305_305413

noncomputable def U : Set Int := {-2, -1, 0, 1, 2}
noncomputable def A : Set Int := {x : Int | -2 ≤ x ∧ x < 0}
noncomputable def B : Set Int := {x : Int | (x = 0 ∨ x = 1)} -- since natural numbers typically include positive integers, adapting B contextually

theorem problem_statement : ((U \ A) ∩ B) = {0, 1} := by
  sorry

end problem_statement_l305_305413


namespace fraction_value_l305_305709

variable (u v w x : ℝ)

-- Conditions
def cond1 : Prop := u / v = 5
def cond2 : Prop := w / v = 3
def cond3 : Prop := w / x = 2 / 3

theorem fraction_value (h1 : cond1 u v) (h2 : cond2 w v) (h3 : cond3 w x) : x / u = 9 / 10 := 
by
  sorry

end fraction_value_l305_305709


namespace system1_solution_system2_solution_l305_305151

theorem system1_solution (x y : ℚ) :
  x + y = 4 ∧ 5 * (x - y) - 2 * (x + y) = -1 →
  x = 27 / 10 ∧ y = 13 / 10 := by
  sorry

theorem system2_solution (x y : ℚ) :
  (2 * (x - y) / 3) - ((x + y) / 4) = -1 / 12 ∧ 3 * (x + y) - 2 * (2 * x - y) = 3 →
  x = 2 ∧ y = 1 := by
  sorry

end system1_solution_system2_solution_l305_305151


namespace sum_infinite_series_l305_305078

theorem sum_infinite_series : ∑' k : ℕ, (k^2 : ℝ) / (3^k) = 7 / 8 :=
sorry

end sum_infinite_series_l305_305078


namespace sum_of_angles_in_segments_outside_pentagon_l305_305634

theorem sum_of_angles_in_segments_outside_pentagon 
  (α β γ δ ε : ℝ) 
  (hα : α = 0.5 * (360 - arc_BCDE))
  (hβ : β = 0.5 * (360 - arc_CDEA))
  (hγ : γ = 0.5 * (360 - arc_DEAB))
  (hδ : δ = 0.5 * (360 - arc_EABC))
  (hε : ε = 0.5 * (360 - arc_ABCD)) 
  (arc_BCDE arc_CDEA arc_DEAB arc_EABC arc_ABCD : ℝ) :
  α + β + γ + δ + ε = 720 := 
by 
  sorry

end sum_of_angles_in_segments_outside_pentagon_l305_305634


namespace correct_square_root_operation_l305_305188

theorem correct_square_root_operation : 
  (sqrt 4)^2 = 4 ∧ sqrt 4 ≠ 2 ∨ -2 ∧ sqrt ((-4)^2) ≠ -4 ∧ (-sqrt 4)^2 ≠ -4 :=
by
  have a : (sqrt 4)^2 = 4, from sorry,
  have b : sqrt 4 ≠ 2 ∨ -2, from sorry,
  have c : sqrt ((-4)^2) ≠ -4, from sorry,
  have d : (-sqrt 4)^2 ≠ -4, from sorry,
  exact ⟨a, b, c, d⟩

end correct_square_root_operation_l305_305188


namespace simplify_polynomial_l305_305705

open Nat

-- Define arithmetic sequence conditions and the polynomial structure
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def binomial (n k : ℕ) : ℕ := choose n k

noncomputable def p (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in range (n + 1), (a k) * (binomial n k) * (x^k) * ((1 - x)^(n - k))

-- The main theorem
theorem simplify_polynomial
  (a : ℕ → ℝ)
  (d : ℝ)
  (h : arithmetic_seq a d)
  (n : ℕ)
  (x : ℝ)
  : p a n x = a 0 + n * d * x :=
sorry

end simplify_polynomial_l305_305705


namespace original_square_side_length_l305_305137

theorem original_square_side_length :
  ∃ n k : ℕ, (n + k) * (n + k) - n * n = 47 ∧ k ≤ 5 ∧ k % 2 = 1 ∧ n = 23 :=
by
  sorry

end original_square_side_length_l305_305137


namespace rocket_coaster_total_cars_l305_305900

theorem rocket_coaster_total_cars (C_4 C_6 : ℕ) (h1 : C_4 = 9) (h2 : 4 * C_4 + 6 * C_6 = 72) :
  C_4 + C_6 = 15 :=
sorry

end rocket_coaster_total_cars_l305_305900


namespace sum_of_fraction_equiv_l305_305020

theorem sum_of_fraction_equiv : 
  let x := 3.714714714
  let num := 3711
  let denom := 999
  3711 + 999 = 4710 :=
by 
  sorry

end sum_of_fraction_equiv_l305_305020


namespace son_father_age_sum_l305_305048

theorem son_father_age_sum
    (S F : ℕ)
    (h1 : F - 6 = 3 * (S - 6))
    (h2 : F = 2 * S) :
    S + F = 36 :=
sorry

end son_father_age_sum_l305_305048


namespace maria_punch_l305_305880

variable (L S W : ℕ)

theorem maria_punch (h1 : S = 3 * L) (h2 : W = 3 * S) (h3 : L = 4) : W = 36 :=
by
  sorry

end maria_punch_l305_305880


namespace factorize_polynomial_l305_305622

theorem factorize_polynomial :
  (∀ x : ℤ, x^{15} + x^{10} + 1 = (x^2 + x + 1) * (x^{13} - x^{12} + x^{10} - x^9 + x^7 - x^6 + x^4 - x^3 + x) + 1) :=
by sorry

end factorize_polynomial_l305_305622


namespace donation_percentage_correct_l305_305498

noncomputable def percentage_donated_to_orphan_house (income remaining : ℝ) (given_to_children_percentage : ℝ) (given_to_wife_percentage : ℝ) (remaining_after_donation : ℝ)
    (before_donation_remaining : income * (1 - given_to_children_percentage / 100 - given_to_wife_percentage / 100) = remaining)
    (after_donation_remaining : remaining - remaining_after_donation * remaining = 500) : Prop :=
    100 * (remaining - 500) / remaining = 16.67

theorem donation_percentage_correct 
    (income : ℝ) 
    (child_percentage : ℝ := 10)
    (num_children : ℕ := 2)
    (wife_percentage : ℝ := 20)
    (final_amount : ℝ := 500)
    (income_value : income = 1000 ) : 
    percentage_donated_to_orphan_house income 
    (income * (1 - (child_percentage * num_children) / 100 - wife_percentage / 100)) 
    (child_percentage * num_children)
    wife_percentage 
    final_amount 
    sorry 
    sorry :=
sorry

end donation_percentage_correct_l305_305498


namespace marble_problem_l305_305359

theorem marble_problem : Nat.lcm (Nat.lcm (Nat.lcm 2 3) 5) 7 = 210 := by
  sorry

end marble_problem_l305_305359


namespace logarithm_simplification_l305_305148

theorem logarithm_simplification :
  (1 / (Real.log 3 / Real.log 12 + 1) + 1 / (Real.log 2 / Real.log 8 + 1) + 1 / (Real.log 7 / Real.log 9 + 1)) =
  1 - (Real.log 7 / Real.log 1008) :=
sorry

end logarithm_simplification_l305_305148


namespace smallest_composite_no_prime_factors_less_than_twenty_l305_305670

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_twenty :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
  ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  sorry

end smallest_composite_no_prime_factors_less_than_twenty_l305_305670


namespace max_side_length_l305_305961

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l305_305961


namespace no_solutions_to_inequality_l305_305690

theorem no_solutions_to_inequality (x : ℝ) : ¬(3 * x^2 + 9 * x + 12 ≤ 0) :=
by {
  intro h,
  -- Simplify the inequality by dividing each term by 3
  have h_simplified : x^2 + 3 * x + 4 ≤ 0 := by linarith,
  -- Compute the discriminant of the quadratic expression to show it's always positive
  let a := (1 : ℝ),
  let b := (3 : ℝ),
  let c := (4 : ℝ),
  let discriminant := b^2 - 4 * a * c,
  have h_discriminant : discriminant < 0 := by norm_num,
  -- Since discriminant is negative, the quadratic has no real roots, thus x^2 + 3x + 4 > 0
  have h_positive : ∀ x, x^2 + 3 * x + 4 > 0 := 
    by {
      intro x,
      apply (quadratic_not_negative_of_discriminant neg_discriminant).mp,
      exact h_discriminant,
    },
  exact absurd (show x^2 + 3 * x + 4 ≤ 0 from h_simplified) (lt_irrefl 0 (h_positive x)),
}

end no_solutions_to_inequality_l305_305690


namespace sqrt_50_product_consecutive_integers_l305_305308

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l305_305308


namespace exists_non_degenerate_triangle_l305_305337

theorem exists_non_degenerate_triangle
  (l : Fin 7 → ℝ)
  (h_ordered : ∀ i j, i ≤ j → l i ≤ l j)
  (h_bounds : ∀ i, 1 ≤ l i ∧ l i ≤ 12) :
  ∃ i j k : Fin 7, i < j ∧ j < k ∧ l i + l j > l k ∧ l j + l k > l i ∧ l k + l i > l j := 
sorry

end exists_non_degenerate_triangle_l305_305337


namespace number_of_arrangements_l305_305462

theorem number_of_arrangements (n : ℕ) (h_n : n = 6) : 
  ∃ total : ℕ, total = 90 := 
sorry

end number_of_arrangements_l305_305462


namespace product_of_consecutive_integers_sqrt_50_l305_305321

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (√50 ≥ m) ∧ (√50 < n) ∧ (m * n = 56) :=
by
  use 7, 8
  split
  exact Nat.lt_succ_self 7
  split
  norm_num
  split
  norm_num
  norm_num

end product_of_consecutive_integers_sqrt_50_l305_305321


namespace intersection_A_B_l305_305703

noncomputable def A : Set ℝ := {x | 9 * x ^ 2 < 1}

noncomputable def B : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2 - 2 * x + 5 / 4}

theorem intersection_A_B :
  (A ∩ B) = {y | y ∈ Set.Ico (1/4 : ℝ) (1/3 : ℝ)} :=
by
  sorry

end intersection_A_B_l305_305703


namespace effect_on_revenue_l305_305575

variables (P Q : ℝ)

def original_revenue : ℝ := P * Q
def new_price : ℝ := 1.60 * P
def new_quantity : ℝ := 0.80 * Q
def new_revenue : ℝ := new_price P * new_quantity Q

theorem effect_on_revenue (h1 : new_price P = 1.60 * P) (h2 : new_quantity Q = 0.80 * Q) :
  new_revenue P Q - original_revenue P Q = 0.28 * original_revenue P Q :=
by
  sorry

end effect_on_revenue_l305_305575


namespace max_crates_first_trip_l305_305215

theorem max_crates_first_trip (x : ℕ) : (∀ w, w ≥ 120) ∧ (600 ≥ x * 120) → x = 5 := 
by
  -- Condition: The weight of any crate is no less than 120 kg
  intro h
  have h1 : ∀ w, w ≥ 120 := h.left
  
  -- Condition: The maximum weight for the first trip
  have h2 : 600 ≥ x * 120 := h.right 
  
  -- Derivation of maximum crates
  have h3 : x ≤ 600 / 120 := by sorry  -- This inequality follows from h2 by straightforward division
  
  have h4 : x ≤ 5 := by sorry  -- This follows from evaluating 600 / 120 = 5
  
  -- Knowing x is an integer and the maximum possible value is 5
  exact by sorry

end max_crates_first_trip_l305_305215


namespace grandma_gave_each_l305_305579

-- Define the conditions
def gasoline: ℝ := 8
def lunch: ℝ := 15.65
def gifts: ℝ := 5 * 2  -- $5 each for two persons
def total_spent: ℝ := gasoline + lunch + gifts
def initial_amount: ℝ := 50
def amount_left: ℝ := 36.35

-- Define the proof problem
theorem grandma_gave_each :
  (amount_left - (initial_amount - total_spent)) / 2 = 10 :=
by
  sorry

end grandma_gave_each_l305_305579


namespace max_side_length_l305_305999

theorem max_side_length (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 :=
by
  sorry

end max_side_length_l305_305999


namespace division_of_difference_squared_l305_305035

theorem division_of_difference_squared :
  ((2222 - 2121)^2) / 196 = 52 := 
sorry

end division_of_difference_squared_l305_305035


namespace Jeff_probability_is_31_90_l305_305744

noncomputable def Jeff_spins : Prop :=
  let possible_starts := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let is_multiple_of_3 (n : ℕ) := n % 3 = 0
  let starts_at_multiple_of_3 := Finset.filter is_multiple_of_3 (Finset.range 11)
  let starts_at_one_more_than_multiple_of_3 := Finset.filter (λ n, (n % 3 = 1)) (Finset.range 11)
  let starts_at_one_less_than_multiple_of_3 := Finset.filter (λ n, (n % 3 = 2)) (Finset.range 11)
  let probability_of_starting := λ N : Finset ℕ, (N.card : ℚ) / 10
  
  let fair_spinner := [1, -1, -1] -- 1 represents one space right, -1 represents one space left
  let transition_probability (start : ℕ) : ℚ :=
    (fair_spinner.filter (λ x, is_multiple_of_3 (start + x))).card / 3
  
  let probability_of_ending_multiple_of_3 :=
    probability_of_starting starts_at_multiple_of_3 * (transition_probability 1 + transition_probability (-1)) +
    probability_of_starting starts_at_one_more_than_multiple_of_3 * transition_probability 1 * transition_probability 1 +
    probability_of_starting starts_at_one_less_than_multiple_of_3 * transition_probability (-1) * transition_probability (-1)
    
  probability_of_ending_multiple_of_3 = 31 / 90

theorem Jeff_probability_is_31_90 : Jeff_spins :=
by
  sorry

end Jeff_probability_is_31_90_l305_305744


namespace quadratic_two_distinct_real_roots_l305_305729

def quadratic_function_has_two_distinct_real_roots (k : ℝ) : Prop :=
  let a := k
  let b := -4
  let c := -2
  b * b - 4 * a * c > 0 ∧ a ≠ 0

theorem quadratic_two_distinct_real_roots (k : ℝ) :
  quadratic_function_has_two_distinct_real_roots k ↔ (k > -2 ∧ k ≠ 0) :=
by
  sorry

end quadratic_two_distinct_real_roots_l305_305729


namespace sqrt_50_between_consecutive_integers_product_l305_305300

theorem sqrt_50_between_consecutive_integers_product :
  ∃ (m n : ℕ), (m + 1 = n) ∧ (m * m < 50) ∧ (50 < n * n) ∧ (m * n = 56) :=
begin
  sorry
end

end sqrt_50_between_consecutive_integers_product_l305_305300


namespace negation_equiv_l305_305468

-- Given problem conditions
def exists_real_x_lt_0 : Prop := ∃ x : ℝ, x^2 + 1 < 0

-- Mathematically equivalent proof problem statement
theorem negation_equiv :
  ¬exists_real_x_lt_0 ↔ ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by
  sorry

end negation_equiv_l305_305468


namespace smaller_number_l305_305472

theorem smaller_number (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 16) : y = 4 := by
  sorry

end smaller_number_l305_305472


namespace determine_m_first_degree_inequality_l305_305860

theorem determine_m_first_degree_inequality (m : ℝ) (x : ℝ) :
  (m + 1) * x ^ |m| + 2 > 0 → |m| = 1 → m = 1 :=
by
  intro h1 h2
  sorry

end determine_m_first_degree_inequality_l305_305860


namespace amount_deducted_from_third_l305_305901

theorem amount_deducted_from_third
  (x : ℝ) 
  (h1 : ((x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 = 16)) 
  (h2 : (( (x - 9) + ((x + 1) - 8) + ((x + 2) - d) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9) ) / 10 = 11.5)) :
  d = 13.5 :=
by
  sorry

end amount_deducted_from_third_l305_305901


namespace ratio_x_y_l305_305058

noncomputable def side_length_x (x : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    a = 5 ∧ b = 12 ∧ c = 13 ∧ 
    (12 - x) / x = 5 / 12 ∧
    12 * x = 5 * x + 60 ∧
    7 * x = 60

noncomputable def side_length_y (y : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    a = 5 ∧ b = 12 ∧ c = 13 ∧
    y = 60 / 17

theorem ratio_x_y (x y : ℝ) (hx : side_length_x x) (hy : side_length_y y) : x / y = 17 / 7 :=
by
  sorry

end ratio_x_y_l305_305058


namespace distance_between_clocks_centers_l305_305028

variable (M m : ℝ)

theorem distance_between_clocks_centers :
  ∃ (c : ℝ), (|c| = (1/2) * (M + m)) := by
  sorry

end distance_between_clocks_centers_l305_305028


namespace sum_of_third_terms_arithmetic_progressions_l305_305407

theorem sum_of_third_terms_arithmetic_progressions
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (d1 d2 : ℕ)
  (h1 : ∃ d1 : ℕ, ∀ n : ℕ, a (n + 1) = a 1 + n * d1)
  (h2 : ∃ d2 : ℕ, ∀ n : ℕ, b (n + 1) = b 1 + n * d2)
  (h3 : a 1 + b 1 = 7)
  (h4 : a 5 + b 5 = 35) :
  a 3 + b 3 = 21 :=
by
  sorry

end sum_of_third_terms_arithmetic_progressions_l305_305407


namespace find_integer_pairs_l305_305525

theorem find_integer_pairs :
  ∃ (x y : ℤ),
    (x, y) = (-7, -99) ∨ (x, y) = (-1, -9) ∨ (x, y) = (1, 5) ∨ (x, y) = (7, -97) ∧
    2 * x^3 + x * y - 7 = 0 :=
by
  sorry

end find_integer_pairs_l305_305525


namespace james_used_5_containers_l305_305562

-- Conditions
def initial_balls : ℕ := 100
def balls_given_away : ℕ := initial_balls / 2
def remaining_balls : ℕ := initial_balls - balls_given_away
def balls_per_container : ℕ := 10

-- Question (statement of the theorem to prove)
theorem james_used_5_containers : (remaining_balls / balls_per_container) = 5 := by
  sorry

end james_used_5_containers_l305_305562


namespace Abby_sits_in_seat_3_l305_305363

theorem Abby_sits_in_seat_3:
  ∃ (positions : Fin 5 → String),
  (positions 3 = "Abby") ∧
  (positions 4 = "Bret") ∧
  ¬ ((positions 3 = "Dana") ∨ (positions 5 = "Dana")) ∧
  ¬ ((positions 2 = "Erin") ∧ (positions 3 = "Carl") ∨
    (positions 3 = "Erin") ∧ (positions 5 = "Carl")) :=
  sorry

end Abby_sits_in_seat_3_l305_305363


namespace fraction_absent_l305_305429

theorem fraction_absent (p : ℕ) (x : ℚ) (h : (W / p) * 1.2 = W / (p * (1 - x))) : x = 1 / 6 :=
by
  sorry

end fraction_absent_l305_305429


namespace point_in_fourth_quadrant_l305_305434

theorem point_in_fourth_quadrant (x y : ℝ) (h1 : x = 2) (h2 : y = -3) : 
(x > 0) → (y < 0) → (x, y) = (2, -3) → quadrant (2, -3) = 4 :=
by
  sorry

end point_in_fourth_quadrant_l305_305434


namespace john_younger_than_mark_l305_305571

variable (Mark_age John_age Parents_age : ℕ)
variable (h_mark : Mark_age = 18)
variable (h_parents_age_relation : Parents_age = 5 * John_age)
variable (h_parents_when_mark_born : Parents_age = 22 + Mark_age)

theorem john_younger_than_mark : Mark_age - John_age = 10 :=
by
  -- We state the theorem and leave the proof as sorry
  sorry

end john_younger_than_mark_l305_305571


namespace pauls_score_is_91_l305_305552

theorem pauls_score_is_91 (q s c w : ℕ) 
  (h1 : q = 35)
  (h2 : s = 35 + 5 * c - 2 * w)
  (h3 : s > 90)
  (h4 : c + w ≤ 35)
  (h5 : ∀ s', 90 < s' ∧ s' < s → ¬ (∃ c' w', s' = 35 + 5 * c' - 2 * w' ∧ c' + w' ≤ 35 ∧ c' ≠ c)) : 
  s = 91 := 
sorry

end pauls_score_is_91_l305_305552


namespace room_perimeter_l305_305111

theorem room_perimeter (b l : ℝ) (h1 : l = 3 * b) (h2 : l * b = 12) : 2 * (l + b) = 16 :=
by sorry

end room_perimeter_l305_305111


namespace nonnegative_diff_roots_eq_8sqrt2_l305_305789

noncomputable def roots_diff (a b c : ℝ) : ℝ :=
  if h : b^2 - 4*a*c ≥ 0 then 
    let root1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
    let root2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
    abs (root1 - root2)
  else 
    0

theorem nonnegative_diff_roots_eq_8sqrt2 : 
  roots_diff 1 42 409 = 8 * Real.sqrt 2 :=
sorry

end nonnegative_diff_roots_eq_8sqrt2_l305_305789


namespace point_equidistant_x_axis_y_axis_line_l305_305816

theorem point_equidistant_x_axis_y_axis_line (x y : ℝ) (h1 : abs y = abs x) (h2 : abs (x + y - 2) / Real.sqrt 2 = abs x) :
  x = 1 :=
  sorry

end point_equidistant_x_axis_y_axis_line_l305_305816


namespace find_n_l305_305107

theorem find_n (n : ℕ) (h : 4 ^ 6 = 8 ^ n) : n = 4 :=
by
  sorry

end find_n_l305_305107


namespace find_x_l305_305406

-- Definitions based on provided conditions

def rectangle_length (x : ℝ) : ℝ := 4 * x
def rectangle_width (x : ℝ) : ℝ := x + 7
def rectangle_area (x : ℝ) : ℝ := rectangle_length x * rectangle_width x
def rectangle_perimeter (x : ℝ) : ℝ := 2 * rectangle_length x + 2 * rectangle_width x

-- Theorem statement
theorem find_x (x : ℝ) (h : rectangle_area x = 2 * rectangle_perimeter x) : x = 1 := 
sorry

end find_x_l305_305406


namespace find_max_side_length_l305_305952

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l305_305952


namespace percentage_of_180_equation_l305_305005

theorem percentage_of_180_equation (P : ℝ) (h : (P / 100) * 180 - (1 / 3) * ((P / 100) * 180) = 36) : P = 30 :=
sorry

end percentage_of_180_equation_l305_305005


namespace percentage_increase_B_more_than_C_l305_305161

noncomputable def percentage_increase :=
  let C_m := 14000
  let A_annual := 470400
  let A_m := A_annual / 12
  let B_m := (2 / 5) * A_m
  ((B_m - C_m) / C_m) * 100

theorem percentage_increase_B_more_than_C : percentage_increase = 12 :=
  sorry

end percentage_increase_B_more_than_C_l305_305161


namespace james_spent_6_dollars_l305_305741

-- Define the constants based on the conditions
def cost_milk : ℝ := 3
def cost_bananas : ℝ := 2
def tax_rate : ℝ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℝ := cost_milk + cost_bananas

-- Define the sales tax
def sales_tax : ℝ := total_cost_before_tax * tax_rate

-- Define the total amount spent
def total_amount_spent : ℝ := total_cost_before_tax + sales_tax

-- The theorem to prove that James spent $6
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end james_spent_6_dollars_l305_305741


namespace product_of_consecutive_integers_sqrt_50_l305_305333

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l305_305333


namespace relationship_points_l305_305109

noncomputable def is_on_inverse_proportion (m x y : ℝ) : Prop :=
  y = (-m^2 - 2) / x

theorem relationship_points (a b c m : ℝ) :
  is_on_inverse_proportion m a (-1) ∧
  is_on_inverse_proportion m b 2 ∧
  is_on_inverse_proportion m c 3 →
  a > c ∧ c > b :=
by
  sorry

end relationship_points_l305_305109


namespace total_lunch_cost_l305_305061

theorem total_lunch_cost
  (jose_lunch : ℚ)
  (rick_lunch : ℚ)
  (sophia_lunch : ℚ)
  (adam_lunch : ℚ)
  (emma_lunch : ℚ)
  (before_tax : ℚ)
  (taxes : ℚ)
  (total_cost : ℚ)
  (jose_lunch_eq : jose_lunch = 60)
  (rick_lunch_eq : rick_lunch = jose_lunch / 1.5)
  (sophia_lunch_eq : sophia_lunch = rick_lunch)
  (adam_lunch_eq : adam_lunch = (2 / 3) * rick_lunch)
  (emma_lunch_eq : emma_lunch = jose_lunch * (1 - 0.2))
  (before_tax_eq : before_tax = adam_lunch + rick_lunch + jose_lunch + sophia_lunch + emma_lunch)
  (tax_eq : taxes = 0.08 * before_tax)
  (total_cost_eq : total_cost = before_tax + taxes) :
  total_cost = 231.84 := sorry

end total_lunch_cost_l305_305061


namespace sum_to_product_cos_l305_305079

theorem sum_to_product_cos (a b : ℝ) : 
  Real.cos (a + b) + Real.cos (a - b) = 2 * Real.cos a * Real.cos b := 
  sorry

end sum_to_product_cos_l305_305079


namespace range_of_m_l305_305110

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 2^|x| + m = 0) → m ≤ -1 :=
by
  sorry

end range_of_m_l305_305110


namespace no_solution_l305_305692

theorem no_solution (x : ℝ) : ¬ (3 * x^2 + 9 * x ≤ -12) :=
sorry

end no_solution_l305_305692


namespace circle_center_l305_305526

theorem circle_center :
    ∃ (h k : ℝ), (x^2 - 10 * x + y^2 - 4 * y = -4) →
                 (x - h)^2 + (y - k)^2 = 25 ∧ h = 5 ∧ k = 2 :=
sorry

end circle_center_l305_305526


namespace minimum_a_l305_305850

theorem minimum_a (a b x : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : b - a = 2013) (h₃ : x > 0) (h₄ : x^2 - a * x + b = 0) : a = 93 :=
by
  sorry

end minimum_a_l305_305850


namespace parabola_equivalence_l305_305821

theorem parabola_equivalence :
  ∃ (a : ℝ) (h k : ℝ),
    (a = -3 ∧ h = -1 ∧ k = 2) ∧
    ∀ (x : ℝ), (y = -3 * x^2 + 1) → (y = -3 * (x + 1)^2 + 2) :=
sorry

end parabola_equivalence_l305_305821


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l305_305669

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l305_305669


namespace smallest_composite_no_prime_factors_lt_20_l305_305666

theorem smallest_composite_no_prime_factors_lt_20 : 
  ∃ n, n = 667 ∧ ∀ p, prime p → p ∣ n → p ≥ 20 ∧ ∀ m, m < 667 → ∃ p, prime p ∧ p ∣ m ∧ p < 20 :=
by
  -- Proof goes here
  sorry

end smallest_composite_no_prime_factors_lt_20_l305_305666


namespace absolute_value_neg_2022_l305_305149

theorem absolute_value_neg_2022 : abs (-2022) = 2022 :=
by sorry

end absolute_value_neg_2022_l305_305149


namespace product_of_integers_around_sqrt_50_l305_305310

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l305_305310


namespace problem_l305_305652

theorem problem (h k : ℤ) 
  (h1 : 5 * 3 ^ 4 - h * 3 ^ 2 + k = 0)
  (h2 : 5 * (-1) ^ 4 - h * (-1) ^ 2 + k = 0)
  (h3 : 5 * 2 ^ 4 - h * 2 ^ 2 + k = 0) :
  |5 * h - 4 * k| = 70 := 
sorry

end problem_l305_305652


namespace intersection_when_m_eq_2_range_of_m_l305_305569

open Set

variables (m x : ℝ)

def A (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}
def B : Set ℝ := {x | -4 ≤ x ∧ x ≤ 2}
def intersection (m : ℝ) : Set ℝ := A m ∩ B

-- First proof: When m = 2, the intersection of A and B is [1,2].
theorem intersection_when_m_eq_2 : intersection 2 = {x | 1 ≤ x ∧ x ≤ 2} :=
sorry

-- Second proof: The range of m such that A ⊆ A ∩ B
theorem range_of_m : {m | A m ⊆ B} = {m | -2 ≤ m ∧ m ≤ 1 / 2} :=
sorry

end intersection_when_m_eq_2_range_of_m_l305_305569


namespace central_angle_of_regular_hexagon_l305_305906

theorem central_angle_of_regular_hexagon:
  ∀ (α : ℝ), 
  (∃ n : ℕ, n = 6 ∧ n * α = 360) →
  α = 60 :=
by
  sorry

end central_angle_of_regular_hexagon_l305_305906


namespace predicted_temperature_l305_305516

-- Define the observation data points
def data_points : List (ℕ × ℝ) :=
  [(20, 25), (30, 27.5), (40, 29), (50, 32.5), (60, 36)]

-- Define the linear regression equation with constant k
def regression (x : ℕ) (k : ℝ) : ℝ :=
  0.25 * x + k

-- Proof statement
theorem predicted_temperature (k : ℝ) (h : regression 40 k = 30) : regression 80 k = 40 :=
by
  sorry

end predicted_temperature_l305_305516


namespace days_left_in_year_is_100_l305_305549

noncomputable def days_left_in_year 
    (daily_average_rain_before : ℝ) 
    (total_rainfall_so_far : ℝ) 
    (average_rain_needed : ℝ) 
    (total_days_in_year : ℕ) : ℕ :=
    sorry

theorem days_left_in_year_is_100 :
    days_left_in_year 2 430 3 365 = 100 := 
sorry

end days_left_in_year_is_100_l305_305549


namespace problem_1_problem_2_l305_305897

noncomputable def problem_1_solution : Set ℝ := {6, -2}
noncomputable def problem_2_solution : Set ℝ := {2 + Real.sqrt 7, 2 - Real.sqrt 7}

theorem problem_1 :
  {x : ℝ | x^2 - 4 * x - 12 = 0} = problem_1_solution :=
by
  sorry

theorem problem_2 :
  {x : ℝ | x^2 - 4 * x - 3 = 0} = problem_2_solution :=
by
  sorry

end problem_1_problem_2_l305_305897


namespace min_value_2a_plus_b_l305_305346

theorem min_value_2a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : (1/a) + (2/b) = 1): 2 * a + b = 8 :=
sorry

end min_value_2a_plus_b_l305_305346


namespace sqrt_50_product_consecutive_integers_l305_305307

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l305_305307


namespace cans_to_collect_l305_305881

theorem cans_to_collect
  (martha_cans : ℕ)
  (diego_half_plus_ten : ℕ)
  (total_cans_required : ℕ)
  (martha_cans_collected : martha_cans = 90)
  (diego_collected : diego_half_plus_ten = (martha_cans / 2) + 10)
  (goal_cans : total_cans_required = 150) :
  total_cans_required - (martha_cans + diego_half_plus_ten) = 5 :=
by
  sorry

end cans_to_collect_l305_305881


namespace like_terms_exp_l305_305533

theorem like_terms_exp (a b : ℝ) (m n x : ℝ)
  (h₁ : 2 * a ^ x * b ^ (n + 1) = -3 * a * b ^ (2 * m))
  (h₂ : x = 1) (h₃ : n + 1 = 2 * m) : 
  (2 * m - n) ^ x = 1 := 
by
  sorry

end like_terms_exp_l305_305533


namespace percentage_of_tip_is_25_l305_305063

-- Definitions of the costs
def cost_samosas : ℕ := 3 * 2
def cost_pakoras : ℕ := 4 * 3
def cost_mango_lassi : ℕ := 2

-- Definition of total food cost
def total_food_cost : ℕ := cost_samosas + cost_pakoras + cost_mango_lassi

-- Definition of the total meal cost including tax
def total_meal_cost_with_tax : ℕ := 25

-- Definition of the tip
def tip : ℕ := total_meal_cost_with_tax - total_food_cost

-- Definition of the percentage of the tip
def percentage_tip : ℕ := (tip * 100) / total_food_cost

-- The theorem to be proved
theorem percentage_of_tip_is_25 :
  percentage_tip = 25 :=
by
  sorry

end percentage_of_tip_is_25_l305_305063


namespace cos_690_eq_sqrt3_div_2_l305_305836

theorem cos_690_eq_sqrt3_div_2 : Real.cos (690 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end cos_690_eq_sqrt3_div_2_l305_305836


namespace num_ways_books_distribution_l305_305807

-- Given conditions
def num_copies_type1 : ℕ := 8
def num_copies_type2 : ℕ := 4
def min_books_in_library_type1 : ℕ := 1
def max_books_in_library_type1 : ℕ := 7
def min_books_in_library_type2 : ℕ := 1
def max_books_in_library_type2 : ℕ := 3

-- The proof problem statement
theorem num_ways_books_distribution : 
  (max_books_in_library_type1 - min_books_in_library_type1 + 1) * 
  (max_books_in_library_type2 - min_books_in_library_type2 + 1) = 21 := by
    sorry

end num_ways_books_distribution_l305_305807


namespace proportional_b_value_l305_305864

theorem proportional_b_value (b : ℚ) : (∃ k : ℚ, k ≠ 0 ∧ (∀ x : ℚ, x + 2 - 3 * b = k * x)) ↔ b = 2 / 3 :=
by
  sorry

end proportional_b_value_l305_305864


namespace sprinkles_remaining_l305_305824

/-- Given that Coleen started with twelve cans of sprinkles and after applying she had 3 less than half as many cans, prove the remaining cans are 3 --/
theorem sprinkles_remaining (initial_sprinkles : ℕ) (h_initial : initial_sprinkles = 12) (h_remaining : ∃ remaining_sprinkles, remaining_sprinkles = initial_sprinkles / 2 - 3) : ∃ remaining_sprinkles, remaining_sprinkles = 3 :=
by
  have half_initial := initial_sprinkles / 2
  have remaining_sprinkles := half_initial - 3
  use remaining_sprinkles
  rw [h_initial, Nat.div_eq_of_lt (by decide : 6 < 12)]
  sorry

end sprinkles_remaining_l305_305824


namespace center_of_symmetry_exists_l305_305249

def translated_function (x : ℝ) : ℝ := 3 * sin (2 * x - π / 6)

theorem center_of_symmetry_exists (k : ℤ) : 
  ∃ x : ℝ, translated_function x = 0 ∧ x = (k : ℝ) * π / 2 + π / 12 := 
sorry

end center_of_symmetry_exists_l305_305249


namespace total_amount_collected_in_paise_total_amount_collected_in_rupees_l305_305207

-- Definitions and conditions
def num_members : ℕ := 96
def contribution_per_member : ℕ := 96
def total_paise_collected : ℕ := num_members * contribution_per_member
def total_rupees_collected : ℚ := total_paise_collected / 100

-- Theorem stating the total amount collected
theorem total_amount_collected_in_paise :
  total_paise_collected = 9216 := by sorry

theorem total_amount_collected_in_rupees :
  total_rupees_collected = 92.16 := by sorry

end total_amount_collected_in_paise_total_amount_collected_in_rupees_l305_305207


namespace min_sum_chessboard_labels_l305_305915

theorem min_sum_chessboard_labels :
  ∃ (r : Fin 9 → Fin 9), 
  (∀ (i j : Fin 9), i ≠ j → r i ≠ r j) ∧ 
  ((Finset.univ : Finset (Fin 9)).sum (λ i => 1 / (r i + i.val + 1)) = 1) :=
by
  sorry

end min_sum_chessboard_labels_l305_305915


namespace three_y_squared_value_l305_305487

theorem three_y_squared_value : ∃ x y : ℤ, 3 * x + y = 40 ∧ 2 * x - y = 20 ∧ 3 * y ^ 2 = 48 :=
by
  sorry

end three_y_squared_value_l305_305487


namespace remainder_polynomial_l305_305614

theorem remainder_polynomial (x : ℤ) : (1 + x) ^ 2010 % (1 + x + x^2) = 1 := 
  sorry

end remainder_polynomial_l305_305614


namespace part_I_part_II_l305_305869

namespace ArithmeticGeometricSequences

-- Definitions of sequences and their properties
def a1 : ℕ := 1
def b1 : ℕ := 2
def b (n : ℕ) : ℕ := 2 * 3 ^ (n - 1) -- General term of the geometric sequence

-- Definitions from given conditions
def a (n : ℕ) : ℕ := 3 * n - 2 -- General term of the arithmetic sequence

-- Sum of the first n terms of the geometric sequence
def S (n : ℕ) : ℕ := if n = 0 then 0 else 3 ^ n - 1

-- Theorem statement
theorem part_I (n : ℕ) : 
  (a1 = 1) ∧ 
  (b1 = 2) ∧ 
  (∀ n > 0, b n > 0) ∧ 
  (∀ b2 : ℕ, 2 * (1 + b2 / 2) = 2 + b2) ∧ 
  (∀ b2 a2 : ℕ, (1 + b2 / 2)^2 = b2 * ((a 3) + 2)) →
  (a n = 3 * n - 2) ∧ 
  (b n = 2 * 3 ^ (n - 1)) :=
  sorry

theorem part_II (n : ℕ) (m : ℝ) :
  (a1 = 1) ∧ 
  (b1 = 2) ∧ 
  (∀ n > 0, b n > 0) ∧ 
  (∀ b2 : ℕ, 2 * (1 + b2 / 2) = 2 + b2) ∧ 
  (∀ b2 a2 : ℕ, (1 + b2 / 2)^2 = b2 * ((a 3) + 2)) → 
  (∀ n > 0, S n + a n > m) → 
  (m < 3) :=
  sorry

end ArithmeticGeometricSequences

end part_I_part_II_l305_305869


namespace tim_kept_amount_l305_305785

-- Definitions as direct conditions
def total_winnings : ℝ := 100
def percentage_given_away : ℝ := 20 / 100

-- The mathematically equivalent proof problem as a theorem statement
theorem tim_kept_amount : total_winnings - (percentage_given_away * total_winnings) = 80 := by
  sorry

end tim_kept_amount_l305_305785


namespace radar_arrangements_l305_305830

-- Define the number of letters in the word RADAR
def total_letters : Nat := 5

-- Define the number of times each letter is repeated
def repetition_R : Nat := 2
def repetition_A : Nat := 2

-- The expected number of unique arrangements
def expected_unique_arrangements : Nat := 30

theorem radar_arrangements :
  (Nat.factorial total_letters) / (Nat.factorial repetition_R * Nat.factorial repetition_A) = expected_unique_arrangements := by
  sorry

end radar_arrangements_l305_305830


namespace ethan_arianna_apart_l305_305225

def ethan_distance := 1000 -- the distance Ethan ran
def arianna_distance := 184 -- the distance Arianna ran

theorem ethan_arianna_apart : ethan_distance - arianna_distance = 816 := by
  sorry

end ethan_arianna_apart_l305_305225


namespace dot_product_focus_hyperbola_l305_305749

-- Definitions related to the problem of the hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / 3) - y^2 = 1

def is_focus (c : ℝ) (x y : ℝ) : Prop := (x = c ∧ y = 0) ∨ (x = -c ∧ y = 0)

-- Problem conditions
def point_on_hyperbola (p : ℝ × ℝ) : Prop := hyperbola p.1 p.2

def triangle_area (a b c : ℝ × ℝ) (area : ℝ) : Prop :=
  0.5 * (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2)) = area

def foci_of_hyperbola : (ℝ × ℝ) × (ℝ × ℝ) := ((2, 0), (-2, 0))

-- Main statement to prove
theorem dot_product_focus_hyperbola
  (m n : ℝ)
  (hP : point_on_hyperbola (m, n))
  (hArea : triangle_area (2, 0) (m, n) (-2, 0) 2) :
  ((-2 - m) * (2 - m) + (-n) * (-n)) = 3 :=
sorry

end dot_product_focus_hyperbola_l305_305749


namespace x_gt_neg2_is_necessary_for_prod_lt_0_l305_305832

theorem x_gt_neg2_is_necessary_for_prod_lt_0 (x : Real) :
  (x > -2) ↔ (((x + 2) * (x - 3)) < 0) → (x > -2) :=
by
  sorry

end x_gt_neg2_is_necessary_for_prod_lt_0_l305_305832


namespace point_in_fourth_quadrant_l305_305433

theorem point_in_fourth_quadrant (x y : ℝ) (h1 : x = 2) (h2 : y = -3) : 
(x > 0) → (y < 0) → (x, y) = (2, -3) → quadrant (2, -3) = 4 :=
by
  sorry

end point_in_fourth_quadrant_l305_305433


namespace otimes_example_l305_305381

def otimes (a b : ℤ) : ℤ := a^2 - abs b

theorem otimes_example : otimes (-2) (-1) = 3 := by
  -- Define the variables
  let a := -2
  let b := -1
  -- Unfold the definition of otimes
  have h1 : otimes a b = a^2 - abs b := rfl
  -- Calculate a^2
  have h2 : a^2 = 4 := rfl
  -- Calculate abs b
  have h3 : abs b = 1 := rfl
  -- Calculate otimes a b
  show otimes a b = 3 from by
    rw [h1, h2, h3]
    rfl

end otimes_example_l305_305381


namespace part1_part2_l305_305849

variable {α : ℝ} (h1 : α ∈ Set.Ioo (π) (3 * π / 2))

def f (α : ℝ) : ℝ := 
  (sin (α - π / 2) * cos (3 * π / 2 + α) * tan (π - α)) / 
  (tan (-α - π) * sin (-α - π))

theorem part1 (h1 : α ∈ Set.Ioo (π) (3 * π / 2)) :
  f(α) = -cos(α) := by
  sorry

theorem part2 (h2 : cos(α - 3 * π / 2) = 1 / 5) (h1 : α ∈ Set.Ioo (π) (3 * π / 2)) :
  f(2 * α) = -23 / 25 := by
  sorry

end part1_part2_l305_305849


namespace final_selling_price_l305_305938

-- Define the conditions as constants
def CP := 750
def loss_percentage := 20 / 100
def sales_tax_percentage := 10 / 100

-- Define the final selling price after loss and adding sales tax
theorem final_selling_price 
  (CP : ℝ) 
  (loss_percentage : ℝ)
  (sales_tax_percentage : ℝ) 
  : 750 = CP ∧ 20 / 100 = loss_percentage ∧ 10 / 100 = sales_tax_percentage → 
    (CP - (loss_percentage * CP) + (sales_tax_percentage * CP) = 675) := 
by
  intros
  sorry

end final_selling_price_l305_305938


namespace problem1_problem2_problem3_l305_305929

-- Problem 1
theorem problem1 (x : ℝ) (h : x^2 - 3 * x = 2) : 1 + 2 * x^2 - 6 * x = 5 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : x^2 - 3 * x - 4 = 0) : 1 + 3 * x - x^2 = -3 :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) (p q : ℝ) (h1 : x = 1 → p * x^3 + q * x + 1 = 5) (h2 : p + q = 4) (hx : x = -1) : p * x^3 + q * x + 1 = -3 :=
by
  sorry

end problem1_problem2_problem3_l305_305929


namespace origami_papers_per_cousin_l305_305038

/-- Haley has 48 origami papers and 6 cousins. Each cousin should receive the same number of papers. -/
theorem origami_papers_per_cousin : ∀ (total_papers : ℕ) (number_of_cousins : ℕ),
  total_papers = 48 → number_of_cousins = 6 → total_papers / number_of_cousins = 8 :=
by
  intros total_papers number_of_cousins
  sorry

end origami_papers_per_cousin_l305_305038


namespace cricketer_average_increase_l305_305545

theorem cricketer_average_increase (A : ℝ) (H1 : 18 * A + 98 = 19 * 26) :
  26 - A = 4 :=
by
  sorry

end cricketer_average_increase_l305_305545


namespace question1_question2_l305_305245

def f (x : ℝ) : ℝ := abs (x - 5) - abs (x - 2)

theorem question1 :
  (∃ x : ℝ, f x ≤ m) ↔ m ≥ -3 :=
sorry

theorem question2 :
  { x : ℝ | x^2 - 8*x + 15 + f x ≤ 0 } = { x | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6 } :=
sorry

end question1_question2_l305_305245


namespace positive_integers_a_2014_b_l305_305071

theorem positive_integers_a_2014_b (a : ℕ) :
  (∃! b : ℕ, 2 ≤ a / b ∧ a / b ≤ 5) → a = 6710 ∨ a = 6712 ∨ a = 6713 :=
by
  sorry

end positive_integers_a_2014_b_l305_305071


namespace gratuities_charged_l305_305640

-- Define the conditions in the problem
def total_bill : ℝ := 140
def sales_tax_rate : ℝ := 0.10
def ny_striploin_cost : ℝ := 80
def wine_cost : ℝ := 10

-- Calculate the total cost before tax and gratuities
def subtotal : ℝ := ny_striploin_cost + wine_cost

-- Calculate the taxes paid
def tax : ℝ := subtotal * sales_tax_rate

-- Calculate the total bill before gratuities
def total_before_gratuities : ℝ := subtotal + tax

-- Goal: Prove that gratuities charged is 41
theorem gratuities_charged : (total_bill - total_before_gratuities) = 41 := by sorry

end gratuities_charged_l305_305640


namespace length_of_AB_l305_305868

variables (AB CD : ℝ)

-- Given conditions
def area_ratio (h : ℝ) : Prop := (1/2 * AB * h) / (1/2 * CD * h) = 4
def sum_condition : Prop := AB + CD = 200

-- The proof problem: proving the length of AB
theorem length_of_AB (h : ℝ) (h_area_ratio : area_ratio AB CD h) 
  (h_sum_condition : sum_condition AB CD) : AB = 160 :=
sorry

end length_of_AB_l305_305868


namespace max_side_of_triangle_exists_max_side_of_elevent_l305_305977

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l305_305977


namespace calculation_correct_l305_305510

noncomputable def calc_expression : Float :=
  20.17 * 69 + 201.7 * 1.3 - 8.2 * 1.7

theorem calculation_correct : calc_expression = 1640 := 
  by 
    sorry

end calculation_correct_l305_305510


namespace quadratic_inequality_solution_l305_305112

theorem quadratic_inequality_solution {a : ℝ} :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ a < -1 ∨ a > 3 :=
by sorry

end quadratic_inequality_solution_l305_305112


namespace sequence_general_formula_l305_305291

theorem sequence_general_formula (a : ℕ → ℕ) :
  (a 1 = 1) ∧ (a 2 = 2) ∧ (a 3 = 4) ∧ (a 4 = 8) ∧ (a 5 = 16) → ∀ n : ℕ, n > 0 → a n = 2^(n-1) :=
by
  intros h n hn
  sorry

end sequence_general_formula_l305_305291


namespace seeds_in_bucket_C_l305_305025

theorem seeds_in_bucket_C (A B C : ℕ) (h1 : A + B + C = 100) (h2 : A = B + 10) (h3 : B = 30) : C = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end seeds_in_bucket_C_l305_305025


namespace sum_of_digits_of_N_l305_305620

theorem sum_of_digits_of_N :
  ∃ N : ℕ, 
    10 ≤ N ∧ N < 100 ∧
    5655 % N = 11 ∧ 
    5879 % N = 14 ∧ 
    ((N / 10) + (N % 10)) = 8 := 
sorry

end sum_of_digits_of_N_l305_305620


namespace find_a_of_parabola_l305_305466

theorem find_a_of_parabola
  (a b c : ℝ)
  (h_point : 2 = c)
  (h_vertex : -2 = a * (2 - 2)^2 + b * 2 + c) :
  a = 1 :=
by
  sorry

end find_a_of_parabola_l305_305466


namespace range_of_f_l305_305224

open Set

noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (2 * x - 1)

theorem range_of_f : range f = Ici (1 / 2) :=
by
  sorry

end range_of_f_l305_305224


namespace cone_curved_surface_area_l305_305345

def radius (r : ℝ) := r = 3
def slantHeight (l : ℝ) := l = 15
def curvedSurfaceArea (csa : ℝ) := csa = 45 * Real.pi

theorem cone_curved_surface_area 
  (r l csa : ℝ) 
  (hr : radius r) 
  (hl : slantHeight l) 
  : curvedSurfaceArea (Real.pi * r * l) 
  := by
  unfold radius at hr
  unfold slantHeight at hl
  unfold curvedSurfaceArea
  rw [hr, hl]
  norm_num
  sorry

end cone_curved_surface_area_l305_305345


namespace luke_payments_difference_l305_305276

theorem luke_payments_difference :
  let principal := 12000
  let rate := 0.08
  let years := 10
  let n_quarterly := 4
  let n_annually := 1
  let quarterly_rate := rate / n_quarterly
  let annually_rate := rate / n_annually
  let balance_plan1_5years := principal * (1 + quarterly_rate)^(n_quarterly * 5)
  let payment_plan1_5years := balance_plan1_5years / 3
  let remaining_balance_plan1_5years := balance_plan1_5years - payment_plan1_5years
  let final_balance_plan1_10years := remaining_balance_plan1_5years * (1 + quarterly_rate)^(n_quarterly * 5)
  let total_payment_plan1 := payment_plan1_5years + final_balance_plan1_10years
  let final_balance_plan2_10years := principal * (1 + annually_rate)^years
  (total_payment_plan1 - final_balance_plan2_10years).abs = 1022 :=
by
  sorry

end luke_payments_difference_l305_305276


namespace quadratic_root_expression_l305_305267

theorem quadratic_root_expression (a b : ℝ) 
  (h : ∀ x : ℝ, x^2 + x - 2023 = 0 → (x = a ∨ x = b)) 
  (ha_neq_b : a ≠ b) :
  a^2 + 2*a + b = 2022 :=
sorry

end quadratic_root_expression_l305_305267


namespace prob_all_fail_prob_at_least_one_pass_l305_305178

def prob_pass := 1 / 2
def prob_fail := 1 - prob_pass

def indep (A B C : Prop) : Prop := true -- Usually we prove independence in a detailed manner, but let's assume it's given as true.

theorem prob_all_fail (A B C : Prop) (hA : prob_pass = 1 / 2) (hB : prob_pass = 1 / 2) (hC : prob_pass = 1 / 2) 
  (indepABC : indep A B C) : (prob_fail * prob_fail * prob_fail) = 1 / 8 :=
by
  sorry

theorem prob_at_least_one_pass (A B C : Prop) (hA : prob_pass = 1 / 2) (hB : prob_pass = 1 / 2) (hC : prob_pass = 1 / 2) 
  (indepABC : indep A B C) : 1 - (prob_fail * prob_fail * prob_fail) = 7 / 8 :=
by
  sorry

end prob_all_fail_prob_at_least_one_pass_l305_305178


namespace midpoint_product_zero_l305_305750

theorem midpoint_product_zero (x y : ℝ) :
  let A := (2, 6)
  let B := (x, y)
  let C := (4, 3)
  (C = ((2 + x) / 2, (6 + y) / 2)) → (x * y = 0) := by
  intros
  sorry

end midpoint_product_zero_l305_305750


namespace distance_travelled_l305_305355

theorem distance_travelled (speed time distance : ℕ) 
  (h1 : speed = 25)
  (h2 : time = 5)
  (h3 : distance = speed * time) : 
  distance = 125 :=
by
  sorry

end distance_travelled_l305_305355


namespace pens_more_than_notebooks_l305_305604

theorem pens_more_than_notebooks
  (N P : ℕ) 
  (h₁ : N = 30) 
  (h₂ : N + P = 110) :
  P - N = 50 := 
by
  sorry

end pens_more_than_notebooks_l305_305604


namespace largest_value_B_l305_305041

theorem largest_value_B :
  let A := ((1 / 2) / (3 / 4))
  let B := (1 / ((2 / 3) / 4))
  let C := (((1 / 2) / 3) / 4)
  let E := ((1 / (2 / 3)) / 4)
  B > A ∧ B > C ∧ B > E :=
by
  sorry

end largest_value_B_l305_305041


namespace smallest_composite_no_prime_factors_less_than_20_l305_305674

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

def has_no_prime_factors_less_than (n m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than n 20 ∧
    ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → n ≤ m :=
  exists.intro 529
    (by
      have h1 : is_composite 529 :=
          ⟨529 > 1, 23 > 1 ∧ 23 > 1, 23 * 23 = 529⟩ sorry
      have h2 : has_no_prime_factors_less_than 529 20 :=
          (by intros p hp1 hp2; cases hp1; cases hp2; sorry)
      have h3 : ∀ m : ℕ, is_composite m ∧ has_no_prime_factors_less_than m 20 → 529 ≤ m := sorry
      exact ⟨h1, h2, h3⟩)

end smallest_composite_no_prime_factors_less_than_20_l305_305674


namespace Jina_has_51_mascots_l305_305876

def teddies := 5
def bunnies := 3 * teddies
def koala_bear := 1
def additional_teddies := 2 * bunnies
def total_mascots := teddies + bunnies + koala_bear + additional_teddies

theorem Jina_has_51_mascots : total_mascots = 51 := by
  sorry

end Jina_has_51_mascots_l305_305876


namespace two_crows_problem_l305_305553

def Bird := { P | P = "parrot" ∨ P = "crow"} -- Define possible bird species.

-- Define birds and their statements
def Adam_statement (Adam Carl : Bird) : Prop := Carl = Adam
def Bob_statement (Adam : Bird) : Prop := Adam = "crow"
def Carl_statement (Dave : Bird) : Prop := Dave = "crow"
def Dave_statement (Adam Bob Carl Dave: Bird) : Prop := 
  (if Adam = "parrot" then 1 else 0) + 
  (if Bob = "parrot" then 1 else 0) + 
  (if Carl = "parrot" then 1 else 0) + 
  (if Dave = "parrot" then 1 else 0) ≥ 3

-- The main proposition to prove
def main_statement : Prop :=
  ∃ (Adam Bob Carl Dave : Bird), 
    (Adam_statement Adam Carl) ∧ 
    (Bob_statement Adam) ∧ 
    (Carl_statement Dave) ∧ 
    (Dave_statement Adam Bob Carl Dave) ∧ 
    (if Adam = "crow" then 1 else 0) + 
    (if Bob = "crow" then 1 else 0) + 
    (if Carl = "crow" then 1 else 0) + 
    (if Dave = "crow" then 1 else 0) = 2

-- Proof statement to be filled
theorem two_crows_problem : main_statement :=
by {
  sorry
}

end two_crows_problem_l305_305553


namespace divisible_by_17_l305_305008

theorem divisible_by_17 (n : ℕ) : 17 ∣ (2 ^ (5 * n + 3) + 5 ^ n * 3 ^ (n + 2)) := 
by {
  sorry
}

end divisible_by_17_l305_305008


namespace part1_part2_l305_305714

noncomputable def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_part2_l305_305714


namespace seeds_in_bucket_C_l305_305026

theorem seeds_in_bucket_C (A B C : ℕ) (h1 : A + B + C = 100) (h2 : A = B + 10) (h3 : B = 30) : C = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end seeds_in_bucket_C_l305_305026


namespace max_triangle_side_l305_305992

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l305_305992


namespace simplify_and_substitute_l305_305581

theorem simplify_and_substitute (x : ℝ) (h1 : x ≠ 1) (h3 : x ≠ 3) : 
  ((1 - (2 / (x - 1))) * ((x^2 - x) / (x^2 - 6*x + 9))) = (x / (x - 3)) ∧ 
  (2 / (2 - 3)) = -2 := by
  sorry

end simplify_and_substitute_l305_305581


namespace percentage_increase_l305_305826

-- Defining the problem constants
def price (P : ℝ) : ℝ := P
def assets_A (A : ℝ) : ℝ := A
def assets_B (B : ℝ) : ℝ := B
def percentage (X : ℝ) : ℝ := X

-- Conditions
axiom price_company_B_double_assets : ∀ (P B: ℝ), price P = 2 * assets_B B
axiom price_seventy_five_percent_combined_assets : ∀ (P A B: ℝ), price P = 0.75 * (assets_A A + assets_B B)
axiom price_percentage_more_than_A : ∀ (P A X: ℝ), price P = assets_A A * (1 + percentage X / 100)

-- Theorem to prove
theorem percentage_increase : ∀ (P A B X : ℝ)
  (h1 : price P = 2 * assets_B B)
  (h2 : price P = 0.75 * (assets_A A + assets_B B))
  (h3 : price P = assets_A A * (1 + percentage X / 100)),
  percentage X = 20 :=
by
  intros P A B X h1 h2 h3
  -- Proof steps would go here
  sorry

end percentage_increase_l305_305826


namespace find_value_of_x2_plus_y2_l305_305567

theorem find_value_of_x2_plus_y2 (x y : ℝ) (h : 5 * x^2 + y^2 - 4 * x * y + 24 ≤ 10 * x - 1) : x^2 + y^2 = 125 := 
sorry

end find_value_of_x2_plus_y2_l305_305567


namespace floor_neg_seven_thirds_l305_305524

theorem floor_neg_seven_thirds : Int.floor (-7 / 3 : ℚ) = -3 := by
  sorry

end floor_neg_seven_thirds_l305_305524


namespace problem_solution_l305_305400

noncomputable def solution_set : Set ℝ :=
  { x : ℝ | x ∈ (Set.Ioo 0 (5 - Real.sqrt 10)) ∨ x ∈ (Set.Ioi (5 + Real.sqrt 10)) }

theorem problem_solution (x : ℝ) : (x^3 - 10*x^2 + 15*x > 0) ↔ x ∈ solution_set :=
by
  sorry

end problem_solution_l305_305400


namespace smallest_composite_no_prime_factors_below_20_l305_305682

theorem smallest_composite_no_prime_factors_below_20 : 
  ∃ n : ℕ, n = 667 ∧ ∀ p : ℕ, prime p → p ∣ n → p ≥ 20 :=
by {
  sorry
}

end smallest_composite_no_prime_factors_below_20_l305_305682


namespace sin_squared_minus_cos_squared_l305_305241

theorem sin_squared_minus_cos_squared {α : ℝ} (h : Real.sin α = Real.sqrt 5 / 5) : 
  Real.sin α ^ 2 - Real.cos α ^ 2 = -3 / 5 :=
by
  sorry -- Proof is omitted

end sin_squared_minus_cos_squared_l305_305241


namespace max_side_of_triangle_exists_max_side_of_elevent_l305_305981

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l305_305981


namespace smallest_n_divisible_by_2016_smallest_n_divisible_by_2016_pow_10_l305_305085

-- Problem (a): Smallest n such that n! is divisible by 2016
theorem smallest_n_divisible_by_2016 : ∃ (n : ℕ), n = 8 ∧ 2016 ∣ n.factorial :=
by
  sorry

-- Problem (b): Smallest n such that n! is divisible by 2016^10
theorem smallest_n_divisible_by_2016_pow_10 : ∃ (n : ℕ), n = 63 ∧ 2016^10 ∣ n.factorial :=
by
  sorry

end smallest_n_divisible_by_2016_smallest_n_divisible_by_2016_pow_10_l305_305085


namespace smallest_period_of_f_intervals_where_f_is_decreasing_l305_305538

open Real
open Int

noncomputable def f (x : ℝ) := 4 * sin x * cos (x - π / 3) - sqrt 3

theorem smallest_period_of_f : ∃ T > 0, (∀ x, f(x + T) = f(x)) ∧ T = π := by
  sorry

theorem intervals_where_f_is_decreasing : 
  ∀ k : ℤ, ∃ a b, a = k * π + 5 * π / 12 ∧ b = k * π + 11 * π / 12 ∧ 
  ∀ x ∈ Icc a b, (∀ x₁ ∈ Ioo a b, deriv f x₁ < 0) := by
  sorry

end smallest_period_of_f_intervals_where_f_is_decreasing_l305_305538


namespace total_area_painted_correct_l305_305057

-- Defining the properties of the shed
def shed_w := 12  -- width in yards
def shed_l := 15  -- length in yards
def shed_h := 7   -- height in yards

-- Calculating area to be painted
def wall_area_1 := 2 * (shed_w * shed_h)
def wall_area_2 := 2 * (shed_l * shed_h)
def floor_ceiling_area := 2 * (shed_w * shed_l)
def total_painted_area := wall_area_1 + wall_area_2 + floor_ceiling_area

-- The theorem to be proved
theorem total_area_painted_correct :
  total_painted_area = 738 := by
  sorry

end total_area_painted_correct_l305_305057


namespace max_triangle_side_l305_305991

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l305_305991


namespace executiveCommittee_ways_l305_305449

noncomputable def numberOfWaysToFormCommittee (totalMembers : ℕ) (positions : ℕ) : ℕ :=
Nat.choose (totalMembers - 1) (positions - 1)

theorem executiveCommittee_ways : numberOfWaysToFormCommittee 30 5 = 25839 := 
by
  -- skipping the proof as it's not required
  sorry

end executiveCommittee_ways_l305_305449


namespace max_side_length_l305_305966

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l305_305966


namespace range_of_a_l305_305865

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 - 2 * x ≤ a^2 - a - 3) ↔ (-1 < a ∧ a < 2) :=
by 
  sorry

end range_of_a_l305_305865


namespace brit_age_after_vacation_l305_305373

-- Define the given conditions and the final proof question

-- Rebecca's age is 25 years
def rebecca_age : ℕ := 25

-- Brittany is older than Rebecca by 3 years
def brit_age_before_vacation (rebecca_age : ℕ) : ℕ := rebecca_age + 3

-- Brittany goes on a 4-year vacation
def vacation_duration : ℕ := 4

-- Prove that Brittany’s age when she returns from her vacation is 32
theorem brit_age_after_vacation (rebecca_age vacation_duration : ℕ) : brit_age_before_vacation rebecca_age + vacation_duration = 32 :=
by
  sorry

end brit_age_after_vacation_l305_305373


namespace abs_nested_expression_l305_305087

theorem abs_nested_expression : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry

end abs_nested_expression_l305_305087


namespace cost_price_is_700_l305_305196

noncomputable def cost_price_was_700 : Prop :=
  ∃ (CP : ℝ),
    (∀ (SP1 SP2 : ℝ),
      SP1 = CP * 0.84 ∧
        SP2 = CP * 1.04 ∧
        SP2 = SP1 + 140) ∧
    CP = 700

theorem cost_price_is_700 : cost_price_was_700 :=
  sorry

end cost_price_is_700_l305_305196


namespace product_of_consecutive_integers_sqrt_50_l305_305320

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (√50 ≥ m) ∧ (√50 < n) ∧ (m * n = 56) :=
by
  use 7, 8
  split
  exact Nat.lt_succ_self 7
  split
  norm_num
  split
  norm_num
  norm_num

end product_of_consecutive_integers_sqrt_50_l305_305320


namespace sqrt_50_product_consecutive_integers_l305_305309

theorem sqrt_50_product_consecutive_integers :
  ∃ (n : ℕ), n^2 < 50 ∧ 50 < (n + 1)^2 ∧ n * (n + 1) = 56 :=
by
  sorry

end sqrt_50_product_consecutive_integers_l305_305309


namespace max_triangle_side_l305_305998

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l305_305998


namespace certain_number_example_l305_305861

theorem certain_number_example (x : ℝ) 
    (h1 : 213 * 16 = 3408)
    (h2 : 0.16 * x = 0.3408) : 
    x = 2.13 := 
by 
  sorry

end certain_number_example_l305_305861


namespace product_of_integers_around_sqrt_50_l305_305313

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l305_305313


namespace direction_vector_of_line_l305_305156

theorem direction_vector_of_line : ∃ Δx Δy : ℚ, y = - (1/2) * x + 1 → Δx = 2 ∧ Δy = -1 :=
sorry

end direction_vector_of_line_l305_305156


namespace valid_starting_days_count_l305_305810

def is_valid_starting_day (d : ℕ) : Prop :=
  (d % 7 = 3 ∨ d % 7 = 4 ∨ d % 7 = 5)

theorem valid_starting_days_count : 
  (finset.filter is_valid_starting_day (finset.range 7)).card = 3 :=
begin
  sorry
end

end valid_starting_days_count_l305_305810


namespace part1_part2_l305_305713

noncomputable def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_part2_l305_305713


namespace solve_P_Q_l305_305398

theorem solve_P_Q :
  ∃ P Q : ℝ, (∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 →
    (P / (x + 6) + Q / (x * (x - 5)) = (x^2 - 3*x + 15) / (x * (x + 6) * (x - 5)))) ∧
    P = 1 ∧ Q = 5/2 :=
by
  sorry

end solve_P_Q_l305_305398


namespace find_number_l305_305633

-- Definitions of the fractions involved
def frac_2_15 : ℚ := 2 / 15
def frac_1_5 : ℚ := 1 / 5
def frac_1_2 : ℚ := 1 / 2

-- Condition that the number is greater than the sum of frac_2_15 and frac_1_5 by frac_1_2 
def number : ℚ := frac_2_15 + frac_1_5 + frac_1_2

-- Theorem statement matching the math proof problem
theorem find_number : number = 5 / 6 :=
by
  sorry

end find_number_l305_305633


namespace find_number_69_3_l305_305289

theorem find_number_69_3 (x : ℝ) (h : (x * 0.004) / 0.03 = 9.237333333333334) : x = 69.3 :=
by
  sorry

end find_number_69_3_l305_305289


namespace general_formula_sequence_less_than_zero_maximum_sum_value_l305_305240

variable (n : ℕ)

-- Helper definition
def arithmetic_seq (d : ℤ) (a₁ : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Conditions given in the problem
def a₁ : ℤ := 31
def a₄ : ℤ := 7
def d : ℤ := (a₄ - a₁) / 3

-- Definitions extracted from problem conditions
def an (n : ℕ) : ℤ := arithmetic_seq d a₁ n
def Sn (n : ℕ) : ℤ := n * a₁ + (n * (n - 1) / 2) * d

-- Proving the general formula aₙ = -8n + 39
theorem general_formula :
  ∀ (n : ℕ), an n = -8 * n + 39 :=
by
  sorry

-- Proving when the sequence starts to be less than 0
theorem sequence_less_than_zero :
  ∀ (n : ℕ), n ≥ 5 → an n < 0 :=
by
  sorry

-- Proving that the sum Sn has a maximum value
theorem maximum_sum_value :
  Sn 4 = 76 ∧ ∀ (n : ℕ), Sn n ≤ 76 :=
by
  sorry

end general_formula_sequence_less_than_zero_maximum_sum_value_l305_305240


namespace smallest_composite_no_prime_factors_less_than_20_l305_305684

/--
Theorem: The smallest composite number that has no prime factors less than 20 is 529.
-/
theorem smallest_composite_no_prime_factors_less_than_20 : ∃ n : ℕ, (∃ k, k > 1 ∧ k < n ∧ k ∣ n) ∧ (∀ p : ℕ, prime p → p < 20 → ¬(p ∣ n)) ∧ n = 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_l305_305684


namespace puppies_adopted_per_day_l305_305636

theorem puppies_adopted_per_day 
    (initial_puppies : ℕ) 
    (additional_puppies : ℕ) 
    (total_days : ℕ) 
    (total_puppies : ℕ)
    (H1 : initial_puppies = 5) 
    (H2 : additional_puppies = 35) 
    (H3 : total_days = 5) 
    (H4 : total_puppies = initial_puppies + additional_puppies) : 
    total_puppies / total_days = 8 := by
  sorry

end puppies_adopted_per_day_l305_305636


namespace arun_weight_average_l305_305198

theorem arun_weight_average :
  ∀ (w : ℝ), (w > 61 ∧ w < 72) ∧ (w > 60 ∧ w < 70) ∧ (w ≤ 64) →
  (w = 62 ∨ w = 63) →
  (62 + 63) / 2 = 62.5 :=
by
  intros w h1 h2
  sorry

end arun_weight_average_l305_305198


namespace parallel_vectors_x_value_l305_305719

-- Defining the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Condition for vectors a and b to be parallel
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value : ∃ x, are_parallel a (b x) ∧ x = 6 := by
  sorry

end parallel_vectors_x_value_l305_305719


namespace fraction_ordering_l305_305184

theorem fraction_ordering :
  (8 : ℚ) / 24 < (6 : ℚ) / 17 ∧ (6 : ℚ) / 17 < (10 : ℚ) / 27 :=
by
  sorry

end fraction_ordering_l305_305184


namespace fraction_of_apples_consumed_l305_305647

theorem fraction_of_apples_consumed (f : ℚ) 
  (bella_eats_per_day : ℚ := 6) 
  (days_per_week : ℕ := 7) 
  (grace_remaining_apples : ℚ := 504) 
  (weeks_passed : ℕ := 6) 
  (total_apples_picked : ℚ := 42 / f) :
  (total_apples_picked - (bella_eats_per_day * days_per_week * weeks_passed) = grace_remaining_apples) 
  → f = 1 / 18 :=
by
  intro h
  sorry

end fraction_of_apples_consumed_l305_305647


namespace additional_miles_needed_l305_305448

theorem additional_miles_needed :
  ∀ (h : ℝ), (25 + 75 * h) / (5 / 8 + h) = 60 → 75 * h = 62.5 := 
by
  intros h H
  -- the rest of the proof goes here
  sorry

end additional_miles_needed_l305_305448


namespace rectangle_area_increase_l305_305928

variable (L B : ℝ)

theorem rectangle_area_increase :
  let L_new := 1.30 * L
  let B_new := 1.45 * B
  let A_original := L * B
  let A_new := L_new * B_new
  let A_increase := A_new - A_original
  let percentage_increase := (A_increase / A_original) * 100
  percentage_increase = 88.5 := by
    sorry

end rectangle_area_increase_l305_305928


namespace max_connected_stations_l305_305214

theorem max_connected_stations (n : ℕ) 
  (h1 : ∀ s : ℕ, s ≤ n → s ≤ 3) 
  (h2 : ∀ x y : ℕ, x < y → ∃ z : ℕ, z < 3 ∧ z ≤ n) : 
  n = 10 :=
by 
  sorry

end max_connected_stations_l305_305214


namespace five_coins_total_cannot_be_30_cents_l305_305091

theorem five_coins_total_cannot_be_30_cents :
  ¬ ∃ (a b c d e : ℕ), 
  a + b + c + d + e = 5 ∧ 
  (a * 1 + b * 5 + c * 10 + d * 25 + e * 50) = 30 := 
sorry

end five_coins_total_cannot_be_30_cents_l305_305091


namespace Nina_saves_enough_to_buy_video_game_in_11_weeks_l305_305884

-- Definitions (directly from conditions)
def game_cost : ℕ := 50
def tax_rate : ℚ := 10 / 100
def sales_tax (cost : ℕ) (rate : ℚ) : ℚ := cost * rate
def total_cost (cost : ℕ) (tax : ℚ) : ℚ := cost + tax
def weekly_allowance : ℕ := 10
def savings_rate : ℚ := 1 / 2
def weekly_savings (allowance : ℕ) (rate : ℚ) : ℚ := allowance * rate
def weeks_to_save (total_cost : ℚ) (savings_per_week : ℚ) : ℚ := total_cost / savings_per_week

-- Theorem to prove
theorem Nina_saves_enough_to_buy_video_game_in_11_weeks :
  weeks_to_save
    (total_cost game_cost (sales_tax game_cost tax_rate))
    (weekly_savings weekly_allowance savings_rate) = 11 := by
-- We skip the proof for now, as per instructions
  sorry

end Nina_saves_enough_to_buy_video_game_in_11_weeks_l305_305884


namespace sqrt_50_between_consecutive_integers_product_l305_305299

theorem sqrt_50_between_consecutive_integers_product :
  ∃ (m n : ℕ), (m + 1 = n) ∧ (m * m < 50) ∧ (50 < n * n) ∧ (m * n = 56) :=
begin
  sorry
end

end sqrt_50_between_consecutive_integers_product_l305_305299


namespace original_number_l305_305797

theorem original_number (x : ℝ) (h : 1.40 * x = 1680) : x = 1200 :=
by {
  sorry -- We will skip the actual proof steps here.
}

end original_number_l305_305797


namespace chess_club_members_l305_305295

theorem chess_club_members {n : ℤ} (h10 : n % 10 = 6) (h11 : n % 11 = 6) (rng : 300 ≤ n ∧ n ≤ 400) : n = 336 :=
  sorry

end chess_club_members_l305_305295


namespace integer_exponentiation_l305_305724

theorem integer_exponentiation
  (a b x y : ℕ)
  (h_gcd : a.gcd b = 1)
  (h_pos_a : 1 < a)
  (h_pos_b : 1 < b)
  (h_pos_x : 1 < x)
  (h_pos_y : 1 < y)
  (h_eq : x^a = y^b) :
  ∃ n : ℕ, 1 < n ∧ x = n^b ∧ y = n^a :=
by sorry

end integer_exponentiation_l305_305724


namespace find_e_l305_305913

theorem find_e (d e f : ℤ) (Q : ℤ → ℤ) (hQ : ∀ x, Q x = 3 * x^3 + d * x^2 + e * x + f)
  (mean_zeros_eq_prod_zeros : let zeros := {x // Q x = 0} in
    (∑ x in zeros, x) / 3 = ∏ x in zeros, x)
  (sum_coeff_eq_mean_zeros : 3 + d + e + f = (∑ x in {x // Q x = 0}, x) / 3)
  (y_intercept : Q 0 = 9) :
  e = -42 :=
sorry

end find_e_l305_305913


namespace max_side_of_triangle_exists_max_side_of_elevent_l305_305980

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l305_305980


namespace sqrt_50_between_7_and_8_l305_305317

theorem sqrt_50_between_7_and_8 (x y : ℕ) (h1 : sqrt 50 > 7) (h2 : sqrt 50 < 8) (h3 : y = x + 1) : x * y = 56 :=
by sorry

end sqrt_50_between_7_and_8_l305_305317


namespace weight_of_bag_l305_305440

-- Definitions
def chicken_price : ℝ := 1.50
def bag_cost : ℝ := 2
def feed_per_chicken : ℝ := 2
def profit_from_50_chickens : ℝ := 65
def total_chickens : ℕ := 50

-- Theorem
theorem weight_of_bag : 
  (bag_cost / (profit_from_50_chickens - 
               (total_chickens * chicken_price)) / 
               (feed_per_chicken * total_chickens)) = 20 := 
sorry

end weight_of_bag_l305_305440


namespace round_trip_average_mileage_l305_305361

theorem round_trip_average_mileage 
  (d1 d2 : ℝ) (m1 m2 : ℝ)
  (h1 : d1 = 150) (h2 : d2 = 150)
  (h3 : m1 = 40) (h4 : m2 = 25) :
  (d1 + d2) / ((d1 / m1) + (d2 / m2)) = 30.77 :=
by
  sorry

end round_trip_average_mileage_l305_305361


namespace remainder_seven_pow_two_thousand_mod_thirteen_l305_305659

theorem remainder_seven_pow_two_thousand_mod_thirteen :
  7^2000 % 13 = 1 := by
  sorry

end remainder_seven_pow_two_thousand_mod_thirteen_l305_305659


namespace least_number_to_subtract_l305_305625

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (h1: n = 509) (h2 : d = 9): ∃ k : ℕ, k = 5 ∧ ∃ m : ℕ, n - k = d * m :=
by
  sorry

end least_number_to_subtract_l305_305625


namespace f_is_odd_l305_305820

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def f (x : ℝ) : ℝ := x * |x|

theorem f_is_odd : is_odd_function f :=
by sorry

end f_is_odd_l305_305820


namespace find_c_of_parabola_l305_305155

theorem find_c_of_parabola 
  (a b c : ℝ)
  (h_eq : ∀ y, -3 = a * (y - 1)^2 + b * (y - 1) - 3)
  (h1 : -1 = a * (3 - 1)^2 + b * (3 - 1) - 3) :
  c = -5/2 := by
  sorry

end find_c_of_parabola_l305_305155


namespace no_prime_p_such_that_22p2_plus_23_is_prime_l305_305233

theorem no_prime_p_such_that_22p2_plus_23_is_prime :
  ∀ p : ℕ, Prime p → ¬ Prime (22 * p ^ 2 + 23) :=
by
  sorry

end no_prime_p_such_that_22p2_plus_23_is_prime_l305_305233


namespace find_x_from_angles_l305_305870

theorem find_x_from_angles : ∀ (x : ℝ), (6 * x + 3 * x + 2 * x + x = 360) → x = 30 := by
  sorry

end find_x_from_angles_l305_305870


namespace find_distinct_numbers_l305_305399

theorem find_distinct_numbers (k l : ℕ) (h : 64 / k = 4 * (64 / l)) : k = 1 ∧ l = 4 :=
by
  sorry

end find_distinct_numbers_l305_305399


namespace neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0_l305_305018

theorem neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0 :
  ¬ (∃ x : ℝ, x > 0 ∧ x^2 - x ≤ 0) ↔ ∀ x : ℝ, x ≤ 0 → x^2 - x > 0 :=
by
    sorry

end neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0_l305_305018


namespace koby_sparklers_correct_l305_305264

-- Define the number of sparklers in each of Koby's boxes as a variable
variable (S : ℕ)

-- Specify the conditions
def koby_sparklers : ℕ := 2 * S
def koby_whistlers : ℕ := 2 * 5
def cherie_sparklers : ℕ := 8
def cherie_whistlers : ℕ := 9
def total_fireworks : ℕ := koby_sparklers S + koby_whistlers + cherie_sparklers + cherie_whistlers

-- The theorem to prove that the number of sparklers in each of Koby's boxes is 3
theorem koby_sparklers_correct : total_fireworks S = 33 → S = 3 := by
  sorry

end koby_sparklers_correct_l305_305264


namespace ratio_of_pond_to_field_area_l305_305294

theorem ratio_of_pond_to_field_area
  (l w : ℕ)
  (field_area pond_area : ℕ)
  (h1 : l = 2 * w)
  (h2 : l = 36)
  (h3 : pond_area = 9 * 9)
  (field_area_def : field_area = l * w)
  (pond_area_def : pond_area = 81) :
  pond_area / field_area = 1 / 8 := 
sorry

end ratio_of_pond_to_field_area_l305_305294


namespace algebra_expression_value_l305_305037

theorem algebra_expression_value (a b : ℝ) (h : (30^3) * a + 30 * b - 7 = 9) :
  (-30^3) * a + (-30) * b + 2 = -14 := 
by
  sorry

end algebra_expression_value_l305_305037


namespace find_point_A_coordinates_l305_305211

theorem find_point_A_coordinates :
  ∃ (A : ℝ × ℝ), (A.2 = 0) ∧ 
  (dist A (-3, 2) = dist A (4, -5)) →
  A = (2, 0) :=
by
-- We'll provide the explicit exact proof later
-- Proof steps would go here
sorry 

end find_point_A_coordinates_l305_305211


namespace melanie_total_dimes_l305_305882

-- Definitions based on the problem conditions
def initial_dimes : ℕ := 7
def dad_dimes : ℕ := 8
def mom_dimes : ℕ := 4

def total_dimes : ℕ := initial_dimes + dad_dimes + mom_dimes

-- Proof statement based on the correct answer
theorem melanie_total_dimes : total_dimes = 19 := by 
  -- Proof here is omitted as per instructions
  sorry

end melanie_total_dimes_l305_305882


namespace find_side_length_of_square_l305_305139

theorem find_side_length_of_square (n k : ℕ) (hk : k ≥ 1) (h : (n + k) * (n + k) - n * n = 47) : n = 23 :=
  sorry

end find_side_length_of_square_l305_305139


namespace evaluate_expression_l305_305339

theorem evaluate_expression : 
  (2 ^ 2015 + 2 ^ 2013 + 2 ^ 2011) / (2 ^ 2015 - 2 ^ 2013 + 2 ^ 2011) = 21 / 13 := 
by 
 sorry

end evaluate_expression_l305_305339


namespace range_of_f_l305_305132

noncomputable def f (x : ℝ) : ℝ := if x < 1 then 3 * x - 1 else 2 * x ^ 2

theorem range_of_f (a : ℝ) : (f (f a) = 2 * (f a) ^ 2) ↔ (a ≥ 2 / 3 ∨ a = 1 / 2) := 
  sorry

end range_of_f_l305_305132


namespace first_shaded_complete_cycle_seat_190_l305_305935

theorem first_shaded_complete_cycle_seat_190 : 
  ∀ (n : ℕ), (n ≥ 1) → 
  ∃ m : ℕ, 
    ((m ≥ n) ∧ 
    (∀ i : ℕ, (1 ≤ i ∧ i ≤ 12) → 
    ∃ k : ℕ, (k ≤ m ∧ (k * (k + 1) / 2) % 12 = (i - 1) % 12))) ↔ 
  ∃ m : ℕ, (m = 19 ∧ 190 = (m * (m + 1)) / 2) :=
by
  sorry

end first_shaded_complete_cycle_seat_190_l305_305935


namespace find_max_side_length_l305_305948

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l305_305948


namespace kiki_scarves_count_l305_305263

variable (money : ℝ) (scarf_cost : ℝ) (hat_spending_ratio : ℝ) (scarves : ℕ) (hats : ℕ)

-- Condition: Kiki has $90.
axiom kiki_money : money = 90

-- Condition: Kiki spends 60% of her money on hats.
axiom kiki_hat_spending_ratio : hat_spending_ratio = 0.60

-- Condition: Each scarf costs $2.
axiom scarf_price : scarf_cost = 2

-- Condition: Kiki buys twice as many hats as scarves.
axiom hat_scarf_relationship : hats = 2 * scarves

theorem kiki_scarves_count 
  (kiki_money : money = 90)
  (kiki_hat_spending_ratio : hat_spending_ratio = 0.60)
  (scarf_price : scarf_cost = 2)
  (hat_scarf_relationship : hats = 2 * scarves)
  : scarves = 18 := 
sorry

end kiki_scarves_count_l305_305263


namespace inequality_problem_l305_305530

theorem inequality_problem {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b + b * c + c * a = a + b + c) :
  a^2 + b^2 + c^2 + 2 * a * b * c ≥ 5 :=
sorry

end inequality_problem_l305_305530


namespace expected_yield_correct_l305_305001

/-- Define the problem variables and conditions -/
def steps_x : ℕ := 25
def steps_y : ℕ := 20
def step_length : ℝ := 2.5
def yield_per_sqft : ℝ := 0.75

/-- Calculate the dimensions in feet -/
def length_x := steps_x * step_length
def length_y := steps_y * step_length

/-- Calculate the area of the orchard -/
def area := length_x * length_y

/-- Calculate the expected yield of apples -/
def expected_yield := area * yield_per_sqft

/-- Prove the expected yield of apples is 2343.75 pounds -/
theorem expected_yield_correct : expected_yield = 2343.75 := sorry

end expected_yield_correct_l305_305001


namespace incorrect_counting_of_students_l305_305253

open Set

theorem incorrect_counting_of_students
  (total_students : ℕ)
  (english_only : ℕ)
  (german_only : ℕ)
  (french_only : ℕ)
  (english_german : ℕ)
  (english_french : ℕ)
  (german_french : ℕ)
  (all_three : ℕ)
  (reported_total : ℕ)
  (h_total_students : total_students = 100)
  (h_english_only : english_only = 30)
  (h_german_only : german_only = 23)
  (h_french_only : french_only = 50)
  (h_english_german : english_german = 10)
  (h_english_french : english_french = 8)
  (h_german_french : german_french = 20)
  (h_all_three : all_three = 5)
  (h_reported_total : reported_total = 100) :
  (english_only + german_only + french_only + english_german +
   english_french + german_french - 2 * all_three) ≠ reported_total :=
by
  sorry

end incorrect_counting_of_students_l305_305253


namespace quadratic_completion_l305_305277

theorem quadratic_completion (x d e f : ℤ) (h1 : 100*x^2 + 80*x - 144 = 0) (hd : d > 0) 
  (hde : (d * x + e)^2 = f) : d + e + f = 174 :=
sorry

end quadratic_completion_l305_305277


namespace inequality_solution_l305_305774

theorem inequality_solution (x : ℝ) (hx : x > 0) : (1 / x > 1) ↔ (0 < x ∧ x < 1) := 
sorry

end inequality_solution_l305_305774


namespace expression_value_l305_305926

theorem expression_value (x y z w : ℝ) (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  (2 * x + y) * (2 * z + w) = 15 := 
sorry

end expression_value_l305_305926


namespace x_squared_plus_y_squared_value_l305_305235

theorem x_squared_plus_y_squared_value (x y : ℝ) (h : (x^2 + y^2 + 1) * (x^2 + y^2 + 2) = 6) : x^2 + y^2 = 1 :=
by
  sorry

end x_squared_plus_y_squared_value_l305_305235


namespace rectangle_dimensions_l305_305082

theorem rectangle_dimensions (x : ℝ) (h : 3 * x * x = 8 * x) : (x = 8 / 3 ∧ 3 * x = 8) :=
by {
  sorry
}

end rectangle_dimensions_l305_305082


namespace weight_of_second_new_player_l305_305919

theorem weight_of_second_new_player 
  (total_weight_seven_players : ℕ)
  (average_weight_seven_players : ℕ)
  (total_players_with_new_players : ℕ)
  (average_weight_with_new_players : ℕ)
  (weight_first_new_player : ℕ)
  (W : ℕ) :
  total_weight_seven_players = 7 * average_weight_seven_players →
  total_players_with_new_players = 9 →
  average_weight_with_new_players = 106 →
  weight_first_new_player = 110 →
  (total_weight_seven_players + weight_first_new_player + W) / total_players_with_new_players = average_weight_with_new_players →
  W = 60 := 
by sorry

end weight_of_second_new_player_l305_305919


namespace part1_part2_l305_305842

theorem part1 (m : ℂ) : (m * (m + 2)).re = 0 ∧ (m^2 + m - 2).im ≠ 0 → m = 0 := by
  sorry

theorem part2 (m : ℝ) : (m * (m + 2) > 0 ∧ m^2 + m - 2 < 0) → 0 < m ∧ m < 1 := by
  sorry

end part1_part2_l305_305842


namespace product_of_consecutive_integers_sqrt_50_l305_305302

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), 49 < 50 ∧ 50 < 64 ∧ n = m + 1 ∧ m * n = 56 :=
by {
  let m := 7,
  let n := 8,
  have h1 : 49 < 50 := by norm_num,
  have h2 : 50 < 64 := by norm_num,
  exact ⟨m, n, h1, h2, rfl, by norm_num⟩,
  sorry -- proof skipped
}

end product_of_consecutive_integers_sqrt_50_l305_305302


namespace one_fourth_way_from_x1_to_x2_l305_305036

-- Definitions of the points
def x1 : ℚ := 1 / 5
def x2 : ℚ := 4 / 5

-- Problem statement: Prove that one fourth of the way from x1 to x2 is 7/20
theorem one_fourth_way_from_x1_to_x2 : (3 * x1 + 1 * x2) / 4 = 7 / 20 := by
  sorry

end one_fourth_way_from_x1_to_x2_l305_305036


namespace Elmer_vs_Milton_food_l305_305457

def Penelope_daily_food := 20  -- Penelope eats 20 pounds per day
def Greta_to_Penelope_ratio := 1 / 10  -- Greta eats 1/10 of what Penelope eats
def Milton_to_Greta_ratio := 1 / 100  -- Milton eats 1/100 of what Greta eats
def Elmer_to_Penelope_difference := 60  -- Elmer eats 60 pounds more than Penelope

def Greta_daily_food := Penelope_daily_food * Greta_to_Penelope_ratio
def Milton_daily_food := Greta_daily_food * Milton_to_Greta_ratio
def Elmer_daily_food := Penelope_daily_food + Elmer_to_Penelope_difference

theorem Elmer_vs_Milton_food :
  Elmer_daily_food = 4000 * Milton_daily_food := by
  sorry

end Elmer_vs_Milton_food_l305_305457


namespace smallest_composite_no_prime_factors_less_than_20_l305_305673

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l305_305673


namespace find_max_side_length_l305_305957

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l305_305957


namespace Johnson_family_seating_l305_305585

theorem Johnson_family_seating : 
  ∃ n : ℕ, number_of_ways_to_seat_Johnson_family = n ∧ n = 288 :=
sorry

end Johnson_family_seating_l305_305585


namespace positive_integer_count_l305_305651

theorem positive_integer_count (n : ℕ) :
  ∃ (count : ℕ), (count = 122) ∧ 
  (∀ (k : ℕ), 27 < k ∧ k < 150 → ((150 * k)^40 > k^80 ∧ k^80 > 3^240)) :=
sorry

end positive_integer_count_l305_305651


namespace valid_starting_days_count_l305_305811

def is_valid_starting_day (d : ℕ) : Prop :=
  (d % 7 = 3 ∨ d % 7 = 4 ∨ d % 7 = 5)

theorem valid_starting_days_count : 
  (finset.filter is_valid_starting_day (finset.range 7)).card = 3 :=
begin
  sorry
end

end valid_starting_days_count_l305_305811


namespace otimes_example_l305_305379

def otimes (a b : ℤ) : ℤ := a^2 - abs b

theorem otimes_example : otimes (-2) (-1) = 3 := by
  -- Define the variables
  let a := -2
  let b := -1
  -- Unfold the definition of otimes
  have h1 : otimes a b = a^2 - abs b := rfl
  -- Calculate a^2
  have h2 : a^2 = 4 := rfl
  -- Calculate abs b
  have h3 : abs b = 1 := rfl
  -- Calculate otimes a b
  show otimes a b = 3 from by
    rw [h1, h2, h3]
    rfl

end otimes_example_l305_305379


namespace pigeonhole_principle_example_l305_305773

theorem pigeonhole_principle_example :
  ∀ (S : Finset ℕ), (∀ (x : ℕ), x ∈ S ↔ x ≤100 ∧ 1≤ x) →
  ∃ (A : Finset ℕ), (A.card ≥ 15 ∧ A ⊆ S) → 
  (∃ a b c d ∈ A, a ≠ b ∧ c ≠ d ∧ a + b = c + d) ∨ 
  (∃ e f g ∈ A, e ≠ f ∧ f ≠ g ∧ e + f = 2 * g) :=
by
  -- implementation of the proof goes here
  sorry

end pigeonhole_principle_example_l305_305773


namespace billy_questions_third_hour_l305_305372

variable (x : ℝ)
variable (questions_in_first_hour : ℝ := x)
variable (questions_in_second_hour : ℝ := 1.5 * x)
variable (questions_in_third_hour : ℝ := 3 * x)
variable (total_questions_solved : ℝ := 242)

theorem billy_questions_third_hour (h : questions_in_first_hour + questions_in_second_hour + questions_in_third_hour = total_questions_solved) :
  questions_in_third_hour = 132 :=
by
  sorry

end billy_questions_third_hour_l305_305372


namespace find_t_over_q_l305_305720

theorem find_t_over_q
  (q r s v t : ℝ)
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : v / t = 4)
  (h4 : s / v = 1 / 3) :
  t / q = 1 / 2 := 
sorry

end find_t_over_q_l305_305720


namespace no_internal_angle_less_than_60_l305_305009

-- Define the concept of a Δ-curve
def delta_curve (K : Type) : Prop := sorry

-- Define the concept of a bicentric Δ-curve
def bicentric_delta_curve (K : Type) : Prop := sorry

-- Define the concept of internal angles of a Δ-curve
def has_internal_angle (K : Type) (A : ℝ) : Prop := sorry

-- The Lean statement for the problem
theorem no_internal_angle_less_than_60 (K : Type) 
  (h1 : delta_curve K) 
  (h2 : has_internal_angle K 60 ↔ bicentric_delta_curve K) :
  (∀ A < 60, ¬has_internal_angle K A) ∧ (has_internal_angle K 60 → bicentric_delta_curve K) := 
sorry

end no_internal_angle_less_than_60_l305_305009


namespace students_behind_Yoongi_l305_305047

theorem students_behind_Yoongi :
  ∀ (n : ℕ), n = 20 → ∀ (j y : ℕ), j = 1 → y = 2 → n - y = 18 :=
by
  intros n h1 j h2 y h3
  sorry

end students_behind_Yoongi_l305_305047


namespace women_attended_l305_305369

theorem women_attended (m w : ℕ) 
  (h_danced_with_4_women : ∀ (k : ℕ), k < m → k * 4 = 60)
  (h_danced_with_3_men : ∀ (k : ℕ), k < w → 3 * (k * (m / 3)) = 60)
  (h_men_count : m = 15) : 
  w = 20 := 
sorry

end women_attended_l305_305369


namespace original_square_side_length_l305_305138

theorem original_square_side_length :
  ∃ n k : ℕ, (n + k) * (n + k) - n * n = 47 ∧ k ≤ 5 ∧ k % 2 = 1 ∧ n = 23 :=
by
  sorry

end original_square_side_length_l305_305138


namespace determine_y_l305_305546

theorem determine_y (x y : ℝ) (h₁ : x^2 = y - 7) (h₂ : x = 7) : y = 56 :=
sorry

end determine_y_l305_305546


namespace half_angle_in_second_quadrant_l305_305443

def quadrant_of_half_alpha (α : ℝ) (hα1 : π < α) (hα2 : α < 3 * π / 2) (hcos : abs (Real.cos (α / 2)) = -Real.cos (α / 2)) : Prop :=
  π / 2 < α / 2 ∧ α / 2 < 3 * π / 4

theorem half_angle_in_second_quadrant (α : ℝ) (hα1 : π < α) (hα2 : α < 3 * π / 2) (hcos : abs (Real.cos (α / 2)) = -Real.cos (α / 2)) : quadrant_of_half_alpha α hα1 hα2 hcos :=
sorry

end half_angle_in_second_quadrant_l305_305443


namespace otimes_neg2_neg1_l305_305383

def otimes (a b : Int) : Int := a ^ 2 - Int.natAbs b

theorem otimes_neg2_neg1 : otimes (-2) (-1) = 3 :=
by
  rw otimes
  rfl

end otimes_neg2_neg1_l305_305383


namespace probability_team_B_wins_first_l305_305463

-- We define the conditions and the problem
def team_A_wins_series (games : List Bool) : Prop :=
  games.filter id.length = 4

def team_B_wins_series (games : List Bool) : Prop :=
  games.filter (λ x, not x).length = 4

def wins_the_series (games : List Bool) : Prop :=
  team_A_wins_series games ∨ team_B_wins_series games

-- B wins 3rd game (assuming 0-based indexing, so 3rd game is at index 2)
def third_game_B (games : List Bool) : Prop :=
  games.length > 2 ∧ games.get? 2 = some false

-- Condition: B wins the 3rd game
def condition := ∀ games : List Bool,
  team_A_wins_series games →
  third_game_B games →
  (∃ games' : List Bool, team_A_wins_series games' ∧ third_game_B games' ∧ games'.head = some false ∧
   (games.length = 7 → games' == games))

-- The theorem to prove the probability
theorem probability_team_B_wins_first :
  ∃ p : ℚ, p = 1 / 5 ∧
  (∀ (games : List Bool), team_A_wins_series games → third_game_B games →
  (∑' (games' : List Bool)
    (hA : team_A_wins_series games').filter (λ x, team_A_wins_series x ∧ third_game_B x ∧ x.head = some false) ∑'  (games' : List Bool):= p)
sorry

end probability_team_B_wins_first_l305_305463


namespace ellipse_equation_x_axis_foci_ellipse_equation_y_axis_foci_l305_305062

theorem ellipse_equation_x_axis_foci (x y : ℝ) (h : 3 * x + 4 * y = 12) :
  ∃ a b c : ℝ, c = 4 ∧ b = 3 ∧ a = 5 ∧ (x^2 / a^2) + (y^2 / b^2) = 1 := by
  sorry

theorem ellipse_equation_y_axis_foci (x y : ℝ) (h : 3 * x + 4 * y = 12) :
  ∃ a b c : ℝ, c = 3 ∧ b = 4 ∧ a = 5 ∧ (x^2 / b^2) + (y^2 / a^2) = 1 := by
  sorry

end ellipse_equation_x_axis_foci_ellipse_equation_y_axis_foci_l305_305062


namespace product_of_consecutive_integers_sqrt_50_l305_305305

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (m n : ℕ), 49 < 50 ∧ 50 < 64 ∧ n = m + 1 ∧ m * n = 56 :=
by {
  let m := 7,
  let n := 8,
  have h1 : 49 < 50 := by norm_num,
  have h2 : 50 < 64 := by norm_num,
  exact ⟨m, n, h1, h2, rfl, by norm_num⟩,
  sorry -- proof skipped
}

end product_of_consecutive_integers_sqrt_50_l305_305305


namespace total_boys_in_camp_l305_305425

theorem total_boys_in_camp (T : ℝ) 
  (h1 : 0.20 * T = number_of_boys_from_school_A)
  (h2 : 0.30 * number_of_boys_from_school_A = number_of_boys_study_science_from_school_A)
  (h3 : number_of_boys_from_school_A - number_of_boys_study_science_from_school_A = 42) :
  T = 300 := 
sorry

end total_boys_in_camp_l305_305425


namespace distance_light_in_50_years_l305_305907

/-- The distance light travels in one year, given in scientific notation -/
def distance_light_per_year : ℝ := 9.4608 * 10^12

/-- The distance light travels in 50 years is calculated -/
theorem distance_light_in_50_years :
  distance_light_per_year * 50 = 4.7304 * 10^14 :=
by
  -- the proof is not demanded, so we use sorry
  sorry

end distance_light_in_50_years_l305_305907


namespace polynomial_solution_exists_l305_305699

open Real

theorem polynomial_solution_exists
    (P : ℝ → ℝ → ℝ)
    (hP : ∃ (f : ℝ → ℝ), ∀ x y : ℝ, P x y = f (x + y) - f x - f y) :
  ∃ (q : ℝ → ℝ), ∀ x y : ℝ, P x y = q (x + y) - q x - q y := sorry

end polynomial_solution_exists_l305_305699


namespace cone_volume_l305_305174

theorem cone_volume (V_cyl : ℝ) (r h : ℝ) (h_cyl : V_cyl = 150 * Real.pi) :
  (1 / 3) * V_cyl = 50 * Real.pi :=
by
  rw [h_cyl]
  ring


end cone_volume_l305_305174


namespace rational_sqrt_condition_l305_305855

variable (r q n : ℚ)

theorem rational_sqrt_condition
  (h : (1 / (r + q * n) + 1 / (q + r * n) = 1 / (r + q))) : 
  ∃ x : ℚ, x^2 = (n - 3) / (n + 1) :=
sorry

end rational_sqrt_condition_l305_305855


namespace min_positive_period_f_mono_inc_interval_f_min_value_g_l305_305853

open Real

noncomputable def f(x : ℝ) := 2 * sqrt(3) * sin x * cos x + 1 - 2 * (sin x)^2
noncomputable def g(x : ℝ) := 2 * sin (4 * x + 5 * π / 6)

theorem min_positive_period_f : (∃ T > 0, ∀ x, f(x) = f(x + T)) ↔ T = π := sorry

theorem mono_inc_interval_f (k : ℤ) : ∀ x ∈ Icc (k * π - π / 3) (k * π + π / 6), 
  ∃ c, ∀ y ∈ Icc (k * π - π / 3) x, f(y) ≤ f(y + c) := sorry

theorem min_value_g 
  : ∃ x ∈ Icc 0 (π / 8), ∀ y ∈ Icc 0 (π / 8), g(x) ≤ g(y) ∧ g(x) = -sqrt(3) := sorry

end min_positive_period_f_mono_inc_interval_f_min_value_g_l305_305853


namespace Nina_saves_enough_to_buy_video_game_in_11_weeks_l305_305883

-- Definitions (directly from conditions)
def game_cost : ℕ := 50
def tax_rate : ℚ := 10 / 100
def sales_tax (cost : ℕ) (rate : ℚ) : ℚ := cost * rate
def total_cost (cost : ℕ) (tax : ℚ) : ℚ := cost + tax
def weekly_allowance : ℕ := 10
def savings_rate : ℚ := 1 / 2
def weekly_savings (allowance : ℕ) (rate : ℚ) : ℚ := allowance * rate
def weeks_to_save (total_cost : ℚ) (savings_per_week : ℚ) : ℚ := total_cost / savings_per_week

-- Theorem to prove
theorem Nina_saves_enough_to_buy_video_game_in_11_weeks :
  weeks_to_save
    (total_cost game_cost (sales_tax game_cost tax_rate))
    (weekly_savings weekly_allowance savings_rate) = 11 := by
-- We skip the proof for now, as per instructions
  sorry

end Nina_saves_enough_to_buy_video_game_in_11_weeks_l305_305883


namespace product_of_consecutive_integers_sqrt_50_l305_305332

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l305_305332


namespace remaining_money_l305_305275

-- Definitions
def cost_per_app : ℕ := 4
def num_apps : ℕ := 15
def total_money : ℕ := 66

-- Theorem
theorem remaining_money : total_money - (num_apps * cost_per_app) = 6 := by
  sorry

end remaining_money_l305_305275


namespace distance_between_foci_of_ellipse_l305_305657

theorem distance_between_foci_of_ellipse (x y : ℝ) :
  9 * x^2 + y^2 = 36 → 2 * real.sqrt (36 - 4) = 8 * real.sqrt 2 :=
by
  intro h
  calc
    2 * real.sqrt (36 - 4) = 2 * real.sqrt (32) : sorry
    ...                   = 2 * 4 * real.sqrt 2  : sorry
    ...                   = 8 * real.sqrt 2      : sorry

end distance_between_foci_of_ellipse_l305_305657


namespace total_cost_of_two_rackets_l305_305453

axiom racket_full_price : ℕ
axiom price_of_first_racket : racket_full_price = 60
axiom price_of_second_racket : racket_full_price / 2 = 30

theorem total_cost_of_two_rackets : 60 + 30 = 90 :=
sorry

end total_cost_of_two_rackets_l305_305453


namespace find_max_side_length_l305_305954

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l305_305954


namespace most_efficient_packing_l305_305576

theorem most_efficient_packing :
  ∃ box_size, 
  (box_size = 3 ∨ box_size = 6 ∨ box_size = 9) ∧ 
  (∀ q ∈ [21, 18, 15, 12, 9], q % box_size = 0) ∧
  box_size = 3 :=
by
  sorry

end most_efficient_packing_l305_305576


namespace Jina_mascots_total_l305_305874

theorem Jina_mascots_total :
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  teddies + bunnies + koala + additional_teddies = 51 :=
by
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  show teddies + bunnies + koala + additional_teddies = 51
  sorry

end Jina_mascots_total_l305_305874


namespace three_digit_odd_sum_count_l305_305364

def countOddSumDigits : Nat :=
  -- Count of three-digit numbers with an odd sum formed by (1, 2, 3, 4, 5)
  24

theorem three_digit_odd_sum_count :
  -- Guarantees that the count of three-digit numbers meeting the criteria is 24
  ∃ n : Nat, n = countOddSumDigits :=
by
  use 24
  sorry

end three_digit_odd_sum_count_l305_305364


namespace school_bought_50_cartons_of_markers_l305_305357

theorem school_bought_50_cartons_of_markers
  (n_puzzles : ℕ := 200)  -- the remaining amount after buying pencils
  (cost_per_carton_marker : ℕ := 4)  -- the cost per carton of markers
  :
  (n_puzzles / cost_per_carton_marker = 50) := -- the theorem to prove
by
  -- Provide skeleton proof strategy here
  sorry  -- details of the proof

end school_bought_50_cartons_of_markers_l305_305357


namespace spherical_to_rectangular_coordinates_l305_305068

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ), ρ = 6 → θ = 7 * Real.pi / 4 → φ = Real.pi / 4 →
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (3, -3, 3 * Real.sqrt 2) := by
  sorry

end spherical_to_rectangular_coordinates_l305_305068


namespace seokjin_paper_count_l305_305438

theorem seokjin_paper_count :
  ∀ (jimin_paper seokjin_paper : ℕ),
  jimin_paper = 41 →
  jimin_paper = seokjin_paper + 1 →
  seokjin_paper = 40 :=
by
  intros jimin_paper seokjin_paper h_jimin h_relation
  sorry

end seokjin_paper_count_l305_305438


namespace crossing_time_correct_l305_305486

def length_of_train : ℝ := 150 -- Length of the train in meters
def speed_of_train_km_per_hr : ℝ := 72 -- Speed of the train in km/hr
def length_of_bridge : ℝ := 132 -- Length of the bridge in meters

noncomputable def speed_of_train_m_per_s : ℝ := (speed_of_train_km_per_hr * 1000) / 3600 -- Speed of the train in m/s

noncomputable def time_to_cross_bridge : ℝ := (length_of_train + length_of_bridge) / speed_of_train_m_per_s -- Time in seconds

theorem crossing_time_correct : time_to_cross_bridge = 14.1 := by
  sorry

end crossing_time_correct_l305_305486


namespace find_real_medal_min_weighings_l305_305190

axiom has_9_medals : Prop
axiom one_real_medal : Prop
axiom real_medal_heavier : Prop
axiom has_balance_scale : Prop

theorem find_real_medal_min_weighings
  (h1 : has_9_medals)
  (h2 : one_real_medal)
  (h3 : real_medal_heavier)
  (h4 : has_balance_scale) :
  ∃ (minimum_weighings : ℕ), minimum_weighings = 2 := 
  sorry

end find_real_medal_min_weighings_l305_305190


namespace quadratic_roots_l305_305223

theorem quadratic_roots (a : ℝ) (k c : ℝ) : 
    (∀ x : ℝ, 2 * x^2 + k * x + c = 0 ↔ (x = 7 ∨ x = a)) →
    k = -2 * a - 14 ∧ c = 14 * a :=
by
  sorry

end quadratic_roots_l305_305223


namespace simplify_sqrt_expression_l305_305478

theorem simplify_sqrt_expression :
  ( (Real.sqrt 112 + Real.sqrt 567) / Real.sqrt 175) = 13 / 5 := by
  -- conditions for simplification
  have h1 : Real.sqrt 112 = 4 * Real.sqrt 7 := sorry
  have h2 : Real.sqrt 567 = 9 * Real.sqrt 7 := sorry
  have h3 : Real.sqrt 175 = 5 * Real.sqrt 7 := sorry
  
  -- Use the conditions to simplify the expression
  rw [h1, h2, h3]
  -- Further simplification to achieve the result 13 / 5
  sorry

end simplify_sqrt_expression_l305_305478


namespace track_width_l305_305212

theorem track_width (r1 r2 : ℝ) (h : 2 * π * r1 - 2 * π * r2 = 10 * π) : r1 - r2 = 5 :=
sorry

end track_width_l305_305212


namespace fraction_replaced_l305_305934

theorem fraction_replaced :
  ∃ x : ℚ, (0.60 * (1 - x) + 0.25 * x = 0.35) ∧ x = 5 / 7 := by
    sorry

end fraction_replaced_l305_305934


namespace angle_T_in_pentagon_l305_305430

theorem angle_T_in_pentagon (P Q R S T : ℝ) 
  (h1 : P = R) (h2 : P = T) (h3 : Q + S = 180) 
  (h4 : P + Q + R + S + T = 540) : T = 120 :=
by
  sorry

end angle_T_in_pentagon_l305_305430


namespace domain_of_f_l305_305908

noncomputable def f (x : ℝ) := 2 ^ (Real.sqrt (3 - x)) + 1 / (x - 1)

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, y = f x) ↔ (x ≤ 3 ∧ x ≠ 1) :=
by
  sorry

end domain_of_f_l305_305908


namespace not_perfect_square_of_sum_300_l305_305559

theorem not_perfect_square_of_sum_300 : ¬(∃ n : ℕ, n = 10^300 - 1 ∧ (∃ m : ℕ, n = m^2)) :=
by
  sorry

end not_perfect_square_of_sum_300_l305_305559


namespace find_c_l305_305599

noncomputable def parabola_equation (a b c y : ℝ) : ℝ :=
  a * y^2 + b * y + c

theorem find_c (a b c : ℝ) (h_vertex : (-4, 2) = (-4, 2)) (h_point : (-2, 4) = (-2, 4)) :
  ∃ c : ℝ, parabola_equation a b c 0 = -2 :=
  by {
    use -2,
    sorry
  }

end find_c_l305_305599


namespace remainder_19008_div_31_l305_305547

theorem remainder_19008_div_31 :
  ∀ (n : ℕ), (n = 432 * 44) → n % 31 = 5 :=
by
  intro n h
  sorry

end remainder_19008_div_31_l305_305547


namespace number_of_A_items_number_of_A_proof_l305_305352

def total_items : ℕ := 600
def ratio_A_B_C := (1, 2, 3)
def selected_items : ℕ := 120

theorem number_of_A_items (total_items : ℕ) (selected_items : ℕ) (rA rB rC : ℕ) (ratio_proof : rA + rB + rC = 6) : ℕ :=
  let total_ratio := rA + rB + rC
  let A_ratio := rA
  (selected_items * A_ratio) / total_ratio

theorem number_of_A_proof : number_of_A_items total_items selected_items 1 2 3 (rfl) = 20 := by
  sorry

end number_of_A_items_number_of_A_proof_l305_305352


namespace exists_points_with_small_distance_l305_305801

theorem exists_points_with_small_distance :
  ∃ A B : ℝ × ℝ, (A.2 = A.1^4) ∧ (B.2 = B.1^4 + B.1^2 + B.1 + 1) ∧ 
  (dist A B < 1 / 100) :=
by
  sorry

end exists_points_with_small_distance_l305_305801


namespace auditorium_rows_l305_305863

noncomputable def rows_in_auditorium : Nat :=
  let class1 := 30
  let class2 := 26
  let condition1 := ∃ row : Nat, row < class1 ∧ ∀ students_per_row : Nat, students_per_row ≤ row 
  let condition2 := ∃ empty_rows : Nat, empty_rows ≥ 3 ∧ ∀ students : Nat, students = class2 - empty_rows
  29

theorem auditorium_rows (n : Nat) (class1 : Nat) (class2 : Nat) (c1 : class1 ≥ n) (c2 : class2 ≤ n - 3)
  : n = 29 :=
by
  sorry

end auditorium_rows_l305_305863


namespace triangular_square_is_triangular_l305_305228

-- Definition of a triangular number
def is_triang_number (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x * (x + 1) / 2

-- The main theorem statement
theorem triangular_square_is_triangular :
  ∃ x : ℕ, 
    is_triang_number x ∧ 
    is_triang_number (x * x) :=
sorry

end triangular_square_is_triangular_l305_305228


namespace remainder_when_3y_divided_by_9_l305_305800

theorem remainder_when_3y_divided_by_9 (y : ℕ) (k : ℕ) (hy : y = 9 * k + 5) : (3 * y) % 9 = 6 :=
sorry

end remainder_when_3y_divided_by_9_l305_305800


namespace union_of_A_and_B_l305_305848

-- Define the sets A and B
def A := {x : ℝ | 0 < x ∧ x < 16}
def B := {y : ℝ | -1 < y ∧ y < 4}

-- Prove that A ∪ B = (-1, 16)
theorem union_of_A_and_B : A ∪ B = {z : ℝ | -1 < z ∧ z < 16} :=
by sorry

end union_of_A_and_B_l305_305848


namespace g_neither_even_nor_odd_l305_305737

noncomputable def g (x : ℝ) : ℝ := Real.log (2 * x)

theorem g_neither_even_nor_odd :
  (∀ x, g (-x) = g x → false) ∧ (∀ x, g (-x) = -g x → false) :=
by
  unfold g
  sorry

end g_neither_even_nor_odd_l305_305737


namespace smallest_composite_no_prime_factors_less_than_20_l305_305679

theorem smallest_composite_no_prime_factors_less_than_20 : 
  ∃ (n : ℕ), (∃ (a b : ℕ), n = a * b ∧ 1 < a ∧ 1 < b) ∧ (∀ p, nat.prime p → p ∣ n → 20 ≤ p) ∧ n = 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l305_305679


namespace max_side_length_l305_305965

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l305_305965


namespace polygon_sides_l305_305469

theorem polygon_sides (n : ℕ) 
  (H : (n * (n - 3)) / 2 = 3 * n) : n = 9 := 
sorry

end polygon_sides_l305_305469


namespace no_integer_solutions_exist_l305_305529

theorem no_integer_solutions_exist (n m : ℤ) : 
  (n ^ 2 - m ^ 2 = 250) → false := 
sorry 

end no_integer_solutions_exist_l305_305529


namespace total_cookies_l305_305731

theorem total_cookies (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) : bags * cookies_per_bag = 703 :=
by
  sorry

end total_cookies_l305_305731


namespace find_a1_a7_l305_305410

-- Definitions based on the problem conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def a_3_5_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 = -6

def a_2_6_condition (a : ℕ → ℝ) : Prop :=
  a 2 * a 6 = 8

-- The theorem we need to prove
theorem find_a1_a7 (a : ℕ → ℝ) (ha : is_geometric_sequence a) (h35 : a_3_5_condition a) (h26 : a_2_6_condition a) :
  a 1 + a 7 = -9 :=
sorry

end find_a1_a7_l305_305410


namespace fraction_sum_geq_zero_l305_305095

theorem fraction_sum_geq_zero (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b) + 1 / (b - c) + 4 / (c - a)) ≥ 0 := 
by 
  sorry

end fraction_sum_geq_zero_l305_305095


namespace otimes_neg2_neg1_l305_305385

def otimes (a b : Int) : Int := a ^ 2 - Int.natAbs b

theorem otimes_neg2_neg1 : otimes (-2) (-1) = 3 :=
by
  rw otimes
  rfl

end otimes_neg2_neg1_l305_305385


namespace triangle_angle_area_l305_305100

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x
variables {A B C : ℝ}
variables {BC : ℝ}
variables {S : ℝ}

theorem triangle_angle_area (hABC : A + B + C = π) (hBC : BC = 2) (h_fA : f A = 0) 
  (hA : A = π / 3) : S = Real.sqrt 3 :=
by
  -- Sorry, proof skipped
  sorry

end triangle_angle_area_l305_305100


namespace seashells_left_l305_305284

-- Definitions based on conditions
def initial_seashells : ℕ := 35
def seashells_given_away : ℕ := 18

-- Theorem stating the proof problem
theorem seashells_left (initial_seashells seashells_given_away : ℕ) : initial_seashells - seashells_given_away = 17 := 
    by
        sorry

end seashells_left_l305_305284


namespace find_ages_of_son_daughter_and_niece_l305_305354

theorem find_ages_of_son_daughter_and_niece
  (S : ℕ) (D : ℕ) (N : ℕ)
  (h1 : ∀ (M : ℕ), M = S + 24) 
  (h2 : ∀ (M : ℕ), 2 * (S + 2) = M + 2)
  (h3 : D = S / 2)
  (h4 : 2 * (D + 6) = 2 * S * 2 / 3)
  (h5 : N = S - 3)
  (h6 : 5 * N = 4 * S) :
  S = 22 ∧ D = 11 ∧ N = 19 := 
by 
  sorry

end find_ages_of_son_daughter_and_niece_l305_305354


namespace esperanzas_gross_monthly_salary_l305_305574

def rent : ℝ := 600
def savings : ℝ := 2000
def food_expenses : ℝ := (3/5) * rent
def mortgage_bill : ℝ := 3 * food_expenses
def taxes : ℝ := (2/5) * savings
def gross_monthly_salary : ℝ := rent + food_expenses + mortgage_bill + savings + taxes

theorem esperanzas_gross_monthly_salary : gross_monthly_salary = 4840 := by
  -- proof steps skipped
  sorry

end esperanzas_gross_monthly_salary_l305_305574


namespace find_function_l305_305080

theorem find_function (f : ℝ → ℝ)
  (h₁ : ∀ x : ℝ, x * (f (x + 1) - f x) = f x)
  (h₂ : ∀ x y : ℝ, |f x - f y| ≤ |x - y|) :
  ∃ k : ℝ, (∀ x : ℝ, f x = k * x) ∧ |k| ≤ 1 :=
sorry

end find_function_l305_305080


namespace no_solutions_to_inequality_l305_305689

theorem no_solutions_to_inequality (x : ℝ) : ¬(3 * x^2 + 9 * x + 12 ≤ 0) :=
by {
  intro h,
  -- Simplify the inequality by dividing each term by 3
  have h_simplified : x^2 + 3 * x + 4 ≤ 0 := by linarith,
  -- Compute the discriminant of the quadratic expression to show it's always positive
  let a := (1 : ℝ),
  let b := (3 : ℝ),
  let c := (4 : ℝ),
  let discriminant := b^2 - 4 * a * c,
  have h_discriminant : discriminant < 0 := by norm_num,
  -- Since discriminant is negative, the quadratic has no real roots, thus x^2 + 3x + 4 > 0
  have h_positive : ∀ x, x^2 + 3 * x + 4 > 0 := 
    by {
      intro x,
      apply (quadratic_not_negative_of_discriminant neg_discriminant).mp,
      exact h_discriminant,
    },
  exact absurd (show x^2 + 3 * x + 4 ≤ 0 from h_simplified) (lt_irrefl 0 (h_positive x)),
}

end no_solutions_to_inequality_l305_305689


namespace evaluate_expression_l305_305653

-- Define the given numbers as real numbers
def x : ℝ := 175.56
def y : ℝ := 54321
def z : ℝ := 36947
def w : ℝ := 1521

-- State the theorem to be proved
theorem evaluate_expression : (x / y) * (z / w) = 0.07845 :=
by 
  -- We skip the proof here
  sorry

end evaluate_expression_l305_305653


namespace isosceles_triangle_l305_305231

-- Given: sides a, b, c of a triangle satisfying a specific condition
-- To Prove: the triangle is isosceles (has at least two equal sides)

theorem isosceles_triangle (a b c : ℝ)
  (h : (c - b) / a + (a - c) / b + (b - a) / c = 0) :
  (a = b ∨ b = c ∨ a = c) :=
sorry

end isosceles_triangle_l305_305231


namespace mother_duck_multiple_of_first_two_groups_l305_305054

variables (num_ducklings : ℕ) (snails_first_batch : ℕ) (snails_second_batch : ℕ)
          (total_snails : ℕ) (mother_duck_snails : ℕ)

-- Given conditions
def conditions : Prop :=
  num_ducklings = 8 ∧ 
  snails_first_batch = 3 * 5 ∧ 
  snails_second_batch = 3 * 9 ∧ 
  total_snails = 294 ∧ 
  total_snails = snails_first_batch + snails_second_batch + 2 * mother_duck_snails ∧ 
  mother_duck_snails > 0

-- Our goal is to prove that the mother duck finds 3 times the snails the first two groups of ducklings find
theorem mother_duck_multiple_of_first_two_groups (h : conditions num_ducklings snails_first_batch snails_second_batch total_snails mother_duck_snails) : 
  mother_duck_snails / (snails_first_batch + snails_second_batch) = 3 :=
by 
  sorry

end mother_duck_multiple_of_first_two_groups_l305_305054


namespace product_of_roots_of_quadratics_l305_305451

noncomputable def product_of_roots : ℝ :=
  let r1 := 2021 / 2020
  let r2 := 2020 / 2019
  let r3 := 2019
  r1 * r2 * r3

theorem product_of_roots_of_quadratics (b : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, 2020 * x1 * x1 + b * x1 + 2021 = 0 ∧ 2020 * x2 * x2 + b * x2 + 2021 = 0) 
  (h2 : ∃ y1 y2 : ℝ, 2019 * y1 * y1 + b * y1 + 2020 = 0 ∧ 2019 * y2 * y2 + b * y2 + 2020 = 0) 
  (h3 : ∃ z1 z2 : ℝ, z1 * z1 + b * z1 + 2019 = 0 ∧ z1 * z1 + b * z2 + 2019 = 0) :
  product_of_roots = 2021 :=
by
  sorry

end product_of_roots_of_quadratics_l305_305451


namespace combined_tax_rate_l305_305199

theorem combined_tax_rate
  (john_income : ℝ) (john_tax_rate : ℝ)
  (ingrid_income : ℝ) (ingrid_tax_rate : ℝ)
  (h_john_income : john_income = 58000)
  (h_john_tax_rate : john_tax_rate = 0.30)
  (h_ingrid_income : ingrid_income = 72000)
  (h_ingrid_tax_rate : ingrid_tax_rate = 0.40) :
  ((john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income)) = 0.3553846154 :=
by
  sorry

end combined_tax_rate_l305_305199


namespace no_integer_n_squared_plus_one_div_by_seven_l305_305281

theorem no_integer_n_squared_plus_one_div_by_seven (n : ℤ) : ¬ (n^2 + 1) % 7 = 0 := 
sorry

end no_integer_n_squared_plus_one_div_by_seven_l305_305281


namespace alpha_beta_square_l305_305377

theorem alpha_beta_square (α β : ℝ) (h₁ : α^2 = 2*α + 1) (h₂ : β^2 = 2*β + 1) (hαβ : α ≠ β) :
  (α - β)^2 = 8 := 
sorry

end alpha_beta_square_l305_305377


namespace ted_speed_l305_305899

variables (T F : ℝ)

-- Ted runs two-thirds as fast as Frank
def condition1 : Prop := T = (2 / 3) * F

-- In two hours, Frank runs eight miles farther than Ted
def condition2 : Prop := 2 * F = 2 * T + 8

-- Prove that Ted runs at a speed of 8 mph
theorem ted_speed (h1 : condition1 T F) (h2 : condition2 T F) : T = 8 :=
by
  sorry

end ted_speed_l305_305899


namespace esperanza_gross_salary_l305_305573

def rent : ℕ := 600
def food_expenses (rent : ℕ) : ℕ := 3 * rent / 5
def mortgage_bill (food_expenses : ℕ) : ℕ := 3 * food_expenses
def savings : ℕ := 2000
def taxes (savings : ℕ) : ℕ := 2 * savings / 5
def total_expenses (rent food_expenses mortgage_bill taxes : ℕ) : ℕ :=
  rent + food_expenses + mortgage_bill + taxes
def gross_salary (total_expenses savings : ℕ) : ℕ :=
  total_expenses + savings

theorem esperanza_gross_salary : 
  gross_salary (total_expenses rent (food_expenses rent) (mortgage_bill (food_expenses rent)) (taxes savings)) savings = 4840 :=
by
  sorry

end esperanza_gross_salary_l305_305573


namespace two_a_minus_b_l305_305858

variables (a b : ℝ × ℝ)
variables (m : ℝ)

def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v

theorem two_a_minus_b 
  (ha : a = (1, -2))
  (hb : b = (m, 4))
  (h_parallel : parallel a b) :
  2 • a - b = (4, -8) :=
sorry

end two_a_minus_b_l305_305858


namespace remaining_integers_after_removal_l305_305056

open Finset

def T : Finset ℕ := range 81 \ {0}

def is_multiple_of (n k : ℕ) : Prop := ∃ m : ℕ, k = m * n

def multiples_of (n : ℕ) (s : Finset ℕ) : Finset ℕ :=
  s.filter (λ x, is_multiple_of n x)

theorem remaining_integers_after_removal : 
  let T := range 81 \ {0} in
  let multiples_of_4 := multiples_of 4 T in
  let multiples_of_5 := multiples_of 5 T in
  let multiples_of_20 := multiples_of 20 T in
  let removed := multiples_of_4 ∪ multiples_of_5 in
  (T.card - removed.card + multiples_of_20.card = 48) := 
by
  sorry

end remaining_integers_after_removal_l305_305056


namespace max_side_of_triangle_exists_max_side_of_elevent_l305_305976

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l305_305976


namespace committee_count_l305_305234

theorem committee_count (total_students : ℕ) (include_students : ℕ) (choose_students : ℕ) :
  total_students = 8 → include_students = 2 → choose_students = 3 →
  Nat.choose (total_students - include_students) choose_students = 20 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end committee_count_l305_305234


namespace minimum_value_frac_l305_305270

theorem minimum_value_frac (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p + q + r = 2) :
  (p + q) / (p * q * r) ≥ 9 :=
sorry

end minimum_value_frac_l305_305270


namespace quadratic_roots_range_no_real_k_for_reciprocal_l305_305707

theorem quadratic_roots_range (k : ℝ) (h : 12 * k + 4 > 0) : k > -1 / 3 ∧ k ≠ 0 :=
by
  sorry

theorem no_real_k_for_reciprocal (k : ℝ) : ¬∃ (x1 x2 : ℝ), (kx^2 - 2*(k+1)*x + k-1 = 0) ∧ (1/x1 + 1/x2 = 0) :=
by
  sorry

end quadratic_roots_range_no_real_k_for_reciprocal_l305_305707


namespace cube_surface_area_l305_305343

theorem cube_surface_area (V : ℝ) (hV : V = 125) : ∃ A : ℝ, A = 25 :=
by
  sorry

end cube_surface_area_l305_305343


namespace numberOfBookshelves_l305_305823

-- Define the conditions as hypotheses
def numBooks : ℕ := 23
def numMagazines : ℕ := 61
def totalItems : ℕ := 2436

-- Define the number of items per bookshelf
def itemsPerBookshelf : ℕ := numBooks + numMagazines

-- State the theorem to be proven
theorem numberOfBookshelves (bookshelves : ℕ) :
  itemsPerBookshelf * bookshelves = totalItems → 
  bookshelves = 29 :=
by
  -- placeholder for proof
  sorry

end numberOfBookshelves_l305_305823


namespace max_side_of_triangle_exists_max_side_of_elevent_l305_305983

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_side_of_triangle (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a + b + c = 24) (h_triangle : is_valid_triangle a b c) :
  max a (max b c) ≤ 11 :=
sorry

theorem exists_max_side_of_elevent (h₄ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c 
  ∧ a + b + c = 24 ∧ is_valid_triangle a b c 
  ∧ max a (max b c) = 11 :=
sorry

end max_side_of_triangle_exists_max_side_of_elevent_l305_305983


namespace randy_trips_l305_305760

def trips_per_month
  (initial : ℕ) -- Randy initially had $200 in his piggy bank
  (final : ℕ)   -- Randy had $104 left in his piggy bank after a year
  (spend_per_trip : ℕ) -- Randy spends $2 every time he goes to the store
  (months_in_year : ℕ) -- Number of months in a year, which is 12
  (total_trips_per_year : ℕ) -- Total trips he makes in a year
  (trips_per_month : ℕ) -- Trips to the store every month
  : Prop :=
  initial = 200 ∧ final = 104 ∧ spend_per_trip = 2 ∧ months_in_year = 12 ∧
  total_trips_per_year = (initial - final) / spend_per_trip ∧ 
  trips_per_month = total_trips_per_year / months_in_year ∧
  trips_per_month = 4

theorem randy_trips :
  trips_per_month 200 104 2 12 ((200 - 104) / 2) (48 / 12) :=
by 
  sorry

end randy_trips_l305_305760


namespace product_of_consecutive_integers_between_sqrt_50_l305_305322

theorem product_of_consecutive_integers_between_sqrt_50 :
  ∃ (m n : ℕ), (m < n) ∧ (sqrt 50 ∈ set.Icc (m : ℝ) (n : ℝ)) ∧ (m * n = 56) := by
  sorry

end product_of_consecutive_integers_between_sqrt_50_l305_305322


namespace avg_move_to_california_l305_305557

noncomputable def avg_people_per_hour (total_people : ℕ) (total_days : ℕ) : ℕ :=
  let total_hours := total_days * 24
  let avg_per_hour := total_people / total_hours
  let remainder := total_people % total_hours
  if remainder * 2 < total_hours then avg_per_hour else avg_per_hour + 1

theorem avg_move_to_california : avg_people_per_hour 3500 5 = 29 := by
  sorry

end avg_move_to_california_l305_305557


namespace product_of_consecutive_integers_sqrt_50_l305_305327

theorem product_of_consecutive_integers_sqrt_50 :
  (∃ (a b : ℕ), a < b ∧ b = a + 1 ∧ a ^ 2 < 50 ∧ 50 < b ^ 2 ∧ a * b = 56) := sorry

end product_of_consecutive_integers_sqrt_50_l305_305327


namespace sum_of_prime_factors_of_2_to_10_minus_1_l305_305917

theorem sum_of_prime_factors_of_2_to_10_minus_1 :
  let n := 2^10 - 1,
      factors := [31, 3, 11] in
  (n = factors.prod) ∧ (factors.all Prime) → factors.sum = 45 :=
by
  let n := 2^10 - 1
  let factors := [31, 3, 11]
  have fact_prod : n = factors.prod := by sorry
  have all_prime : factors.all Prime := by sorry
  have sum_factors : factors.sum = 45 := by sorry
  exact ⟨fact_prod, all_prime, sum_factors⟩

end sum_of_prime_factors_of_2_to_10_minus_1_l305_305917


namespace max_value_of_expression_l305_305444

noncomputable def max_expression_value (x y : ℝ) :=
  x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  max_expression_value x y ≤ 961 / 8 :=
sorry

end max_value_of_expression_l305_305444


namespace distance_between_foci_of_ellipse_l305_305658

theorem distance_between_foci_of_ellipse (a b : ℝ) (ha : a = 2) (hb : b = 6) :
  ∀ (x y : ℝ), 9 * x^2 + y^2 = 36 → 2 * Real.sqrt (b^2 - a^2) = 8 * Real.sqrt 2 :=
by
  intros x y h
  sorry

end distance_between_foci_of_ellipse_l305_305658


namespace smallest_composite_no_prime_factors_less_than_20_l305_305662

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l305_305662


namespace a9_proof_l305_305106

variable {a : ℕ → ℝ}

-- Conditions
axiom a1 : a 1 = 1
axiom an_recurrence : ∀ n > 1, a n = (a (n - 1)) * 2^(n - 1)

-- Goal
theorem a9_proof : a 9 = 2^36 := 
by 
  sorry

end a9_proof_l305_305106


namespace two_digit_number_l305_305482

theorem two_digit_number (x y : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (h1 : x^2 + y^2 = 10*x + y + 11) (h2 : 2*x*y = 10*x + y - 5) :
  10*x + y = 95 ∨ 10*x + y = 15 := 
sorry

end two_digit_number_l305_305482


namespace smallest_composite_no_prime_factors_less_than_20_l305_305672

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l305_305672


namespace geometric_seq_a3_equals_3_l305_305698

variable {a : ℕ → ℝ}
variable (h_geometric : ∀ m n p q, m + n = p + q → a m * a n = a p * a q)
variable (h_pos : ∀ n, n > 0 → a n > 0)
variable (h_cond : a 2 * a 4 = 9)

theorem geometric_seq_a3_equals_3 : a 3 = 3 := by
  sorry

end geometric_seq_a3_equals_3_l305_305698


namespace find_max_side_length_l305_305950

noncomputable def max_side_length (a b c : ℕ) : ℕ :=
  if a + b + c = 24 ∧ a < b ∧ b < c ∧ a + b > c ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) then c else 0

theorem find_max_side_length
  (a b c : ℕ)
  (h₁ : a ≠ b)
  (h₂ : b ≠ c)
  (h₃ : a ≠ c)
  (h₄ : a + b + c = 24)
  (h₅ : a < b)
  (h₆ : b < c)
  (h₇ : a + b > c) :
  max_side_length a b c = 10 :=
sorry

end find_max_side_length_l305_305950


namespace find_divisor_l305_305527

-- Define the initial number
def num := 1387

-- Define the number to subtract to make it divisible by some divisor
def least_subtract := 7

-- Define the resulting number after subtraction
def remaining_num := num - least_subtract

-- Define the divisor
def divisor := 23

-- The statement to prove: 1380 is divisible by 23
theorem find_divisor (num_subtract_div : num - least_subtract = remaining_num) 
                     (remaining_divisor : remaining_num = 1380) : 
                     ∃ k : ℕ, 1380 = k * divisor := by
  sorry

end find_divisor_l305_305527


namespace rose_paid_after_discount_l305_305283

-- Define the conditions as given in the problem statement
def original_price : ℕ := 10
def discount_rate : ℕ := 10

-- Define the theorem that needs to be proved
theorem rose_paid_after_discount : 
  original_price - (original_price * discount_rate / 100) = 9 :=
by
  -- Here we skip the proof with sorry
  sorry

end rose_paid_after_discount_l305_305283


namespace katya_sum_greater_than_masha_l305_305134

theorem katya_sum_greater_than_masha (a b c d : ℝ) (h : a + b + c + d = 2021) :
  ((a+1)*(b+1) + (b+1)*(c+1) + (c+1)*(d+1) + (d+1)*(a+1)) - (a*b + b*c + c*d + d*a) = 4046 := by
  sorry

end katya_sum_greater_than_masha_l305_305134


namespace marbles_in_jar_l305_305428

theorem marbles_in_jar (g y p : ℕ) (h1 : y + p = 7) (h2 : g + p = 10) (h3 : g + y = 5) :
  g + y + p = 11 :=
sorry

end marbles_in_jar_l305_305428


namespace part1_part2_l305_305712

noncomputable def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1 (a : ℝ) (h : a = 2) :
  ∀ x : ℝ, f x a ≥ 4 ↔ x ≤ (3 / 2 : ℝ) ∨ x ≥ (11 / 2 : ℝ) :=
by 
  rw h
  sorry

theorem part2 (h : ∀ x a : ℝ, f x a ≥ 4) :
  ∀ a : ℝ, (a - 1)^2 ≥ 4 ↔ a ≤ -1 ∨ a ≥ 3 :=
by 
  sorry

end part1_part2_l305_305712


namespace short_trees_after_planting_l305_305779

-- Defining the conditions as Lean definitions
def current_short_trees : Nat := 3
def newly_planted_short_trees : Nat := 9

-- Defining the question (assertion to prove) with the expected answer
theorem short_trees_after_planting : current_short_trees + newly_planted_short_trees = 12 := by
  sorry

end short_trees_after_planting_l305_305779


namespace real_root_of_P_l305_305021

noncomputable def P : ℕ → ℝ → ℝ
| 0, x => 0
| 1, x => x
| n+2, x => x * P (n + 1) x + (1 - x) * P n x

theorem real_root_of_P (n : ℕ) (hn : 1 ≤ n) : ∀ x : ℝ, P n x = 0 → x = 0 := 
by 
  sorry

end real_root_of_P_l305_305021


namespace smallest_composite_no_prime_factors_less_than_20_is_529_l305_305668

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end smallest_composite_no_prime_factors_less_than_20_is_529_l305_305668


namespace expression_equals_five_l305_305089

-- Define the innermost expression
def inner_expr : ℤ := -|( -2 + 3 )|

-- Define the next layer of the expression
def middle_expr : ℤ := |(inner_expr) - 2|

-- Define the outer expression
def outer_expr : ℤ := |middle_expr + 2|

-- The proof problem statement (in this case, without the proof)
theorem expression_equals_five : outer_expr = 5 :=
by
  -- Insert precise conditions directly from the problem statement
  have h_inner : inner_expr = -|1| := by sorry
  have h_middle : middle_expr = |-1 - 2| := by sorry
  have h_outer : outer_expr = |(-3) + 2| := by sorry
  -- The final goal that needs to be proven
  sorry

end expression_equals_five_l305_305089


namespace min_value_exp_sum_l305_305404

theorem min_value_exp_sum (a b : ℝ) (h : a + b = 2) : 3^a + 3^b ≥ 6 :=
by sorry

end min_value_exp_sum_l305_305404


namespace triangle_land_area_l305_305017

theorem triangle_land_area :
  let base_cm := 12
  let height_cm := 9
  let scale_cm_to_miles := 3
  let square_mile_to_acres := 640
  let area_cm2 := (1 / 2 : Float) * base_cm * height_cm
  let area_miles2 := area_cm2 * (scale_cm_to_miles ^ 2)
  let area_acres := area_miles2 * square_mile_to_acres
  area_acres = 311040 :=
by
  -- Skipped proofs
  sorry

end triangle_land_area_l305_305017


namespace smallest_composite_no_prime_factors_less_than_20_l305_305663

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l305_305663


namespace lana_total_spending_l305_305265

noncomputable def general_admission_cost : ℝ := 6
noncomputable def vip_cost : ℝ := 10
noncomputable def premium_cost : ℝ := 15

noncomputable def num_general_admission_tickets : ℕ := 6
noncomputable def num_vip_tickets : ℕ := 2
noncomputable def num_premium_tickets : ℕ := 1

noncomputable def discount_general_admission : ℝ := 0.10
noncomputable def discount_vip : ℝ := 0.15

noncomputable def total_spending (gen_cost : ℝ) (vip_cost : ℝ) (prem_cost : ℝ) (gen_num : ℕ) (vip_num : ℕ) (prem_num : ℕ) (gen_disc : ℝ) (vip_disc : ℝ) : ℝ :=
  let general_cost := gen_cost * gen_num
  let general_discount := general_cost * gen_disc
  let discounted_general_cost := general_cost - general_discount
  let vip_cost_total := vip_cost * vip_num
  let vip_discount := vip_cost_total * vip_disc
  let discounted_vip_cost := vip_cost_total - vip_discount
  let premium_cost_total := prem_cost * prem_num
  discounted_general_cost + discounted_vip_cost + premium_cost_total

theorem lana_total_spending : total_spending general_admission_cost vip_cost premium_cost num_general_admission_tickets num_vip_tickets num_premium_tickets discount_general_admission discount_vip = 64.40 := 
sorry

end lana_total_spending_l305_305265


namespace travel_time_difference_l305_305279

theorem travel_time_difference :
  (160 / 40) - (280 / 40) = 3 := by
  sorry

end travel_time_difference_l305_305279


namespace negation_of_universal_statement_l305_305769

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x^2 ≠ x) ↔ ∃ x : ℝ, x^2 = x :=
by sorry

end negation_of_universal_statement_l305_305769


namespace volume_inside_sphere_outside_cylinder_l305_305499

noncomputable def volumeDifference (r_cylinder base_radius_sphere : ℝ) :=
  let height := 4 * Real.sqrt 5
  let V_sphere := (4/3) * Real.pi * base_radius_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * height
  V_sphere - V_cylinder

theorem volume_inside_sphere_outside_cylinder
  (base_radius_sphere r_cylinder : ℝ) (h_base_radius_sphere : base_radius_sphere = 6) (h_r_cylinder : r_cylinder = 4) :
  volumeDifference r_cylinder base_radius_sphere = (288 - 64 * Real.sqrt 5) * Real.pi := by
  sorry

end volume_inside_sphere_outside_cylinder_l305_305499


namespace initial_goldfish_correct_l305_305753

-- Define the constants related to the conditions
def weekly_die := 5
def weekly_purchase := 3
def final_goldfish := 4
def weeks := 7

-- Define the initial number of goldfish that we need to prove
def initial_goldfish := 18

-- The proof statement: initial_goldfish - weekly_change * weeks = final_goldfish
theorem initial_goldfish_correct (G : ℕ)
  (h : G - weeks * (weekly_purchase - weekly_die) = final_goldfish) :
  G = initial_goldfish := by
  sorry

end initial_goldfish_correct_l305_305753


namespace max_side_length_l305_305972

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l305_305972


namespace correct_system_of_equations_l305_305595

variable (x y : ℕ) -- We assume non-negative numbers for counts of chickens and rabbits

theorem correct_system_of_equations :
  (x + y = 35) ∧ (2 * x + 4 * y = 94) ↔
  (∃ (a b : ℕ), a = x ∧ b = y) :=
by
  sorry

end correct_system_of_equations_l305_305595


namespace find_side_length_of_square_l305_305140

theorem find_side_length_of_square (n k : ℕ) (hk : k ≥ 1) (h : (n + k) * (n + k) - n * n = 47) : n = 23 :=
  sorry

end find_side_length_of_square_l305_305140


namespace rectangle_length_l305_305344

theorem rectangle_length :
  ∀ (side : ℕ) (width : ℕ) (length : ℕ), 
  side = 4 → 
  width = 8 → 
  side * side = width * length → 
  length = 2 := 
by
  -- sorry to skip the proof
  intros side width length h1 h2 h3
  sorry

end rectangle_length_l305_305344


namespace graph_depicts_one_line_l305_305504

theorem graph_depicts_one_line {x y : ℝ} :
  (x - 1) ^ 2 * (x + y - 2) = (y - 1) ^ 2 * (x + y - 2) →
  ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b :=
by
  intros h
  sorry

end graph_depicts_one_line_l305_305504


namespace expression_equals_five_l305_305088

-- Define the innermost expression
def inner_expr : ℤ := -|( -2 + 3 )|

-- Define the next layer of the expression
def middle_expr : ℤ := |(inner_expr) - 2|

-- Define the outer expression
def outer_expr : ℤ := |middle_expr + 2|

-- The proof problem statement (in this case, without the proof)
theorem expression_equals_five : outer_expr = 5 :=
by
  -- Insert precise conditions directly from the problem statement
  have h_inner : inner_expr = -|1| := by sorry
  have h_middle : middle_expr = |-1 - 2| := by sorry
  have h_outer : outer_expr = |(-3) + 2| := by sorry
  -- The final goal that needs to be proven
  sorry

end expression_equals_five_l305_305088


namespace smallest_positive_multiple_l305_305615

theorem smallest_positive_multiple (a : ℕ) :
  (37 * a) % 97 = 7 → 37 * a = 481 :=
sorry

end smallest_positive_multiple_l305_305615


namespace ratio_of_bases_l305_305736

theorem ratio_of_bases 
(AB CD : ℝ) 
(h_trapezoid : AB < CD) 
(h_AC : ∃ k : ℝ, k = 2 * CD ∧ k = AC) 
(h_altitude : AB = (D - foot)) : 
AB / CD = 3 := 
sorry

end ratio_of_bases_l305_305736


namespace weight_of_original_piece_of_marble_l305_305944

theorem weight_of_original_piece_of_marble (W : ℝ) 
  (h1 : W > 0)
  (h2 : (0.75 * 0.56 * W) = 105) : 
  W = 250 :=
by
  sorry

end weight_of_original_piece_of_marble_l305_305944


namespace quadruple_application_of_h_l305_305829

-- Define the function as specified in the condition
def h (N : ℝ) : ℝ := 0.6 * N + 2

-- State the theorem
theorem quadruple_application_of_h : h (h (h (h 40))) = 9.536 :=
  by
    sorry

end quadruple_application_of_h_l305_305829


namespace max_triangle_side_l305_305993

-- Definitions of conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def has_perimeter (a b c : ℕ) (p : ℕ) : Prop :=
  a + b + c = p

def different_integers (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main theorem to prove
theorem max_triangle_side (a b c : ℕ) (h_triangle : is_triangle a b c)
                         (h_perimeter : has_perimeter a b c 24)
                         (h_diff : different_integers a b c) :
  c ≤ 11 :=
sorry

end max_triangle_side_l305_305993


namespace find_tan_half_angle_l305_305099

variable {α : Real} (h₁ : Real.sin α = -24 / 25) (h₂ : α ∈ Set.Ioo (π:ℝ) (3 * π / 2))

theorem find_tan_half_angle : Real.tan (α / 2) = -4 / 3 :=
sorry

end find_tan_half_angle_l305_305099


namespace floor_neg_seven_thirds_l305_305521

theorem floor_neg_seven_thirds : ⌊-7 / 3⌋ = -3 :=
sorry

end floor_neg_seven_thirds_l305_305521


namespace given_polynomial_l305_305718

noncomputable def f (x : ℝ) := x^3 - 2

theorem given_polynomial (x : ℝ) : 
  8 * f (x^3) - x^6 * f (2 * x) - 2 * f (x^2) + 12 = 0 :=
by
  sorry

end given_polynomial_l305_305718


namespace censusSurveys_l305_305822

-- Definitions corresponding to the problem conditions
inductive Survey where
  | TVLifespan
  | ManuscriptReview
  | PollutionInvestigation
  | StudentSizeSurvey

open Survey

-- The aim is to identify which surveys are more suitable for a census.
def suitableForCensus (s : Survey) : Prop :=
  match s with
  | TVLifespan => False  -- Lifespan destruction implies sample survey.
  | ManuscriptReview => True  -- Significant and needs high accuracy, thus census.
  | PollutionInvestigation => False  -- Broad scope implies sample survey.
  | StudentSizeSurvey => True  -- Manageable scope makes census appropriate.

-- The theorem to be formalized.
theorem censusSurveys : (suitableForCensus ManuscriptReview) ∧ (suitableForCensus StudentSizeSurvey) :=
  by sorry

end censusSurveys_l305_305822


namespace intersection_eq_zero_l305_305130

def M := { x : ℤ | abs (x - 3) < 4 }
def N := { x : ℤ | x^2 + x - 2 < 0 }

theorem intersection_eq_zero : M ∩ N = {0} := 
  by
    sorry

end intersection_eq_zero_l305_305130


namespace triangle_final_position_after_rotation_l305_305360

-- Definitions for the initial conditions
def square_rolls_clockwise_around_octagon : Prop := 
  true -- placeholder definition, assume this defines the motion correctly

def triangle_initial_position : ℕ := 0 -- representing bottom as 0

-- Defining the proof problem
theorem triangle_final_position_after_rotation :
  square_rolls_clockwise_around_octagon →
  triangle_initial_position = 0 →
  triangle_initial_position = 0 :=
by
  intros
  sorry

end triangle_final_position_after_rotation_l305_305360


namespace imaginary_number_condition_fourth_quadrant_condition_l305_305841

-- Part 1: Prove that if \( z \) is purely imaginary, then \( m = 0 \)
theorem imaginary_number_condition (m : ℝ) :
  (m * (m + 2) = 0) ∧ (m^2 + m - 2 ≠ 0) → m = 0 :=
by
  sorry

-- Part 2: Prove that if \( z \) is in the fourth quadrant, then \( 0 < m < 1 \)
theorem fourth_quadrant_condition (m : ℝ) :
  (m * (m + 2) > 0) ∧ (m^2 + m - 2 < 0) → (0 < m ∧ m < 1) :=
by
  sorry

end imaginary_number_condition_fourth_quadrant_condition_l305_305841


namespace find_xy_l305_305421

variable (x y : ℝ)

theorem find_xy (hx : x ≠ 0) (hy : y ≠ 0) (h : (3 / x) * (2 / y) = 1 / 3) : x * y = 18 := by
  sorry

end find_xy_l305_305421


namespace reciprocal_neg_half_l305_305172

theorem reciprocal_neg_half : 1 / (-1 / 2 : ℝ) = (-2 : ℝ) :=
by
  sorry

end reciprocal_neg_half_l305_305172


namespace value_of_x_plus_inv_x_l305_305121

theorem value_of_x_plus_inv_x (x : ℝ) (h : x + (1 / x) = v) (hr : x^2 + (1 / x)^2 = 23) : v = 5 :=
sorry

end value_of_x_plus_inv_x_l305_305121


namespace possible_even_and_odd_functions_l305_305794

def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem possible_even_and_odd_functions :
  ∃ p q : ℝ → ℝ, is_even_function p ∧ is_odd_function (p ∘ q) ∧ (¬(∀ x, p (q x) = 0)) :=
by
  sorry

end possible_even_and_odd_functions_l305_305794


namespace gcd_min_value_l305_305419

theorem gcd_min_value {a b c : ℕ} (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (gcd_ab : Nat.gcd a b = 210) (gcd_ac : Nat.gcd a c = 770) : Nat.gcd b c = 10 :=
sorry

end gcd_min_value_l305_305419


namespace parabola_focus_directrix_eq_l305_305768

open Real

def distance (p : ℝ × ℝ) (l : ℝ) : ℝ := abs (p.fst - l)

def parabola_eq (focus_x focus_y l : ℝ) : Prop :=
  ∀ x y, (distance (x, y) focus_x = distance (x, y) l) ↔ y^2 = 2 * x - 1

theorem parabola_focus_directrix_eq :
  parabola_eq 1 0 0 :=
by
  sorry

end parabola_focus_directrix_eq_l305_305768


namespace Dabbie_spends_99_dollars_l305_305378

noncomputable def total_cost_turkeys (w1 w2 w3 w4 : ℝ) (cost_per_kg : ℝ) : ℝ :=
  (w1 + w2 + w3 + w4) * cost_per_kg

theorem Dabbie_spends_99_dollars :
  let w1 := 6
  let w2 := 9
  let w3 := 2 * w2
  let w4 := (w1 + w2 + w3) / 2
  let cost_per_kg := 2
  total_cost_turkeys w1 w2 w3 w4 cost_per_kg = 99 := 
by
  sorry

end Dabbie_spends_99_dollars_l305_305378


namespace bhishma_speed_l305_305064

-- Given definitions based on conditions
def track_length : ℝ := 600
def bruce_speed : ℝ := 30
def time_meet : ℝ := 90

-- Main theorem we want to prove
theorem bhishma_speed : ∃ v : ℝ, v = 23.33 ∧ (bruce_speed * time_meet) = (v * time_meet + track_length) :=
  by
    sorry

end bhishma_speed_l305_305064


namespace initial_thickness_of_blanket_l305_305394

theorem initial_thickness_of_blanket (T : ℝ)
  (h : ∀ n, n = 4 → T * 2^n = 48) : T = 3 :=
by
  have h4 := h 4 rfl
  sorry

end initial_thickness_of_blanket_l305_305394


namespace max_side_length_l305_305968

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l305_305968


namespace final_score_proof_l305_305073

def final_score (bullseye_points : ℕ) (miss_points : ℕ) (half_bullseye_points : ℕ) : ℕ :=
  bullseye_points + miss_points + half_bullseye_points

theorem final_score_proof : final_score 50 0 25 = 75 :=
by
  -- Considering the given conditions
  -- bullseye_points = 50
  -- miss_points = 0
  -- half_bullseye_points = half of bullseye_points = 25
  -- Summing them up: 50 + 0 + 25 = 75
  sorry

end final_score_proof_l305_305073


namespace max_side_length_l305_305967

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l305_305967


namespace dress_total_selling_price_l305_305631

theorem dress_total_selling_price (original_price discount_rate tax_rate : ℝ) 
  (h1 : original_price = 100) (h2 : discount_rate = 0.30) (h3 : tax_rate = 0.15) : 
  (original_price * (1 - discount_rate) * (1 + tax_rate)) = 80.5 := by
  sorry

end dress_total_selling_price_l305_305631


namespace triangle_type_l305_305561

theorem triangle_type (A B C : ℝ) (a b c : ℝ)
  (h1 : B = 30) 
  (h2 : c = 15) 
  (h3 : b = 5 * Real.sqrt 3) 
  (h4 : a ≠ 0) 
  (h5 : b ≠ 0)
  (h6 : c ≠ 0) 
  (h7 : 0 < A ∧ A < 180) 
  (h8 : 0 < B ∧ B < 180) 
  (h9 : 0 < C ∧ C < 180) 
  (h10 : A + B + C = 180) : 
  (A = 90 ∨ A = C) ∧ A + B + C = 180 :=
by 
  sorry

end triangle_type_l305_305561

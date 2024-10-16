import Mathlib

namespace NUMINAMATH_CALUDE_parabola_ratio_l2947_294702

-- Define the parabola R
def Parabola (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = a * p.1^2}

-- Define the vertex and focus of a parabola
def vertex (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry
def focus (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Define the locus of midpoints
def midpointLocus (R : Set (ℝ × ℝ)) (W : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Main theorem
theorem parabola_ratio 
  (R : Set (ℝ × ℝ)) 
  (h_R : ∃ a : ℝ, R = Parabola a) 
  (W₁ : ℝ × ℝ) 
  (G₁ : ℝ × ℝ) 
  (h_W₁ : W₁ = vertex R) 
  (h_G₁ : G₁ = focus R) 
  (S : Set (ℝ × ℝ)) 
  (h_S : S = midpointLocus R W₁) 
  (W₂ : ℝ × ℝ) 
  (G₂ : ℝ × ℝ) 
  (h_W₂ : W₂ = vertex S) 
  (h_G₂ : G₂ = focus S) : 
  ‖G₁ - G₂‖ / ‖W₁ - W₂‖ = 1/4 := by sorry


end NUMINAMATH_CALUDE_parabola_ratio_l2947_294702


namespace NUMINAMATH_CALUDE_daisy_field_count_l2947_294734

theorem daisy_field_count : ∃! n : ℕ,
  (n : ℚ) / 14 + 2 * ((n : ℚ) / 14) + 4 * ((n : ℚ) / 14) + 7000 = n ∧
  n > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_daisy_field_count_l2947_294734


namespace NUMINAMATH_CALUDE_required_moles_of_reactants_l2947_294762

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String

-- Define the molar ratio
def molarRatio : ℚ := 1

-- Define the desired amount of products
def desiredProduct : ℚ := 3

-- Define the chemical equation
def chemicalEquation : Reaction := {
  reactant1 := "AgNO3"
  reactant2 := "NaOH"
  product1 := "AgOH"
  product2 := "NaNO3"
}

-- Theorem statement
theorem required_moles_of_reactants :
  let requiredReactant1 := desiredProduct * molarRatio
  let requiredReactant2 := desiredProduct * molarRatio
  requiredReactant1 = 3 ∧ requiredReactant2 = 3 :=
sorry

end NUMINAMATH_CALUDE_required_moles_of_reactants_l2947_294762


namespace NUMINAMATH_CALUDE_nested_fraction_value_l2947_294742

theorem nested_fraction_value : 
  (1 : ℚ) / (1 + 1 / (1 + 1 / 2)) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_value_l2947_294742


namespace NUMINAMATH_CALUDE_kramer_packing_theorem_l2947_294747

/-- Kramer's packing rate in cases per hour -/
def packing_rate : ℝ := 120

/-- Number of boxes Kramer packs per minute -/
def boxes_per_minute : ℕ := 10

/-- Number of boxes in one case -/
def boxes_per_case : ℕ := 5

/-- Number of cases Kramer packs in 2 hours -/
def cases_in_two_hours : ℕ := 240

/-- The number of cases Kramer can pack in x hours -/
def cases_packed (x : ℝ) : ℝ := packing_rate * x

theorem kramer_packing_theorem (x : ℝ) : 
  cases_packed x = packing_rate * x ∧
  (boxes_per_minute : ℝ) * 60 / boxes_per_case = packing_rate ∧
  cases_in_two_hours = packing_rate * 2 :=
sorry

end NUMINAMATH_CALUDE_kramer_packing_theorem_l2947_294747


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l2947_294777

theorem add_preserves_inequality (a b c : ℝ) : a > b → a + c > b + c := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l2947_294777


namespace NUMINAMATH_CALUDE_be_length_l2947_294703

-- Define the parallelogram and points
structure Parallelogram :=
  (A B C D E F G : ℝ × ℝ)

-- Define the conditions
def is_valid_configuration (p : Parallelogram) : Prop :=
  let ⟨A, B, C, D, E, F, G⟩ := p
  -- F is on the extension of AD
  ∃ t : ℝ, t > 1 ∧ F = A + t • (D - A) ∧
  -- ABCD is a parallelogram
  B - A = C - D ∧ D - A = C - B ∧
  -- BF intersects AC at E
  ∃ s : ℝ, 0 < s ∧ s < 1 ∧ E = A + s • (C - A) ∧
  -- BF intersects DC at G
  ∃ r : ℝ, 0 < r ∧ r < 1 ∧ G = D + r • (C - D) ∧
  -- EF = 15
  ‖F - E‖ = 15 ∧
  -- GF = 20
  ‖F - G‖ = 20

-- The theorem to prove
theorem be_length (p : Parallelogram) (h : is_valid_configuration p) : 
  ‖p.B - p.E‖ = 5 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_be_length_l2947_294703


namespace NUMINAMATH_CALUDE_gina_college_cost_l2947_294712

/-- Calculates the total cost of Gina's college expenses --/
def total_college_cost (
  total_credits : ℕ) 
  (regular_credits : ℕ) 
  (lab_credits : ℕ) 
  (regular_class_cost : ℕ) 
  (lab_class_cost : ℕ) 
  (textbook_count : ℕ) 
  (textbook_cost : ℕ) 
  (online_resource_count : ℕ) 
  (online_resource_cost : ℕ) 
  (facilities_fee : ℕ) 
  (lab_fee : ℕ) : ℕ :=
  regular_credits * regular_class_cost +
  lab_credits * lab_class_cost +
  textbook_count * textbook_cost +
  online_resource_count * online_resource_cost +
  facilities_fee +
  lab_credits * lab_fee

theorem gina_college_cost : 
  total_college_cost 18 12 6 450 550 3 150 4 95 200 75 = 10180 := by
  sorry

end NUMINAMATH_CALUDE_gina_college_cost_l2947_294712


namespace NUMINAMATH_CALUDE_car_cost_calculation_l2947_294739

/-- The cost of a car shared between two people, where one pays $900 for 3/7 of the usage -/
theorem car_cost_calculation (sue_payment : ℝ) (sue_usage : ℚ) (total_cost : ℝ) : 
  sue_payment = 900 → 
  sue_usage = 3/7 → 
  sue_payment / total_cost = sue_usage →
  total_cost = 2100 := by
  sorry

#check car_cost_calculation

end NUMINAMATH_CALUDE_car_cost_calculation_l2947_294739


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2947_294726

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_theorem :
  (U \ (A ∩ B)) = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2947_294726


namespace NUMINAMATH_CALUDE_range_of_t_squared_minus_one_l2947_294766

theorem range_of_t_squared_minus_one :
  ∀ z : ℝ, ∃ x y : ℝ, x ≠ 0 ∧ (y / x)^2 - 1 = z :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_squared_minus_one_l2947_294766


namespace NUMINAMATH_CALUDE_perfect_square_implies_zero_a_l2947_294723

theorem perfect_square_implies_zero_a (a b : ℤ) :
  (∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_implies_zero_a_l2947_294723


namespace NUMINAMATH_CALUDE_prob_two_red_balls_l2947_294774

/-- The probability of selecting 2 red balls from a bag with 5 red, 6 blue, and 4 green balls -/
theorem prob_two_red_balls (red blue green : ℕ) (total : ℕ) (h1 : red = 5) (h2 : blue = 6) (h3 : green = 4) (h4 : total = red + blue + green) :
  (red.choose 2 : ℚ) / (total.choose 2) = 2 / 21 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_balls_l2947_294774


namespace NUMINAMATH_CALUDE_initial_apples_count_l2947_294741

/-- The number of apples initially on the tree -/
def initial_apples : ℕ := sorry

/-- The number of apples picked from the tree -/
def picked_apples : ℕ := 7

/-- The number of new apples that grew on the tree -/
def new_apples : ℕ := 2

/-- The number of apples remaining on the tree -/
def remaining_apples : ℕ := 6

/-- Theorem stating that the initial number of apples is 11 -/
theorem initial_apples_count : initial_apples = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_count_l2947_294741


namespace NUMINAMATH_CALUDE_equation_solutions_l2947_294797

-- Define the equation
def equation (a : ℝ) (t : ℝ) : Prop :=
  (4*a*(Real.sin t)^2 + 4*a*(1 + 2*Real.sqrt 2)*Real.cos t - 4*(a - 1)*Real.sin t - 5*a + 2) / 
  (2*Real.sqrt 2*Real.cos t - Real.sin t) = 4*a

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {t : ℝ | equation a t ∧ 0 < t ∧ t < Real.pi/2}

-- Define the condition for exactly two distinct solutions
def has_two_distinct_solutions (a : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), t₁ ∈ solution_set a ∧ t₂ ∈ solution_set a ∧ t₁ ≠ t₂ ∧
  ∀ (t : ℝ), t ∈ solution_set a → t = t₁ ∨ t = t₂

-- The main theorem
theorem equation_solutions (a : ℝ) :
  has_two_distinct_solutions a ↔ (a > 6 ∧ a < 18 + 24*Real.sqrt 2) ∨ a > 18 + 24*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2947_294797


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2947_294743

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem fifth_term_of_geometric_sequence (a : ℕ → ℝ) (y : ℝ) :
  geometric_sequence a (3 * y) →
  a 0 = 3 →
  a 1 = 9 * y →
  a 2 = 27 * y^2 →
  a 3 = 81 * y^3 →
  a 4 = 243 * y^4 :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2947_294743


namespace NUMINAMATH_CALUDE_willowton_vampires_l2947_294706

def vampire_growth (initial_population : ℕ) (initial_vampires : ℕ) (turns_per_night : ℕ) (nights : ℕ) : ℕ :=
  sorry

theorem willowton_vampires :
  vampire_growth 300 2 5 2 = 72 :=
sorry

end NUMINAMATH_CALUDE_willowton_vampires_l2947_294706


namespace NUMINAMATH_CALUDE_total_paths_is_fifteen_l2947_294729

/-- A graph representing paths between points A, B, C, and D. -/
structure PathGraph where
  paths_AB : Nat
  paths_BC : Nat
  paths_CD : Nat
  direct_AC : Nat

/-- Calculates the total number of paths from A to D in the given graph. -/
def total_paths (g : PathGraph) : Nat :=
  g.paths_AB * g.paths_BC * g.paths_CD + g.direct_AC * g.paths_CD

/-- Theorem stating that the total number of paths from A to D is 15. -/
theorem total_paths_is_fifteen (g : PathGraph) 
  (h1 : g.paths_AB = 2)
  (h2 : g.paths_BC = 2)
  (h3 : g.paths_CD = 3)
  (h4 : g.direct_AC = 1) : 
  total_paths g = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_paths_is_fifteen_l2947_294729


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_3_and_3n_multiple_of_5_l2947_294736

theorem smallest_n_multiple_of_3_and_3n_multiple_of_5 :
  ∃ n : ℕ, n > 0 ∧ 3 ∣ n ∧ 5 ∣ (3 * n) ∧
  ∀ m : ℕ, m > 0 → 3 ∣ m → 5 ∣ (3 * m) → n ≤ m :=
by
  -- The proof goes here
  sorry

#check smallest_n_multiple_of_3_and_3n_multiple_of_5

end NUMINAMATH_CALUDE_smallest_n_multiple_of_3_and_3n_multiple_of_5_l2947_294736


namespace NUMINAMATH_CALUDE_intersection_M_N_l2947_294765

def M : Set ℝ := {x | |x - 2| ≤ 1}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2947_294765


namespace NUMINAMATH_CALUDE_cube_volume_to_surface_area_l2947_294752

theorem cube_volume_to_surface_area :
  ∀ (s : ℝ), s > 0 → s^3 = 729 → 6 * s^2 = 486 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_to_surface_area_l2947_294752


namespace NUMINAMATH_CALUDE_helicopter_rental_hours_per_day_l2947_294753

/-- Given the total cost, hourly rate, and number of days for renting a helicopter,
    calculate the number of hours rented per day. -/
theorem helicopter_rental_hours_per_day 
  (total_cost : ℝ) 
  (hourly_rate : ℝ) 
  (num_days : ℝ) 
  (h1 : total_cost = 450)
  (h2 : hourly_rate = 75)
  (h3 : num_days = 3)
  (h4 : hourly_rate > 0)
  (h5 : num_days > 0) :
  total_cost / (hourly_rate * num_days) = 2 := by
  sorry

#check helicopter_rental_hours_per_day

end NUMINAMATH_CALUDE_helicopter_rental_hours_per_day_l2947_294753


namespace NUMINAMATH_CALUDE_inequality_proof_l2947_294740

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (1/x) + (4/y) + (9/z) ≥ 36 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2947_294740


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2947_294732

theorem polynomial_division_theorem (z : ℂ) : 
  ∃ (r : ℂ), 4*z^5 - 3*z^4 + 2*z^3 - 5*z^2 + 9*z - 4 = 
  (z + 3) * (4*z^4 - 15*z^3 + 47*z^2 - 146*z + 447) + r ∧ 
  Complex.abs r < Complex.abs (z + 3) := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2947_294732


namespace NUMINAMATH_CALUDE_boarding_students_count_total_boarding_students_l2947_294745

theorem boarding_students_count (total_students : ℕ) (male_students : ℕ) 
  (female_youth_league : ℕ) (female_boarding : ℕ) (non_boarding_youth_league : ℕ) 
  (male_boarding_youth_league : ℕ) (male_non_youth_league_non_boarding : ℕ) 
  (female_non_youth_league_non_boarding : ℕ) : ℕ :=
  sorry

/-- Given the following conditions:
    - There are 50 students in total
    - There are 33 male students
    - There are 7 female members of the Youth League
    - There are 9 female boarding students
    - There are 15 non-boarding members of the Youth League
    - There are 6 male boarding members of the Youth League
    - There are 8 male students who are non-members of the Youth League and non-boarding
    - There are 3 female students who are non-members of the Youth League and non-boarding
    The total number of boarding students is 28. -/
theorem total_boarding_students :
  boarding_students_count 50 33 7 9 15 6 8 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_boarding_students_count_total_boarding_students_l2947_294745


namespace NUMINAMATH_CALUDE_sin_15_cos_15_l2947_294713

theorem sin_15_cos_15 : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_l2947_294713


namespace NUMINAMATH_CALUDE_prime_difference_theorem_l2947_294755

theorem prime_difference_theorem (x y : ℝ) 
  (h1 : Prime (⌊x - y⌋ : ℤ))
  (h2 : Prime (⌊x^2 - y^2⌋ : ℤ))
  (h3 : Prime (⌊x^3 - y^3⌋ : ℤ)) :
  x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_difference_theorem_l2947_294755


namespace NUMINAMATH_CALUDE_oranges_harvested_per_day_l2947_294750

theorem oranges_harvested_per_day :
  let total_sacks : ℕ := 56
  let total_days : ℕ := 14
  let sacks_per_day : ℕ := total_sacks / total_days
  sacks_per_day = 4 := by sorry

end NUMINAMATH_CALUDE_oranges_harvested_per_day_l2947_294750


namespace NUMINAMATH_CALUDE_tv_price_change_l2947_294793

theorem tv_price_change (P : ℝ) : 
  P > 0 → (P * 0.9) * 1.3 = P * 1.17 := by
sorry

end NUMINAMATH_CALUDE_tv_price_change_l2947_294793


namespace NUMINAMATH_CALUDE_emperor_strategy_exists_l2947_294704

/-- Represents the nature of a wizard -/
inductive WizardNature
| Good
| Evil

/-- Represents a wizard -/
structure Wizard where
  nature : WizardNature

/-- Represents the Emperor's knowledge about a wizard -/
inductive WizardKnowledge
| Unknown
| KnownGood
| KnownEvil

/-- Represents the state of the festival -/
structure FestivalState where
  wizards : Finset Wizard
  knowledge : Wizard → WizardKnowledge

/-- Represents a strategy for the Emperor -/
structure EmperorStrategy where
  askQuestion : FestivalState → Wizard → Prop
  expelWizard : FestivalState → Option Wizard

/-- The main theorem -/
theorem emperor_strategy_exists :
  ∃ (strategy : EmperorStrategy),
    ∀ (initial_state : FestivalState),
      initial_state.wizards.card = 2015 →
      ∃ (final_state : FestivalState),
        (∀ w ∈ final_state.wizards, w.nature = WizardNature.Good) ∧
        (∃! w, w ∉ final_state.wizards ∧ w.nature = WizardNature.Good) :=
by sorry

end NUMINAMATH_CALUDE_emperor_strategy_exists_l2947_294704


namespace NUMINAMATH_CALUDE_total_trout_caught_l2947_294749

/-- The number of trout caught by Sara, Melanie, and John -/
def total_trout (sara melanie john : ℕ) : ℕ := sara + melanie + john

/-- Theorem stating the total number of trout caught -/
theorem total_trout_caught :
  ∃ (sara melanie john : ℕ),
    sara = 5 ∧
    melanie = 2 * sara ∧
    john = 3 * melanie ∧
    total_trout sara melanie john = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_trout_caught_l2947_294749


namespace NUMINAMATH_CALUDE_equation_solution_l2947_294718

theorem equation_solution : 
  let x : ℝ := 14.8 / 0.13
  0.05 * x + 0.04 * (30 + 2 * x) = 16 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2947_294718


namespace NUMINAMATH_CALUDE_total_troll_count_l2947_294707

/-- The number of trolls Erin counted in different locations -/
structure TrollCount where
  forest : ℕ
  bridge : ℕ
  plains : ℕ

/-- The conditions given in the problem -/
def troll_conditions (t : TrollCount) : Prop :=
  t.forest = 6 ∧
  t.bridge = 4 * t.forest - 6 ∧
  t.plains = t.bridge / 2

/-- The theorem stating the total number of trolls Erin counted -/
theorem total_troll_count (t : TrollCount) (h : troll_conditions t) : 
  t.forest + t.bridge + t.plains = 33 := by
  sorry


end NUMINAMATH_CALUDE_total_troll_count_l2947_294707


namespace NUMINAMATH_CALUDE_complement_union_problem_l2947_294773

def U : Finset Nat := {1,2,3,4,5,6,7,8}

theorem complement_union_problem (A B : Finset Nat) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : A ∩ B = {3})
  (h4 : (U \ B) ∩ A = {1,2})
  (h5 : (U \ A) ∩ B = {4,5}) :
  U \ (A ∪ B) = {6,7,8} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_problem_l2947_294773


namespace NUMINAMATH_CALUDE_octal_subtraction_l2947_294780

/-- Converts a base-8 number represented as a list of digits to a natural number -/
def fromOctal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits -/
def toOctal (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: toOctal (n / 8)

/-- The main theorem stating that 5273₈ - 3614₈ = 1457₈ -/
theorem octal_subtraction :
  fromOctal [3, 7, 2, 5] - fromOctal [4, 1, 6, 3] = fromOctal [7, 5, 4, 1] := by
  sorry

#eval toOctal (fromOctal [3, 7, 2, 5] - fromOctal [4, 1, 6, 3])

end NUMINAMATH_CALUDE_octal_subtraction_l2947_294780


namespace NUMINAMATH_CALUDE_pell_equation_solution_form_l2947_294720

/-- Pell's equation solution type -/
structure PellSolution (d : ℕ) :=
  (x : ℕ)
  (y : ℕ)
  (eq : x^2 - d * y^2 = 1)

/-- Fundamental solution to Pell's equation -/
def fundamental_solution (d : ℕ) : PellSolution d := sorry

/-- Any solution to Pell's equation -/
def any_solution (d : ℕ) : PellSolution d := sorry

/-- Square-free natural number -/
def is_square_free (d : ℕ) : Prop := sorry

theorem pell_equation_solution_form 
  (d : ℕ) 
  (h_square_free : is_square_free d) 
  (x₁ y₁ : ℕ) 
  (h_fund : fundamental_solution d = ⟨x₁, y₁, sorry⟩) 
  (xₙ yₙ : ℕ) 
  (h_any : any_solution d = ⟨xₙ, yₙ, sorry⟩) :
  ∃ (n : ℕ), (xₙ : ℝ) + yₙ * Real.sqrt d = ((x₁ : ℝ) + y₁ * Real.sqrt d) ^ n :=
sorry

end NUMINAMATH_CALUDE_pell_equation_solution_form_l2947_294720


namespace NUMINAMATH_CALUDE_amount_saved_calculation_l2947_294710

def initial_amount : ℕ := 6000
def pen_cost : ℕ := 3200
def eraser_cost : ℕ := 1000
def candy_cost : ℕ := 500

theorem amount_saved_calculation :
  initial_amount - (pen_cost + eraser_cost + candy_cost) = 1300 := by
  sorry

end NUMINAMATH_CALUDE_amount_saved_calculation_l2947_294710


namespace NUMINAMATH_CALUDE_box_volume_increase_l2947_294776

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + w * h + l * h) = 1800)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l2947_294776


namespace NUMINAMATH_CALUDE_lawn_mowing_solution_l2947_294783

/-- Represents the lawn mowing problem -/
def LawnMowingProblem (lawn_length lawn_width swath_width overlap mowing_speed : ℝ) : Prop :=
  let effective_width := (swath_width - overlap) / 12  -- Convert to feet
  let strips := lawn_width / effective_width
  let total_distance := strips * lawn_length
  let mowing_time := total_distance / mowing_speed
  0.94 < mowing_time ∧ mowing_time < 0.96

/-- Theorem stating the solution to the lawn mowing problem -/
theorem lawn_mowing_solution :
  LawnMowingProblem 72 120 (30/12) (6/12) 4500 :=
sorry

end NUMINAMATH_CALUDE_lawn_mowing_solution_l2947_294783


namespace NUMINAMATH_CALUDE_square_difference_plus_constant_l2947_294715

theorem square_difference_plus_constant : (262^2 - 258^2) + 150 = 2230 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_plus_constant_l2947_294715


namespace NUMINAMATH_CALUDE_complex_power_problem_l2947_294700

theorem complex_power_problem (z : ℂ) (i : ℂ) (h : i^2 = -1) (eq : (1 + z) / (1 - z) = i) : z^2019 = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_problem_l2947_294700


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2947_294771

theorem consecutive_integers_sum (n : ℤ) : 
  (∀ k : ℤ, n - 4 ≤ k ∧ k ≤ n + 4 → k > 0) →
  (n - 4) + (n - 3) + (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 99 →
  n + 4 = 15 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2947_294771


namespace NUMINAMATH_CALUDE_loan_time_period_l2947_294717

/-- Calculates the time period of a loan using simple interest -/
theorem loan_time_period (principal : ℝ) (interest : ℝ) (rate : ℝ) : 
  principal = 900 → 
  interest = 729 → 
  rate = 9 → 
  (principal * rate * 9) / 100 = interest :=
by
  sorry

end NUMINAMATH_CALUDE_loan_time_period_l2947_294717


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l2947_294738

theorem cylinder_volume_equality (x : ℚ) : x > 0 →
  (5 + x)^2 * 4 = 25 * (4 + x) → x = 35/4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l2947_294738


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2947_294794

open Real

/-- Cyclic sum of a function over three variables -/
def cyclicSum (f : ℝ → ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f a b c + f b c a + f c a b

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hsum : a + b + c = 3) :
    cyclicSum (fun x y z => 1 / (2 * x^2 + y^2 + z^2)) a b c ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2947_294794


namespace NUMINAMATH_CALUDE_pencil_price_l2947_294721

theorem pencil_price (num_pens : ℕ) (num_pencils : ℕ) (total_cost : ℕ) (pen_price : ℕ) :
  num_pens = 30 →
  num_pencils = 75 →
  total_cost = 630 →
  pen_price = 16 →
  (total_cost - num_pens * pen_price) / num_pencils = 2 :=
by sorry

end NUMINAMATH_CALUDE_pencil_price_l2947_294721


namespace NUMINAMATH_CALUDE_bottle_cost_difference_l2947_294763

/-- Represents a bottle of capsules -/
structure Bottle where
  capsules : ℕ
  cost : ℚ

/-- Calculates the cost per capsule for a given bottle -/
def costPerCapsule (b : Bottle) : ℚ := b.cost / b.capsules

/-- The difference in cost per capsule between two bottles -/
def costDifference (b1 b2 : Bottle) : ℚ := costPerCapsule b2 - costPerCapsule b1

theorem bottle_cost_difference :
  let bottleR : Bottle := { capsules := 250, cost := 25/4 }
  let bottleT : Bottle := { capsules := 100, cost := 3 }
  costDifference bottleR bottleT = 1/200
  := by sorry

end NUMINAMATH_CALUDE_bottle_cost_difference_l2947_294763


namespace NUMINAMATH_CALUDE_justin_jerseys_l2947_294782

theorem justin_jerseys (long_sleeve_cost : ℕ) (striped_cost : ℕ) (long_sleeve_count : ℕ) (total_spent : ℕ) :
  long_sleeve_cost = 15 →
  striped_cost = 10 →
  long_sleeve_count = 4 →
  total_spent = 80 →
  (total_spent - long_sleeve_cost * long_sleeve_count) / striped_cost = 2 :=
by sorry

end NUMINAMATH_CALUDE_justin_jerseys_l2947_294782


namespace NUMINAMATH_CALUDE_probability_of_decagon_side_l2947_294756

/-- A regular decagon -/
def RegularDecagon : Type := Unit

/-- A triangle formed by three vertices of a regular decagon -/
def DecagonTriangle : Type := Fin 3 → Fin 10

/-- Predicate to check if a DecagonTriangle has at least one side that is also a side of the decagon -/
def HasDecagonSide (t : DecagonTriangle) : Prop := sorry

/-- The set of all possible DecagonTriangles -/
def AllDecagonTriangles : Finset DecagonTriangle := sorry

/-- The set of DecagonTriangles that have at least one side that is also a side of the decagon -/
def TrianglesWithDecagonSide : Finset DecagonTriangle := sorry

/-- The probability of selecting a DecagonTriangle that has at least one side that is also a side of the decagon -/
def ProbabilityOfDecagonSide : ℚ := Finset.card TrianglesWithDecagonSide / Finset.card AllDecagonTriangles

theorem probability_of_decagon_side :
  ProbabilityOfDecagonSide = 7 / 12 :=
sorry

end NUMINAMATH_CALUDE_probability_of_decagon_side_l2947_294756


namespace NUMINAMATH_CALUDE_solutions_count_2x_3y_763_l2947_294746

theorem solutions_count_2x_3y_763 : 
  (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 + 3 * p.2 = 763 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 764) (Finset.range 764))).card = 127 := by
  sorry

end NUMINAMATH_CALUDE_solutions_count_2x_3y_763_l2947_294746


namespace NUMINAMATH_CALUDE_train_crossing_time_l2947_294767

/-- The time taken for a train to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 700 ∧ train_speed_kmh = 125.99999999999999 →
  (train_length / (train_speed_kmh * (1000 / 3600))) = 20 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l2947_294767


namespace NUMINAMATH_CALUDE_cara_age_difference_l2947_294799

/-- The age difference between Cara and her mom -/
def age_difference (grandmother_age mom_age_difference cara_age : ℕ) : ℕ :=
  grandmother_age - mom_age_difference - cara_age

/-- Proof that Cara is 20 years younger than her mom -/
theorem cara_age_difference :
  age_difference 75 15 40 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cara_age_difference_l2947_294799


namespace NUMINAMATH_CALUDE_point_B_in_third_quadrant_l2947_294789

/-- Given that point A (-x, y-1) is in the fourth quadrant, 
    prove that point B (y-1, x) is in the third quadrant. -/
theorem point_B_in_third_quadrant 
  (x y : ℝ) 
  (h_fourth : -x > 0 ∧ y - 1 < 0) : 
  y - 1 < 0 ∧ x < 0 := by
sorry

end NUMINAMATH_CALUDE_point_B_in_third_quadrant_l2947_294789


namespace NUMINAMATH_CALUDE_knowledge_group_theorem_l2947_294775

/-- A group of people where some know each other -/
structure KnowledgeGroup (k : ℕ) where
  knows : Fin k → Fin k → Prop
  symm : ∀ i j, knows i j ↔ knows j i

/-- For any n people, there's an (n+1)-th person who knows them all -/
def HasKnowledgeable (n : ℕ) (g : KnowledgeGroup k) : Prop :=
  ∀ (s : Finset (Fin k)), s.card = n → 
    ∃ i, i ∉ s ∧ ∀ j ∈ s, g.knows i j

theorem knowledge_group_theorem (n : ℕ) :
  (∃ (g : KnowledgeGroup (2*n + 1)), HasKnowledgeable n g → 
    ∃ i, ∀ j, g.knows i j) ∧
  (∃ (g : KnowledgeGroup (2*n + 2)), HasKnowledgeable n g ∧ 
    ∀ i, ∃ j, ¬g.knows i j) := by
  sorry

end NUMINAMATH_CALUDE_knowledge_group_theorem_l2947_294775


namespace NUMINAMATH_CALUDE_coin_difference_is_six_l2947_294760

/-- Represents the available coin denominations in cents -/
def CoinDenominations : List ℕ := [5, 10, 25]

/-- The amount Paul needs to pay in cents -/
def AmountToPay : ℕ := 45

/-- Calculates the minimum number of coins needed to make the payment -/
def MinCoins (amount : ℕ) (denominations : List ℕ) : ℕ :=
  sorry

/-- Calculates the maximum number of coins needed to make the payment -/
def MaxCoins (amount : ℕ) (denominations : List ℕ) : ℕ :=
  sorry

/-- Theorem stating the difference between max and min coins is 6 -/
theorem coin_difference_is_six :
  MaxCoins AmountToPay CoinDenominations - MinCoins AmountToPay CoinDenominations = 6 :=
sorry

end NUMINAMATH_CALUDE_coin_difference_is_six_l2947_294760


namespace NUMINAMATH_CALUDE_soccer_team_red_cards_l2947_294735

theorem soccer_team_red_cards 
  (total_players : ℕ) 
  (players_without_cautions : ℕ) 
  (yellow_cards_per_cautioned_player : ℕ) 
  (yellow_cards_per_red_card : ℕ) 
  (h1 : total_players = 11) 
  (h2 : players_without_cautions = 5) 
  (h3 : yellow_cards_per_cautioned_player = 1) 
  (h4 : yellow_cards_per_red_card = 2) : 
  (total_players - players_without_cautions) * yellow_cards_per_cautioned_player / yellow_cards_per_red_card = 3 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_red_cards_l2947_294735


namespace NUMINAMATH_CALUDE_percentage_of_men_in_class_l2947_294758

theorem percentage_of_men_in_class (
  women_science_major_percentage : Real)
  (non_science_major_percentage : Real)
  (men_science_major_percentage : Real)
  (h1 : women_science_major_percentage = 0.1)
  (h2 : non_science_major_percentage = 0.6)
  (h3 : men_science_major_percentage = 0.8500000000000001)
  : Real :=
by
  sorry

#check percentage_of_men_in_class

end NUMINAMATH_CALUDE_percentage_of_men_in_class_l2947_294758


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2947_294798

/-- Given complex numbers a and b, prove that 2a - 3bi equals 22 - 12i -/
theorem complex_expression_equality (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + 4*I) :
  2*a - 3*b*I = 22 - 12*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2947_294798


namespace NUMINAMATH_CALUDE_suitcase_lock_settings_l2947_294796

/-- The number of dials on the suitcase lock. -/
def num_dials : ℕ := 4

/-- The number of digits available for each dial. -/
def num_digits : ℕ := 10

/-- Calculates the number of different settings for a suitcase lock. -/
def lock_settings : ℕ := num_digits * (num_digits - 1) * (num_digits - 2) * (num_digits - 3)

/-- Theorem stating that the number of different settings for the suitcase lock is 5040. -/
theorem suitcase_lock_settings :
  lock_settings = 5040 := by sorry

end NUMINAMATH_CALUDE_suitcase_lock_settings_l2947_294796


namespace NUMINAMATH_CALUDE_solve_equation_l2947_294751

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 5 / 3 → x = -27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2947_294751


namespace NUMINAMATH_CALUDE_middle_term_coefficient_l2947_294768

/-- Given a natural number n, returns the binomial expansion of (1-2x)^n -/
def binomialExpansion (n : ℕ) : List ℤ := sorry

/-- Returns the sum of coefficients of even-numbered terms in a list -/
def sumEvenTerms (coeffs : List ℤ) : ℤ := sorry

/-- Returns the middle coefficient of a list -/
def middleCoefficient (coeffs : List ℤ) : ℤ := sorry

theorem middle_term_coefficient (n : ℕ) :
  sumEvenTerms (binomialExpansion n) = 128 →
  middleCoefficient (binomialExpansion n) = 1120 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_coefficient_l2947_294768


namespace NUMINAMATH_CALUDE_custom_op_example_l2947_294787

/-- Custom operation $\$$ defined for two integers -/
def custom_op (a b : Int) : Int := a * (b - 1) + a * b

/-- Theorem stating that 5 $\$$ (-3) = -35 -/
theorem custom_op_example : custom_op 5 (-3) = -35 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l2947_294787


namespace NUMINAMATH_CALUDE_random_co_captains_probability_l2947_294725

def team_sizes : List Nat := [4, 5, 6, 7]
def co_captains_per_team : Nat := 3

def prob_both_co_captains (n : Nat) : Rat :=
  (co_captains_per_team.choose 2) / (n.choose 2)

theorem random_co_captains_probability :
  (1 / team_sizes.length : Rat) *
  (team_sizes.map prob_both_co_captains).sum = 2/7 := by sorry

end NUMINAMATH_CALUDE_random_co_captains_probability_l2947_294725


namespace NUMINAMATH_CALUDE_shifts_needed_is_six_l2947_294719

/-- Represents the problem of assigning workers to shifts -/
structure ShiftAssignment where
  total_workers : ℕ
  workers_per_shift : ℕ
  total_assignments : ℕ

/-- Calculates the number of shifts needed -/
def number_of_shifts (assignment : ShiftAssignment) : ℕ :=
  assignment.total_workers / assignment.workers_per_shift

/-- Theorem stating that the number of shifts is 6 for the given conditions -/
theorem shifts_needed_is_six (assignment : ShiftAssignment) 
  (h1 : assignment.total_workers = 12)
  (h2 : assignment.workers_per_shift = 2)
  (h3 : assignment.total_assignments = 23760) :
  number_of_shifts assignment = 6 := by
  sorry

#eval number_of_shifts ⟨12, 2, 23760⟩

end NUMINAMATH_CALUDE_shifts_needed_is_six_l2947_294719


namespace NUMINAMATH_CALUDE_daily_sales_extrema_l2947_294786

-- Define the sales volume function
def g (t : ℝ) : ℝ := 80 - 2 * t

-- Define the price function
def f (t : ℝ) : ℝ := 20 - abs (t - 10)

-- Define the daily sales function
def y (t : ℝ) : ℝ := g t * f t

-- Theorem statement
theorem daily_sales_extrema :
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 20 → y t ≤ 1200) ∧
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 20 ∧ y t = 1200) ∧
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 20 → y t ≥ 400) ∧
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 20 ∧ y t = 400) :=
by sorry

end NUMINAMATH_CALUDE_daily_sales_extrema_l2947_294786


namespace NUMINAMATH_CALUDE_total_spent_l2947_294733

def weekend_expenses (adidas nike skechers clothes : ℕ) : Prop :=
  nike = 3 * adidas ∧
  adidas = skechers / 5 ∧
  adidas = 600 ∧
  clothes = 2600

theorem total_spent (adidas nike skechers clothes : ℕ) :
  weekend_expenses adidas nike skechers clothes →
  adidas + nike + skechers + clothes = 8000 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_l2947_294733


namespace NUMINAMATH_CALUDE_three_balls_four_boxes_l2947_294737

theorem three_balls_four_boxes :
  (∀ n : ℕ, n ≤ 3 → n > 0 → 4 ^ n = (Fintype.card (Fin 4)) ^ n) →
  4 ^ 3 = 64 :=
by sorry

end NUMINAMATH_CALUDE_three_balls_four_boxes_l2947_294737


namespace NUMINAMATH_CALUDE_star_equation_solution_l2947_294722

/-- The star operation defined as a * b = ab + 2b - a -/
def star (a b : ℝ) : ℝ := a * b + 2 * b - a

/-- Theorem stating that if 3 * x = 27 under the star operation, then x = 6 -/
theorem star_equation_solution :
  ∀ x : ℝ, star 3 x = 27 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l2947_294722


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2947_294708

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊2 * x⌋ - (1/2 : ℝ)⌋ = ⌊x + 3⌋ ↔ 3.5 ≤ x ∧ x < 4.5 := by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2947_294708


namespace NUMINAMATH_CALUDE_basketball_game_difference_l2947_294727

theorem basketball_game_difference (total_games won_games lost_games : ℕ) : 
  total_games = 62 →
  won_games > lost_games →
  won_games = 45 →
  lost_games = 17 →
  won_games - lost_games = 28 :=
by sorry

end NUMINAMATH_CALUDE_basketball_game_difference_l2947_294727


namespace NUMINAMATH_CALUDE_impossible_to_use_all_parts_l2947_294770

theorem impossible_to_use_all_parts (p q r : ℕ) : 
  ¬∃ (x y z : ℕ), (2 * x + 2 * z = 2 * p + 2 * r + 2) ∧ 
                   (2 * x + y = 2 * p + q + 1) ∧ 
                   (y + z = q + r) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_use_all_parts_l2947_294770


namespace NUMINAMATH_CALUDE_sequence_relationship_l2947_294744

def y (x : ℕ) : ℕ := x^2 + x + 1

theorem sequence_relationship (x : ℕ) :
  x > 0 →
  (y (x + 1) - y x = 2 * x + 2) ∧
  (y (x + 2) - y (x + 1) = 2 * x + 4) ∧
  (y (x + 3) - y (x + 2) = 2 * x + 6) ∧
  (y (x + 4) - y (x + 3) = 2 * x + 8) :=
by sorry

end NUMINAMATH_CALUDE_sequence_relationship_l2947_294744


namespace NUMINAMATH_CALUDE_train_length_l2947_294748

/-- The length of a train given specific passing times -/
theorem train_length : ∃ (L : ℝ), 
  (L / 24 = (L + 650) / 89) ∧ L = 240 := by sorry

end NUMINAMATH_CALUDE_train_length_l2947_294748


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2947_294788

/-- Given a line segment with one endpoint at (-3, -15) and midpoint at (2, -5),
    the sum of coordinates of the other endpoint is 12 -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (2 : ℝ) = (-3 + x) / 2 → 
    (-5 : ℝ) = (-15 + y) / 2 → 
    x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2947_294788


namespace NUMINAMATH_CALUDE_ratio_K_L_l2947_294795

theorem ratio_K_L : ∃ (K L : ℤ),
  (∀ (x : ℝ), x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 →
    (K / (x + 3 : ℝ)) + (L / (x^2 - 3*x : ℝ)) = ((x^2 - x + 5) / (x^3 + x^2 - 9*x) : ℝ)) →
  (K : ℚ) / (L : ℚ) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_K_L_l2947_294795


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l2947_294728

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l2947_294728


namespace NUMINAMATH_CALUDE_hours_in_year_correct_hours_in_year_l2947_294730

theorem hours_in_year : ℕ → ℕ → ℕ → Prop :=
  fun hours_per_day days_per_year hours_per_year =>
    hours_per_day = 24 ∧ days_per_year = 365 →
    hours_per_year = hours_per_day * days_per_year

theorem correct_hours_in_year : hours_in_year 24 365 8760 := by
  sorry

end NUMINAMATH_CALUDE_hours_in_year_correct_hours_in_year_l2947_294730


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2947_294761

def point : ℝ × ℝ := (8, -3)

theorem point_in_fourth_quadrant :
  let (x, y) := point
  x > 0 ∧ y < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2947_294761


namespace NUMINAMATH_CALUDE_max_area_circular_sector_l2947_294785

/-- Given a circular sector with perimeter 2p, prove that the maximum area is p^2/4 -/
theorem max_area_circular_sector (p : ℝ) (hp : p > 0) :
  ∃ (R : ℝ), R > 0 ∧
    (∀ (S : ℝ → ℝ), (∀ r, r > 0 → S r = r * (p - r)) →
      (∀ r, r > 0 → S r ≤ S R) ∧ S R = p^2 / 4) :=
sorry

end NUMINAMATH_CALUDE_max_area_circular_sector_l2947_294785


namespace NUMINAMATH_CALUDE_soda_discount_theorem_l2947_294754

/-- Calculates the discounted price for purchasing soda cans -/
def discounted_price (regular_price : ℚ) (num_cans : ℕ) : ℚ :=
  let cases := (num_cans + 23) / 24  -- Round up to nearest case
  let total_regular_price := regular_price * num_cans
  let discount_rate := 
    if cases ≤ 2 then 25/100
    else if cases ≤ 4 then 30/100
    else 35/100
  total_regular_price * (1 - discount_rate)

/-- Theorem stating the discounted price for 70 cans of soda -/
theorem soda_discount_theorem :
  discounted_price (55/100) 70 = 2772/100 := by
  sorry

end NUMINAMATH_CALUDE_soda_discount_theorem_l2947_294754


namespace NUMINAMATH_CALUDE_no_square_prime_ratio_in_triangular_sequence_l2947_294791

theorem no_square_prime_ratio_in_triangular_sequence (p : ℕ) (hp : Prime p) :
  ∀ (x y l : ℕ), l ≥ 1 →
    (x * (x + 1)) / (y * (y + 1)) ≠ p^(2 * l) := by
  sorry

end NUMINAMATH_CALUDE_no_square_prime_ratio_in_triangular_sequence_l2947_294791


namespace NUMINAMATH_CALUDE_dogs_in_park_l2947_294714

theorem dogs_in_park (total_legs : ℕ) (legs_per_dog : ℕ) (h1 : total_legs = 436) (h2 : legs_per_dog = 4) :
  total_legs / legs_per_dog = 109 := by
  sorry

end NUMINAMATH_CALUDE_dogs_in_park_l2947_294714


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l2947_294711

noncomputable def g (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3 + x - 5

theorem g_behavior_at_infinity :
  (∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, x > N → g x > ε) ∧
  (∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > ε) :=
sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l2947_294711


namespace NUMINAMATH_CALUDE_sqrt_nine_minus_sqrt_four_l2947_294701

theorem sqrt_nine_minus_sqrt_four : Real.sqrt 9 - Real.sqrt 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_minus_sqrt_four_l2947_294701


namespace NUMINAMATH_CALUDE_quadrilateral_exists_l2947_294790

/-- A quadrilateral with side lengths and a diagonal -/
structure Quadrilateral :=
  (AB BC CD DA AC : ℝ)

/-- The triangle inequality theorem -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b

/-- Theorem: There exists a quadrilateral ABCD with diagonal AC, where AB = 10, BC = 9, CD = 19, DA = 5, and AC = 15 -/
theorem quadrilateral_exists : ∃ (q : Quadrilateral), 
  q.AB = 10 ∧ 
  q.BC = 9 ∧ 
  q.CD = 19 ∧ 
  q.DA = 5 ∧ 
  q.AC = 15 ∧
  triangle_inequality q.AB q.BC q.AC ∧
  triangle_inequality q.AC q.CD q.DA :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_exists_l2947_294790


namespace NUMINAMATH_CALUDE_emma_cookies_problem_l2947_294759

theorem emma_cookies_problem :
  ∃! (N : ℕ), N < 150 ∧ N % 13 = 7 ∧ N % 8 = 5 ∧ N = 85 := by
  sorry

end NUMINAMATH_CALUDE_emma_cookies_problem_l2947_294759


namespace NUMINAMATH_CALUDE_rational_solution_system_l2947_294772

theorem rational_solution_system (x y z t w : ℚ) :
  t^2 - w^2 + z^2 = 2*x*y ∧
  t^2 - y^2 + w^2 = 2*x*z ∧
  t^2 - w^2 + x^2 = 2*y*z →
  x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

#check rational_solution_system

end NUMINAMATH_CALUDE_rational_solution_system_l2947_294772


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l2947_294764

theorem circle_area_from_circumference (c : ℝ) (h : c = 36) :
  let r := c / (2 * Real.pi)
  (Real.pi * r^2) = 324 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l2947_294764


namespace NUMINAMATH_CALUDE_frictional_force_is_10N_l2947_294709

/-- The acceleration due to gravity (m/s²) -/
def g : ℝ := 9.8

/-- Mass of the tank (kg) -/
def m₁ : ℝ := 2

/-- Mass of the cart (kg) -/
def m₂ : ℝ := 10

/-- Acceleration of the cart (m/s²) -/
def a : ℝ := 5

/-- Coefficient of friction between the tank and cart -/
def μ : ℝ := 0.6

/-- The frictional force acting on the tank from the cart (N) -/
def frictional_force : ℝ := m₁ * a

theorem frictional_force_is_10N : frictional_force = 10 := by
  sorry

#check frictional_force_is_10N

end NUMINAMATH_CALUDE_frictional_force_is_10N_l2947_294709


namespace NUMINAMATH_CALUDE_function_inequality_solution_l2947_294781

theorem function_inequality_solution (f g h : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, (x - y) * (f x) + h x - x * y + y^2 ≤ h y)
  (h2 : ∀ x y : ℝ, h y ≤ (x - y) * (g x) + h x - x * y + y^2) :
  ∃ a b : ℝ, 
    (∀ x : ℝ, f x = -x + a) ∧ 
    (∀ x : ℝ, g x = -x + a) ∧ 
    (∀ x : ℝ, h x = x^2 - a*x + b) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_solution_l2947_294781


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_l2947_294731

theorem square_area_from_rectangle (s r l b : ℝ) : 
  r = s →                  -- radius of circle equals side of square
  l = (2 / 5) * r →        -- length of rectangle is two-fifths of radius
  b = 10 →                 -- breadth of rectangle is 10 units
  l * b = 120 →            -- area of rectangle is 120 sq. units
  s^2 = 900 :=             -- area of square is 900 sq. units
by sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_l2947_294731


namespace NUMINAMATH_CALUDE_athlete_distance_l2947_294779

/-- Proves that an athlete running at 28.8 km/h for 25 seconds covers a distance of 200 meters. -/
theorem athlete_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 28.8 → time = 25 → distance = speed * time * 1000 / 3600 → distance = 200 := by
  sorry

end NUMINAMATH_CALUDE_athlete_distance_l2947_294779


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2947_294778

theorem polynomial_factorization (x : ℝ) : 
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2947_294778


namespace NUMINAMATH_CALUDE_expression_evaluation_l2947_294769

theorem expression_evaluation :
  let a : ℤ := (-2)^2
  5 * a^2 - (a^2 - (2*a - 5*a^2) - 2*(a^2 - 3*a)) = 32 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2947_294769


namespace NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_factor_l2947_294757

/-- A number is a police emergency number if it ends with 133 in decimal system -/
def is_police_emergency_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 1000 * k + 133

/-- Every police emergency number has a prime factor greater than 7 -/
theorem police_emergency_number_has_large_prime_factor (n : ℕ) 
  (h : is_police_emergency_number n) : 
  ∃ p : ℕ, p > 7 ∧ Nat.Prime p ∧ p ∣ n := by
  sorry

end NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_factor_l2947_294757


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l2947_294716

/-- Two lines with the same non-zero y-intercept -/
structure TwoLines where
  y_intercept : ℝ
  slope1 : ℝ
  slope2 : ℝ
  x_intercept1 : ℝ
  x_intercept2 : ℝ
  y_intercept_nonzero : y_intercept ≠ 0
  slope1_is_8 : slope1 = 8
  slope2_is_4 : slope2 = 4

/-- The ratio of x-intercepts is 1/2 -/
theorem x_intercept_ratio (l : TwoLines) : l.x_intercept1 / l.x_intercept2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l2947_294716


namespace NUMINAMATH_CALUDE_calculation_proof_inequalities_solution_l2947_294724

-- Problem 1
theorem calculation_proof :
  Real.pi ^ 0 + |3 - Real.sqrt 2| - (1/3)⁻¹ = 1 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem inequalities_solution (x : ℝ) :
  (2*x > x - 2 ∧ x + 1 < 2) ↔ (-2 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_inequalities_solution_l2947_294724


namespace NUMINAMATH_CALUDE_factorization_problems_l2947_294792

theorem factorization_problems :
  (∀ x y : ℝ, 4 * x^2 - 9 * y^2 = (2*x + 3*y) * (2*x - 3*y)) ∧
  (∀ a b : ℝ, -16 * a^2 + 25 * b^2 = (5*b + 4*a) * (5*b - 4*a)) ∧
  (∀ x y : ℝ, x^3 * y - x * y^3 = x * y * (x + y) * (x - y)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l2947_294792


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l2947_294705

/-- Given a man's speed with and against a stream, calculate his rowing speed in still water. -/
theorem mans_rowing_speed (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 26)
  (h2 : speed_against_stream = 4) :
  (speed_with_stream + speed_against_stream) / 2 = 15 := by
  sorry

#check mans_rowing_speed

end NUMINAMATH_CALUDE_mans_rowing_speed_l2947_294705


namespace NUMINAMATH_CALUDE_bart_earnings_l2947_294784

/-- The amount of money Bart earns per question answered -/
def money_per_question : ℚ := 0.2

/-- The number of questions in each survey -/
def questions_per_survey : ℕ := 10

/-- The number of surveys Bart completed on Monday -/
def surveys_monday : ℕ := 3

/-- The number of surveys Bart completed on Tuesday -/
def surveys_tuesday : ℕ := 4

/-- Theorem stating the total money Bart earned over two days -/
theorem bart_earnings : 
  (surveys_monday + surveys_tuesday) * questions_per_survey * money_per_question = 14 := by
  sorry

end NUMINAMATH_CALUDE_bart_earnings_l2947_294784

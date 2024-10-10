import Mathlib

namespace right_triangle_inequality_l647_64731

theorem right_triangle_inequality (a b c h : ℝ) (n : ℕ) (h1 : 0 < n) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) (h5 : 0 < h) 
  (h6 : a^2 + b^2 = c^2) (h7 : a * b = c * h) (h8 : a + b < c + h) :
  a^n + b^n < c^n + h^n := by
sorry

end right_triangle_inequality_l647_64731


namespace reflect_center_of_circle_l647_64709

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), -(p.1))

theorem reflect_center_of_circle : reflect_about_y_eq_neg_x (3, -7) = (7, -3) := by
  sorry

end reflect_center_of_circle_l647_64709


namespace inequality_system_solution_set_l647_64728

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x + 2 > 3 * (1 - x) ∧ 1 - 2 * x ≤ 2) ↔ x > (1 : ℝ) / 4 := by
  sorry

end inequality_system_solution_set_l647_64728


namespace solve_equation_l647_64777

theorem solve_equation : ∃ x : ℝ, x + 1 - 2 + 3 - 4 = 5 - 6 + 7 - 8 ∧ x = 0 := by
  sorry

end solve_equation_l647_64777


namespace sum_of_ages_l647_64761

theorem sum_of_ages (maria_age jose_age : ℕ) : 
  maria_age = 14 → 
  jose_age = maria_age + 12 → 
  maria_age + jose_age = 40 := by
sorry

end sum_of_ages_l647_64761


namespace unique_value_at_three_l647_64787

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * g y + 2 * x) = 2 * x * y + g x

/-- The theorem stating that g(3) = 6 is the only possible value -/
theorem unique_value_at_three
  (g : ℝ → ℝ) (h : SatisfiesFunctionalEquation g) :
  g 3 = 6 :=
sorry

end unique_value_at_three_l647_64787


namespace polynomial_evaluation_l647_64752

theorem polynomial_evaluation (x y p q : ℝ) 
  (h1 : x + y = -p) 
  (h2 : x * y = q) : 
  x * (1 + y) - y * (x * y - 1) - x^2 * y = p * q + q - p :=
by sorry

end polynomial_evaluation_l647_64752


namespace max_red_balls_l647_64766

theorem max_red_balls 
  (total : ℕ) 
  (green : ℕ) 
  (h1 : total = 28) 
  (h2 : green = 12) 
  (h3 : ∀ red : ℕ, red + green < 24) : 
  ∃ max_red : ℕ, max_red = 11 ∧ ∀ red : ℕ, red ≤ max_red := by
sorry

end max_red_balls_l647_64766


namespace trig_problem_l647_64747

theorem trig_problem (α : Real) 
  (h1 : α ∈ Set.Ioo (5 * Real.pi / 4) (3 * Real.pi / 2))
  (h2 : Real.tan α + 1 / Real.tan α = 8) : 
  Real.sin α * Real.cos α = 1 / 8 ∧ Real.sin α - Real.cos α = -Real.sqrt 3 / 2 := by
  sorry

end trig_problem_l647_64747


namespace modulus_of_complex_fraction_l647_64764

theorem modulus_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + i) / i
  Complex.abs z = Real.sqrt 2 := by
sorry

end modulus_of_complex_fraction_l647_64764


namespace truck_travel_distance_l647_64700

/-- Represents the distance a truck can travel -/
def truck_distance (miles_per_gallon : ℝ) (initial_gallons : ℝ) (added_gallons : ℝ) : ℝ :=
  miles_per_gallon * (initial_gallons + added_gallons)

/-- Theorem: A truck traveling 3 miles per gallon with 12 gallons initially and 18 gallons added can travel 90 miles -/
theorem truck_travel_distance :
  truck_distance 3 12 18 = 90 := by
  sorry

#eval truck_distance 3 12 18

end truck_travel_distance_l647_64700


namespace polygon_interior_angle_sum_l647_64737

theorem polygon_interior_angle_sum (n : ℕ) (h1 : n > 2) (h2 : 40 * n = 360) :
  (n - 2) * 180 = 1260 := by
  sorry

end polygon_interior_angle_sum_l647_64737


namespace least_m_satisfying_condition_l647_64703

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The problem statement -/
theorem least_m_satisfying_condition : ∃ m : ℕ, 
  (m > 0) ∧ 
  (∀ k : ℕ, k > 0 → k < m → 
    ¬(∃ p : ℕ, trailingZeros k = p ∧ 
      trailingZeros (2 * k) = ⌊(5 * p : ℚ) / 2⌋)) ∧
  (∃ p : ℕ, trailingZeros m = p ∧ 
    trailingZeros (2 * m) = ⌊(5 * p : ℚ) / 2⌋) ∧
  m = 25 := by
  sorry

end least_m_satisfying_condition_l647_64703


namespace triangle_angle_proof_l647_64713

-- Define a triangle ABC
structure Triangle (α : Type) [Field α] where
  A : α
  B : α
  C : α
  a : α
  b : α
  c : α

-- State the theorem
theorem triangle_angle_proof {α : Type} [Field α] (ABC : Triangle α) :
  ABC.b = 2 * ABC.a →
  ABC.B = ABC.A + 60 →
  ABC.A = 30 :=
by sorry

end triangle_angle_proof_l647_64713


namespace composition_may_have_no_fixed_point_l647_64705

-- Define a type for our functions
def RealFunction := ℝ → ℝ

-- Define what it means for a function to have a fixed point
def has_fixed_point (f : RealFunction) : Prop :=
  ∃ x : ℝ, f x = x

-- State the theorem
theorem composition_may_have_no_fixed_point :
  ∃ (f g : RealFunction),
    has_fixed_point f ∧ 
    has_fixed_point g ∧ 
    ¬(has_fixed_point (f ∘ g)) :=
sorry

end composition_may_have_no_fixed_point_l647_64705


namespace percentage_given_away_l647_64734

def total_amount : ℝ := 100
def amount_kept : ℝ := 80

theorem percentage_given_away : 
  (total_amount - amount_kept) / total_amount * 100 = 20 := by sorry

end percentage_given_away_l647_64734


namespace ellipse_foci_distance_l647_64740

-- Define the points
def p1 : ℝ × ℝ := (1, 3)
def p2 : ℝ × ℝ := (5, -1)
def p3 : ℝ × ℝ := (10, 3)
def p4 : ℝ × ℝ := (5, 7)

-- Define the ellipse
def ellipse (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  let center := ((p1.1 + p3.1) / 2, (p1.2 + p3.2) / 2)
  let a := (p3.1 - p1.1) / 2
  let b := (p4.2 - p2.2) / 2
  (center.1 = (p2.1 + p4.1) / 2) ∧ 
  (center.2 = (p2.2 + p4.2) / 2) ∧
  (a > b) ∧ (b > 0)

-- Theorem statement
theorem ellipse_foci_distance (h : ellipse p1 p2 p3 p4) :
  let a := (p3.1 - p1.1) / 2
  let b := (p4.2 - p2.2) / 2
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 * Real.sqrt 4.25 := by
  sorry

end ellipse_foci_distance_l647_64740


namespace arithmetic_progression_unique_solution_l647_64729

theorem arithmetic_progression_unique_solution (n₁ n₂ : ℕ) (hn : n₁ ≠ n₂) :
  ∃! (a₁ d : ℚ),
    (∀ (n : ℕ), n * (2 * a₁ + (n - 1) * d) / 2 = n^2) ∧
    (n₁ * (2 * a₁ + (n₁ - 1) * d) / 2 = n₁^2) ∧
    (n₂ * (2 * a₁ + (n₂ - 1) * d) / 2 = n₂^2) ∧
    a₁ = 1 ∧ d = 2 :=
by sorry

end arithmetic_progression_unique_solution_l647_64729


namespace scientific_notation_of_32100000_l647_64738

theorem scientific_notation_of_32100000 : 
  32100000 = 3.21 * (10 ^ 7) := by sorry

end scientific_notation_of_32100000_l647_64738


namespace arithmetic_sequence_fraction_zero_l647_64762

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem arithmetic_sequence_fraction_zero 
  (a₁ d : ℚ) (h₁ : a₁ ≠ 0) (h₂ : arithmetic_sequence a₁ d 9 = 0) :
  (arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 8 + 
   arithmetic_sequence a₁ d 11 + arithmetic_sequence a₁ d 16) / 
  (arithmetic_sequence a₁ d 7 + arithmetic_sequence a₁ d 8 + 
   arithmetic_sequence a₁ d 14) = 0 := by
sorry

end arithmetic_sequence_fraction_zero_l647_64762


namespace nineteen_ninetyeight_impossible_l647_64735

/-- The type of operations that can be performed on a number -/
inductive Operation
| Square : Operation
| AddOne : Operation

/-- Apply an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Square => n * n
  | Operation.AddOne => n + 1

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- Apply a sequence of operations to a number -/
def applySequence (n : ℕ) (seq : OperationSequence) : ℕ :=
  seq.foldl applyOperation n

/-- The theorem stating that 19 and 98 cannot be made equal with the same number of operations -/
theorem nineteen_ninetyeight_impossible :
  ∀ (seq : OperationSequence), applySequence 19 seq ≠ applySequence 98 seq :=
sorry

end nineteen_ninetyeight_impossible_l647_64735


namespace student_D_most_stable_smallest_variance_most_stable_l647_64704

-- Define the variances for each student
def variance_A : ℝ := 6
def variance_B : ℝ := 5.5
def variance_C : ℝ := 10
def variance_D : ℝ := 3.8

-- Define a function to determine if a student has the most stable performance
def has_most_stable_performance (student_variance : ℝ) : Prop :=
  student_variance ≤ variance_A ∧
  student_variance ≤ variance_B ∧
  student_variance ≤ variance_C ∧
  student_variance ≤ variance_D

-- Theorem stating that student D has the most stable performance
theorem student_D_most_stable : has_most_stable_performance variance_D := by
  sorry

-- Theorem stating that the student with the smallest variance has the most stable performance
theorem smallest_variance_most_stable :
  ∀ (student_variance : ℝ),
    has_most_stable_performance student_variance →
    student_variance = min (min (min variance_A variance_B) variance_C) variance_D := by
  sorry

end student_D_most_stable_smallest_variance_most_stable_l647_64704


namespace max_value_on_interval_max_value_attained_l647_64795

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem max_value_on_interval (x : ℝ) (h : x ∈ Set.Icc (-1) 1) : f x ≤ 2 := by
  sorry

theorem max_value_attained : ∃ x ∈ Set.Icc (-1) 1, f x = 2 := by
  sorry

end max_value_on_interval_max_value_attained_l647_64795


namespace card_distribution_l647_64781

theorem card_distribution (total : ℕ) (black red : ℕ) (spades diamonds hearts clubs : ℕ) : 
  total = 13 →
  black = 7 →
  red = 6 →
  diamonds = 2 * spades →
  hearts = 2 * diamonds →
  total = spades + diamonds + hearts + clubs →
  black = spades + clubs →
  red = diamonds + hearts →
  clubs = 6 := by
sorry

end card_distribution_l647_64781


namespace square_perimeter_equals_circle_area_l647_64715

theorem square_perimeter_equals_circle_area (r : ℝ) : r = 8 / Real.pi :=
  -- Define the perimeter of the square
  let square_perimeter := 8 * r
  -- Define the area of the circle
  let circle_area := Real.pi * r^2
  -- State that the perimeter of the square equals the area of the circle
  have h : square_perimeter = circle_area := by sorry
  -- Prove that r = 8 / π
  sorry

end square_perimeter_equals_circle_area_l647_64715


namespace five_dollar_four_equals_85_l647_64746

/-- Custom operation $\$$ defined as a $ b = a(2b + 1) + 2ab -/
def dollar_op (a b : ℕ) : ℕ := a * (2 * b + 1) + 2 * a * b

/-- Theorem stating that 5 $ 4 = 85 -/
theorem five_dollar_four_equals_85 : dollar_op 5 4 = 85 := by
  sorry

end five_dollar_four_equals_85_l647_64746


namespace stadium_empty_seats_l647_64702

/-- The number of empty seats in a stadium -/
def empty_seats (total_seats people_present : ℕ) : ℕ :=
  total_seats - people_present

/-- Theorem: Given a stadium with 92 seats and 47 people present, there are 45 empty seats -/
theorem stadium_empty_seats : empty_seats 92 47 = 45 := by
  sorry

end stadium_empty_seats_l647_64702


namespace gcd_sum_and_sum_of_squares_l647_64719

theorem gcd_sum_and_sum_of_squares (a b : ℕ+) (h : Nat.Coprime a b) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 := by
sorry

end gcd_sum_and_sum_of_squares_l647_64719


namespace truck_distance_l647_64776

/-- Given a bike and a truck traveling for 8 hours, where the bike covers 136 miles
    and the truck's speed is 3 mph faster than the bike's, prove that the truck covers 160 miles. -/
theorem truck_distance (time : ℝ) (bike_distance : ℝ) (speed_difference : ℝ) :
  time = 8 ∧ bike_distance = 136 ∧ speed_difference = 3 →
  (bike_distance / time + speed_difference) * time = 160 :=
by sorry

end truck_distance_l647_64776


namespace no_nonneg_int_solutions_l647_64790

theorem no_nonneg_int_solutions : 
  ¬∃ (x : ℕ), 4 * (x - 2) > 2 * (3 * x + 5) := by
sorry

end no_nonneg_int_solutions_l647_64790


namespace solution_to_equation_l647_64793

theorem solution_to_equation :
  ∃! (x y : ℝ), (x - y)^2 + (y - 2 * Real.sqrt x + 2)^2 = (1/2 : ℝ) ∧ x = 1 ∧ y = 1/2 := by
  sorry

end solution_to_equation_l647_64793


namespace sector_area_l647_64714

theorem sector_area (r : ℝ) (θ : ℝ) (h : θ = 72 * π / 180) :
  let A := (θ / (2 * π)) * π * r^2
  r = 20 → A = 80 * π := by
  sorry

end sector_area_l647_64714


namespace smallest_number_is_three_l647_64744

/-- Represents the systematic sampling of classes -/
structure ClassSampling where
  total_classes : Nat
  selected_classes : Nat
  sum_of_selected : Nat

/-- Calculates the smallest number in the systematic sample -/
def smallest_number (sampling : ClassSampling) : Nat :=
  let interval := sampling.total_classes / sampling.selected_classes
  (sampling.sum_of_selected - (interval * (sampling.selected_classes - 1) * sampling.selected_classes / 2)) / sampling.selected_classes

/-- Theorem: The smallest number in the given systematic sample is 3 -/
theorem smallest_number_is_three (sampling : ClassSampling) 
  (h1 : sampling.total_classes = 30)
  (h2 : sampling.selected_classes = 5)
  (h3 : sampling.sum_of_selected = 75) :
  smallest_number sampling = 3 := by
  sorry

#eval smallest_number { total_classes := 30, selected_classes := 5, sum_of_selected := 75 }

end smallest_number_is_three_l647_64744


namespace sum_of_inverse_cubes_of_roots_l647_64722

theorem sum_of_inverse_cubes_of_roots (r s : ℝ) : 
  (3 * r^2 + 5 * r + 2 = 0) → 
  (3 * s^2 + 5 * s + 2 = 0) → 
  (r ≠ s) →
  (1 / r^3 + 1 / s^3 = 25 / 8) :=
by sorry

end sum_of_inverse_cubes_of_roots_l647_64722


namespace mode_of_sample_data_l647_64712

def sample_data : List Int := [-2, 0, 6, 3, 6]

def mode (data : List Int) : Int :=
  data.foldl (fun acc x => if data.count x > data.count acc then x else acc) 0

theorem mode_of_sample_data :
  mode sample_data = 6 := by sorry

end mode_of_sample_data_l647_64712


namespace find_other_number_l647_64717

theorem find_other_number (A B : ℕ) (h1 : A = 24) (h2 : Nat.gcd A B = 17) (h3 : Nat.lcm A B = 312) :
  B = 221 := by
  sorry

end find_other_number_l647_64717


namespace handshake_count_is_correct_handshakes_per_person_is_correct_l647_64751

/-- Represents a social gathering with married couples -/
structure SocialGathering where
  couples : ℕ
  people : ℕ
  handshakes_per_person : ℕ

/-- Calculate the total number of unique handshakes in the gathering -/
def total_handshakes (g : SocialGathering) : ℕ :=
  g.people * g.handshakes_per_person / 2

/-- The specific social gathering described in the problem -/
def our_gathering : SocialGathering :=
  { couples := 8
  , people := 16
  , handshakes_per_person := 12 }

theorem handshake_count_is_correct :
  total_handshakes our_gathering = 96 := by
  sorry

/-- Prove that the number of handshakes per person is correct -/
theorem handshakes_per_person_is_correct (g : SocialGathering) :
  g.handshakes_per_person = g.people - 1 - 3 := by
  sorry

end handshake_count_is_correct_handshakes_per_person_is_correct_l647_64751


namespace inscribed_triangle_area_l647_64767

/-- The area of a triangle inscribed in a circle, given the circle's radius and the ratio of the triangle's sides. -/
theorem inscribed_triangle_area
  (r : ℝ) -- radius of the circle
  (a b c : ℝ) -- ratios of the triangle's sides
  (h_positive : r > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0) -- positivity conditions
  (h_ratio : a^2 + b^2 = c^2) -- Pythagorean theorem condition for the ratios
  (h_diameter : c * (a + b + c)⁻¹ * 2 * r = c) -- condition relating the longest side to the diameter
  : (1/2 * a * b * (a + b + c)⁻¹ * 2 * r)^2 = 216/25 ∧ r = 3 ∧ (a, b, c) = (3, 4, 5) :=
sorry

end inscribed_triangle_area_l647_64767


namespace rectangular_garden_width_l647_64791

theorem rectangular_garden_width (width length area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 675 →
  width = 15 := by
sorry

end rectangular_garden_width_l647_64791


namespace min_value_z_l647_64773

theorem min_value_z (x y : ℝ) : x^2 + 3*y^2 + 8*x - 6*y + 30 ≥ 11 := by
  sorry

end min_value_z_l647_64773


namespace fib_mod_10_periodic_fib_mod_10_smallest_period_l647_64730

/-- Fibonacci sequence modulo 10 -/
def fib_mod_10 : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (fib_mod_10 n + fib_mod_10 (n + 1)) % 10

/-- The period of the Fibonacci sequence modulo 10 -/
def fib_mod_10_period : ℕ := 60

/-- Theorem: The Fibonacci sequence modulo 10 has a period of 60 -/
theorem fib_mod_10_periodic :
  ∀ n : ℕ, fib_mod_10 (n + fib_mod_10_period) = fib_mod_10 n :=
by
  sorry

/-- Theorem: 60 is the smallest positive period of the Fibonacci sequence modulo 10 -/
theorem fib_mod_10_smallest_period :
  ∀ k : ℕ, k > 0 → k < fib_mod_10_period →
    ∃ n : ℕ, fib_mod_10 (n + k) ≠ fib_mod_10 n :=
by
  sorry

end fib_mod_10_periodic_fib_mod_10_smallest_period_l647_64730


namespace password_count_l647_64771

/-- The number of case-insensitive English letters -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of letters in the password -/
def num_password_letters : ℕ := 2

/-- The number of digits in the password -/
def num_password_digits : ℕ := 2

/-- Calculates the number of permutations of r items chosen from n items -/
def permutations (n r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

/-- The number of possible passwords -/
def num_passwords : ℕ := 
  permutations num_letters num_password_letters * permutations num_digits num_password_digits

theorem password_count : 
  num_passwords = permutations num_letters num_password_letters * permutations num_digits num_password_digits :=
by sorry

end password_count_l647_64771


namespace rectangular_prism_volume_l647_64794

theorem rectangular_prism_volume (x y z : ℝ) 
  (eq1 : 2*x + 2*y = 38)
  (eq2 : y + z = 14)
  (eq3 : x + z = 11) :
  x * y * z = 264 :=
by sorry

end rectangular_prism_volume_l647_64794


namespace cubic_polynomial_fits_points_l647_64724

def f (x : ℝ) : ℝ := -10 * x^3 + 20 * x^2 - 60 * x + 200

theorem cubic_polynomial_fits_points :
  f 0 = 200 ∧
  f 1 = 150 ∧
  f 2 = 80 ∧
  f 3 = 0 ∧
  f 4 = -140 :=
by sorry

end cubic_polynomial_fits_points_l647_64724


namespace sibling_ages_sum_l647_64743

theorem sibling_ages_sum (a b c : ℕ+) 
  (h_order : c < b ∧ b < a) 
  (h_product : a * b * c = 72) : 
  a + b + c = 13 := by
  sorry

end sibling_ages_sum_l647_64743


namespace square_13_on_top_l647_64741

/-- Represents a 5x5 grid of numbers -/
def Grid := Fin 5 → Fin 5 → Fin 25

/-- The initial configuration of the grid -/
def initial_grid : Grid :=
  fun i j => ⟨i.val * 5 + j.val + 1, by sorry⟩

/-- Represents a folding operation on the grid -/
def Fold := Grid → Grid

/-- Fold the top half over the bottom half -/
def fold1 : Fold := sorry

/-- Fold the bottom half over the top half -/
def fold2 : Fold := sorry

/-- Fold the left half over the right half -/
def fold3 : Fold := sorry

/-- Fold the right half over the left half -/
def fold4 : Fold := sorry

/-- Fold diagonally from bottom left to top right -/
def fold5 : Fold := sorry

/-- The final configuration after all folds -/
def final_grid : Grid :=
  fold5 (fold4 (fold3 (fold2 (fold1 initial_grid))))

/-- The theorem stating that square 13 is on top after all folds -/
theorem square_13_on_top :
  final_grid 0 0 = ⟨13, by sorry⟩ := by sorry

end square_13_on_top_l647_64741


namespace min_value_theorem_l647_64711

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, a * x + b * y - 2 = 0 → x^2 + y^2 - 6*x - 4*y - 12 = 0) →
  (∃ x y : ℝ, a * x + b * y - 2 = 0 ∧ x^2 + y^2 - 6*x - 4*y - 12 = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∀ x y : ℝ, a' * x + b' * y - 2 = 0 → x^2 + y^2 - 6*x - 4*y - 12 = 0) →
    (∃ x y : ℝ, a' * x + b' * y - 2 = 0 ∧ x^2 + y^2 - 6*x - 4*y - 12 = 0) →
    3/a + 2/b ≤ 3/a' + 2/b') →
  3/a + 2/b = 25/2 := by
sorry

end min_value_theorem_l647_64711


namespace x_range_for_negative_f_l647_64716

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + a + 1

-- Define the theorem
theorem x_range_for_negative_f :
  (∀ x : ℝ, ∀ a ∈ Set.Icc (-1 : ℝ) 1, f a x < 0) →
  (∀ x : ℝ, f (-1) x < 0 ∧ f 1 x < 0) →
  {x : ℝ | 1 < x ∧ x < 2} = {x : ℝ | ∀ a ∈ Set.Icc (-1 : ℝ) 1, f a x < 0} :=
by sorry


end x_range_for_negative_f_l647_64716


namespace cat_and_mouse_positions_l647_64799

/-- Represents the position of the cat -/
inductive CatPosition
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

/-- Represents the position of the mouse -/
inductive MousePosition
  | TopLeft
  | TopMiddle
  | TopRight
  | RightMiddle
  | BottomRight
  | BottomMiddle
  | BottomLeft
  | LeftMiddle

/-- The number of squares in the cat's cycle -/
def catCycleLength : Nat := 4

/-- The number of segments in the mouse's cycle -/
def mouseCycleLength : Nat := 8

/-- The total number of moves -/
def totalMoves : Nat := 317

/-- Function to determine the cat's position after a given number of moves -/
def catPositionAfterMoves (moves : Nat) : CatPosition :=
  match moves % catCycleLength with
  | 0 => CatPosition.BottomLeft
  | 1 => CatPosition.TopLeft
  | 2 => CatPosition.TopRight
  | 3 => CatPosition.BottomRight
  | _ => CatPosition.TopLeft  -- This case should never occur due to the modulo operation

/-- Function to determine the mouse's position after a given number of moves -/
def mousePositionAfterMoves (moves : Nat) : MousePosition :=
  match moves % mouseCycleLength with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.LeftMiddle
  | 2 => MousePosition.BottomLeft
  | 3 => MousePosition.BottomMiddle
  | 4 => MousePosition.BottomRight
  | 5 => MousePosition.RightMiddle
  | 6 => MousePosition.TopRight
  | 7 => MousePosition.TopMiddle
  | _ => MousePosition.TopLeft  -- This case should never occur due to the modulo operation

theorem cat_and_mouse_positions :
  catPositionAfterMoves totalMoves = CatPosition.TopLeft ∧
  mousePositionAfterMoves totalMoves = MousePosition.BottomMiddle := by
  sorry

end cat_and_mouse_positions_l647_64799


namespace tangent_point_and_perpendicular_line_l647_64721

/-- The curve y = x^3 + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

/-- Point P₀ -/
def P₀ : ℝ × ℝ := (-1, -4)

/-- The slope of the line parallel to the tangent at P₀ -/
def m : ℝ := 4

/-- The equation of the line perpendicular to the tangent at P₀ -/
def l (x y : ℝ) : Prop := x + 4*y + 17 = 0

theorem tangent_point_and_perpendicular_line :
  (∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ f' x = m) →  -- P₀ exists in third quadrant with slope m
  (P₀.1 = -1 ∧ P₀.2 = -4) ∧  -- P₀ has coordinates (-1, -4)
  (∀ (x y : ℝ), l x y ↔ y - P₀.2 = -(1/m) * (x - P₀.1)) :=  -- l is perpendicular to tangent at P₀
by sorry

end tangent_point_and_perpendicular_line_l647_64721


namespace election_majority_l647_64739

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 600 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = 240 := by
  sorry

end election_majority_l647_64739


namespace total_money_divided_l647_64779

/-- The total amount of money divided among A, B, and C is 120, given the specified conditions. -/
theorem total_money_divided (a b c : ℕ) : 
  b = 20 → a = b + 20 → c = a + 20 → a + b + c = 120 := by
  sorry

end total_money_divided_l647_64779


namespace month_days_l647_64732

theorem month_days (days_took_capsules days_forgot_capsules : ℕ) 
  (h1 : days_took_capsules = 29)
  (h2 : days_forgot_capsules = 2) : 
  days_took_capsules + days_forgot_capsules = 31 := by
sorry

end month_days_l647_64732


namespace largest_fraction_l647_64733

theorem largest_fraction : 
  (26 : ℚ) / 51 > 101 / 203 ∧ 
  (26 : ℚ) / 51 > 47 / 93 ∧ 
  (26 : ℚ) / 51 > 5 / 11 ∧ 
  (26 : ℚ) / 51 > 199 / 401 := by
  sorry

end largest_fraction_l647_64733


namespace flag_arrangement_theorem_l647_64754

/-- The number of distinguishable flagpoles -/
def num_poles : ℕ := 2

/-- The total number of flags -/
def total_flags : ℕ := 25

/-- The number of blue flags -/
def blue_flags : ℕ := 15

/-- The number of green flags -/
def green_flags : ℕ := 10

/-- Function to calculate the number of distinguishable arrangements -/
def calculate_arrangements (np gf bf : ℕ) : ℕ := sorry

/-- Theorem stating that the number of distinguishable arrangements,
    when divided by 1000, yields a remainder of 122 -/
theorem flag_arrangement_theorem :
  calculate_arrangements num_poles green_flags blue_flags % 1000 = 122 := by sorry

end flag_arrangement_theorem_l647_64754


namespace savings_calculation_l647_64798

/-- Given a person's income and expenditure ratio, and their income, calculate their savings -/
def calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (income : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem stating that given the specific income-expenditure ratio and income, the savings are 3000 -/
theorem savings_calculation :
  let income_ratio : ℕ := 10
  let expenditure_ratio : ℕ := 7
  let income : ℕ := 10000
  calculate_savings income_ratio expenditure_ratio income = 3000 := by
  sorry

#eval calculate_savings 10 7 10000

end savings_calculation_l647_64798


namespace equation_solution_l647_64742

theorem equation_solution : ∃ x : ℝ, (12 - 2 * x = 6) ∧ (x = 3) := by
  sorry

end equation_solution_l647_64742


namespace row_5_seat_4_denotation_l647_64753

/-- Represents a seat in a theater -/
structure Seat where
  row : ℕ
  number : ℕ

/-- Converts a seat to its denotation as an ordered pair -/
def seat_denotation (s : Seat) : ℕ × ℕ := (s.row, s.number)

/-- Given condition: "Row 4, Seat 5" is denoted as (4, 5) -/
axiom example_seat : seat_denotation ⟨4, 5⟩ = (4, 5)

/-- Theorem: The denotation of "Row 5, Seat 4" is (5, 4) -/
theorem row_5_seat_4_denotation : seat_denotation ⟨5, 4⟩ = (5, 4) := by
  sorry

end row_5_seat_4_denotation_l647_64753


namespace mouse_seeds_l647_64796

theorem mouse_seeds (mouse_seeds_per_burrow rabbit_seeds_per_burrow : ℕ)
  (mouse_burrows rabbit_burrows : ℕ) :
  mouse_seeds_per_burrow = 4 →
  rabbit_seeds_per_burrow = 6 →
  mouse_seeds_per_burrow * mouse_burrows = rabbit_seeds_per_burrow * rabbit_burrows →
  mouse_burrows = rabbit_burrows + 2 →
  mouse_seeds_per_burrow * mouse_burrows = 24 :=
by
  sorry

end mouse_seeds_l647_64796


namespace last_two_digits_sum_l647_64792

theorem last_two_digits_sum (n : ℕ) : n = 7^15 + 13^15 → n % 100 = 0 := by
  sorry

end last_two_digits_sum_l647_64792


namespace total_amount_not_unique_l647_64768

/-- Represents the investment scenario with two different interest rates -/
structure Investment where
  x : ℝ  -- Amount invested at 10%
  y : ℝ  -- Amount invested at 8%
  T : ℝ  -- Total amount invested

/-- The conditions of the investment problem -/
def investment_conditions (inv : Investment) : Prop :=
  inv.x * 0.10 - inv.y * 0.08 = 65 ∧ inv.x + inv.y = inv.T

/-- Theorem stating that the total amount T cannot be uniquely determined -/
theorem total_amount_not_unique :
  ∃ (inv1 inv2 : Investment), 
    investment_conditions inv1 ∧ 
    investment_conditions inv2 ∧ 
    inv1.T ≠ inv2.T :=
sorry

#check total_amount_not_unique

end total_amount_not_unique_l647_64768


namespace cube_shadow_problem_l647_64750

/-- Given a cube with edge length 2 cm and a light source x cm above one upper vertex,
    if the shadow area (excluding the area beneath the cube) is 192 cm²,
    then the greatest integer not exceeding 1000x is 25780. -/
theorem cube_shadow_problem (x : ℝ) : 
  let cube_edge : ℝ := 2
  let shadow_area : ℝ := 192
  let total_shadow_area : ℝ := shadow_area + cube_edge^2
  let shadow_side : ℝ := (total_shadow_area).sqrt
  x = (shadow_side - cube_edge) / 2 →
  ⌊1000 * x⌋ = 25780 := by sorry

end cube_shadow_problem_l647_64750


namespace goldfish_equality_month_l647_64710

theorem goldfish_equality_month : ∃ n : ℕ, n > 0 ∧ 3^(n+1) = 125 * 5^n ∧ ∀ m : ℕ, 0 < m ∧ m < n → 3^(m+1) ≠ 125 * 5^m :=
by
  sorry

end goldfish_equality_month_l647_64710


namespace range_of_m_chord_length_l647_64759

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → m < 5 :=
sorry

-- Theorem for the length of chord MN when m = 4
theorem chord_length :
  let m : ℝ := 4
  ∃ M N : ℝ × ℝ,
    circle_equation M.1 M.2 m ∧
    circle_equation N.1 N.2 m ∧
    line_equation M.1 M.2 ∧
    line_equation N.1 N.2 ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 4 * Real.sqrt 5 / 5 :=
sorry

end range_of_m_chord_length_l647_64759


namespace jerry_reaches_first_l647_64785

-- Define the points
variable (A B C D : Point)

-- Define the distances
variable (AB BD AC CD : ℝ)

-- Define the speeds
variable (speed_tom speed_jerry : ℝ)

-- Define the delay
variable (delay : ℝ)

-- Theorem statement
theorem jerry_reaches_first (h1 : AB = 32) (h2 : BD = 12) (h3 : AC = 13) (h4 : CD = 27)
  (h5 : speed_tom = 5) (h6 : speed_jerry = 4) (h7 : delay = 5) :
  (AB + BD) / speed_jerry < delay + (AC + CD) / speed_tom := by
  sorry

end jerry_reaches_first_l647_64785


namespace complex_equation_solution_l647_64765

theorem complex_equation_solution (z : ℂ) : 
  (1 + Complex.I)^2 * z = 3 + 2 * Complex.I → z = 1 - (3/2) * Complex.I :=
by sorry

end complex_equation_solution_l647_64765


namespace fixed_point_on_line_l647_64770

-- Define the line equation
def line_equation (x y m : ℝ) : Prop := y - 2 = m * x + m

-- Theorem statement
theorem fixed_point_on_line :
  ∀ m : ℝ, line_equation (-1) 2 m :=
by sorry

end fixed_point_on_line_l647_64770


namespace joseph_baseball_cards_l647_64783

theorem joseph_baseball_cards (X : ℚ) : 
  X - (3/8) * X - 2 = (1/2) * X → X = 16 := by
  sorry

end joseph_baseball_cards_l647_64783


namespace rectangular_solid_depth_l647_64782

/-- The surface area of a rectangular solid -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem rectangular_solid_depth :
  ∃ (h : ℝ), h > 0 ∧ surface_area 5 4 h = 58 → h = 1 := by
  sorry

end rectangular_solid_depth_l647_64782


namespace cistern_emptying_time_l647_64736

/-- Given a cistern with normal fill time and leak-affected fill time, 
    calculate the time to empty through the leak. -/
theorem cistern_emptying_time 
  (normal_fill_time : ℝ) 
  (leak_fill_time : ℝ) 
  (h1 : normal_fill_time = 2) 
  (h2 : leak_fill_time = 4) : 
  (1 / (1 / normal_fill_time - 1 / leak_fill_time)) = 4 := by
  sorry

#check cistern_emptying_time

end cistern_emptying_time_l647_64736


namespace valentines_packs_given_away_l647_64774

def initial_valentines : ℕ := 450
def remaining_valentines : ℕ := 70
def valentines_per_pack : ℕ := 10

theorem valentines_packs_given_away : 
  (initial_valentines - remaining_valentines) / valentines_per_pack = 38 := by
  sorry

end valentines_packs_given_away_l647_64774


namespace geometric_sequence_common_ratio_l647_64788

/-- Given a geometric sequence {a_n} with a_1 = 1/2 and a_4 = 4, 
    the common ratio q is equal to 2. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) 
  (h_a1 : a 1 = 1/2) 
  (h_a4 : a 4 = 4) :
  q = 2 :=
sorry

end geometric_sequence_common_ratio_l647_64788


namespace power_three_thirds_of_675_l647_64718

theorem power_three_thirds_of_675 : (675 : ℝ) ^ (3/3) = 675 := by
  sorry

end power_three_thirds_of_675_l647_64718


namespace q_is_false_l647_64706

theorem q_is_false (p q : Prop) (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end q_is_false_l647_64706


namespace min_distance_parallel_lines_l647_64707

/-- The minimum distance between two points on parallel lines -/
theorem min_distance_parallel_lines :
  let l₁ : ℝ → ℝ → Prop := λ x y => x + 3 * y - 9 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y => x + 3 * y + 1 = 0
  ∃ (d : ℝ), d = Real.sqrt 10 ∧
    ∀ (P₁ P₂ : ℝ × ℝ), l₁ P₁.1 P₁.2 → l₂ P₂.1 P₂.2 →
      Real.sqrt ((P₁.1 - P₂.1)^2 + (P₁.2 - P₂.2)^2) ≥ d :=
by sorry

end min_distance_parallel_lines_l647_64707


namespace milford_lake_algae_increase_l647_64784

/-- The increase in algae plants in Milford Lake -/
def algae_increase (original : ℕ) (current : ℕ) : ℕ :=
  current - original

/-- Theorem stating the increase in algae plants in Milford Lake -/
theorem milford_lake_algae_increase :
  algae_increase 809 3263 = 2454 := by
  sorry

end milford_lake_algae_increase_l647_64784


namespace storks_joined_l647_64763

theorem storks_joined (initial_birds initial_storks final_difference : ℕ) :
  initial_birds = 4 →
  initial_storks = 3 →
  final_difference = 5 →
  ∃ joined : ℕ, initial_storks + joined = initial_birds + final_difference ∧ joined = 6 :=
by
  sorry

end storks_joined_l647_64763


namespace vectors_in_plane_implies_x_eq_neg_one_l647_64756

-- Define the vectors
def a (x : ℝ) : Fin 3 → ℝ := ![1, x, -2]
def b : Fin 3 → ℝ := ![0, 1, 2]
def c : Fin 3 → ℝ := ![1, 0, 0]

-- Define the condition that vectors lie in the same plane
def vectors_in_same_plane (x : ℝ) : Prop :=
  ∃ (m n : ℝ), a x = m • b + n • c

-- Theorem statement
theorem vectors_in_plane_implies_x_eq_neg_one :
  ∀ x : ℝ, vectors_in_same_plane x → x = -1 :=
by sorry

end vectors_in_plane_implies_x_eq_neg_one_l647_64756


namespace isosceles_triangle_perimeter_l647_64769

/-- An isosceles triangle with side lengths 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 7 ∧ c = 7 →  -- Two sides are 7, one side is 3
  a + b + c = 17 :=        -- The perimeter is 17
by sorry

end isosceles_triangle_perimeter_l647_64769


namespace starting_lineup_combinations_l647_64720

def total_players : ℕ := 15
def predetermined_players : ℕ := 3
def players_to_choose : ℕ := 2

theorem starting_lineup_combinations :
  Nat.choose (total_players - predetermined_players) players_to_choose = 66 := by
  sorry

end starting_lineup_combinations_l647_64720


namespace no_equilateral_right_triangle_l647_64758

theorem no_equilateral_right_triangle :
  ¬ ∃ (a b c : ℝ) (A B C : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
    A > 0 ∧ B > 0 ∧ C > 0 ∧  -- Positive angles
    a = b ∧ b = c ∧          -- Equilateral condition
    A = 90 ∧                 -- Right angle condition
    A + B + C = 180          -- Sum of angles in a triangle
    := by sorry

end no_equilateral_right_triangle_l647_64758


namespace gcd_g_x_eq_six_l647_64745

def g (x : ℤ) : ℤ := (5*x+3)*(8*x+2)*(11*x+7)*(3*x+5)

theorem gcd_g_x_eq_six (x : ℤ) (h : 18432 ∣ x) : 
  Nat.gcd (g x).natAbs x.natAbs = 6 := by
  sorry

end gcd_g_x_eq_six_l647_64745


namespace largest_angle_in_triangle_l647_64757

theorem largest_angle_in_triangle (y : ℝ) : 
  y + 60 + 70 = 180 → 
  max y (max 60 70) = 70 := by
sorry

end largest_angle_in_triangle_l647_64757


namespace fabric_cutting_l647_64789

theorem fabric_cutting (initial_length : ℚ) (cut_length : ℚ) (desired_length : ℚ) :
  initial_length = 2/3 →
  cut_length = 1/6 →
  desired_length = 1/2 →
  initial_length - cut_length = desired_length :=
by sorry

end fabric_cutting_l647_64789


namespace same_color_probability_l647_64786

def total_plates : ℕ := 13
def red_plates : ℕ := 7
def blue_plates : ℕ := 6
def plates_to_select : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates plates_to_select + Nat.choose blue_plates plates_to_select) /
  Nat.choose total_plates plates_to_select = 55 / 286 :=
by sorry

end same_color_probability_l647_64786


namespace no_solution_to_system_l647_64778

theorem no_solution_to_system :
  ∀ x : ℝ, ¬(x^5 + 3*x^4 + 5*x^3 + 5*x^2 + 6*x + 2 = 0 ∧ x^3 + 3*x^2 + 4*x + 1 = 0) := by
  sorry

end no_solution_to_system_l647_64778


namespace sum_of_five_variables_l647_64708

theorem sum_of_five_variables (a b c d e : ℚ) : 
  (a + 1 = b + 2) ∧ 
  (b + 2 = c + 3) ∧ 
  (c + 3 = d + 4) ∧ 
  (d + 4 = e + 5) ∧ 
  (e + 5 = a + b + c + d + e + 10) → 
  a + b + c + d + e = -35/4 := by
sorry

end sum_of_five_variables_l647_64708


namespace equal_expressions_l647_64775

theorem equal_expressions (x : ℝ) : 2 * x - 1 = 3 * x + 3 ↔ x = -4 := by
  sorry

end equal_expressions_l647_64775


namespace square_properties_l647_64748

structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

def square : Square := {
  A := (0, 0),
  B := (-5, -3),
  C := (-4, -8),
  D := (1, -5)
}

theorem square_properties (s : Square) (h : s = square) :
  let side_length := Real.sqrt ((s.B.1 - s.A.1)^2 + (s.B.2 - s.A.2)^2)
  (side_length^2 = 34) ∧ (4 * side_length = 4 * Real.sqrt 34) := by
  sorry

#check square_properties

end square_properties_l647_64748


namespace m_squared_plus_inverse_squared_plus_six_l647_64725

theorem m_squared_plus_inverse_squared_plus_six (m : ℝ) (h : m + 1/m = 10) : 
  m^2 + 1/m^2 + 6 = 104 := by
  sorry

end m_squared_plus_inverse_squared_plus_six_l647_64725


namespace work_completion_time_l647_64755

theorem work_completion_time (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (1/a + 1/b = 1/4) → (1/b = 1/6) → (1/a = 1/12) := by
  sorry

end work_completion_time_l647_64755


namespace other_solution_quadratic_l647_64780

theorem other_solution_quadratic (x : ℚ) :
  56 * (5/7)^2 + 27 = 89 * (5/7) - 8 →
  56 * (7/8)^2 + 27 = 89 * (7/8) - 8 :=
by sorry

end other_solution_quadratic_l647_64780


namespace second_number_calculation_l647_64749

theorem second_number_calculation (A B : ℝ) : 
  A = 6400 → 
  0.05 * A = 0.20 * B + 190 → 
  B = 650 := by
sorry

end second_number_calculation_l647_64749


namespace power_sine_inequality_l647_64727

theorem power_sine_inequality (α : Real) (x₁ x₂ : Real) 
  (h1 : 0 < α ∧ α < π)
  (h2 : 0 < x₁)
  (h3 : x₁ < x₂) :
  (x₂ / x₁) ^ (Real.sin α) > 1 := by
  sorry

end power_sine_inequality_l647_64727


namespace donalds_oranges_l647_64797

theorem donalds_oranges (initial : ℕ) : initial + 5 = 9 → initial = 4 := by
  sorry

end donalds_oranges_l647_64797


namespace quadratic_solution_difference_squared_l647_64760

theorem quadratic_solution_difference_squared :
  ∀ (α β : ℝ),
    α ≠ β →
    α^2 - 3*α + 2 = 0 →
    β^2 - 3*β + 2 = 0 →
    (α - β)^2 = 1 := by
  sorry

end quadratic_solution_difference_squared_l647_64760


namespace reflection_line_is_x_equals_zero_l647_64772

-- Define the points
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (5, 7)
def R : ℝ × ℝ := (-2, 5)
def P' : ℝ × ℝ := (-1, 2)
def Q' : ℝ × ℝ := (-5, 7)
def R' : ℝ × ℝ := (2, 5)

-- Define the reflection line
def M : Set (ℝ × ℝ) := {(x, y) | x = 0}

-- Theorem statement
theorem reflection_line_is_x_equals_zero :
  (∀ (x y : ℝ), (x, y) ∈ M ↔ x = 0) ∧
  (P.1 + P'.1 = 0) ∧ (P.2 = P'.2) ∧
  (Q.1 + Q'.1 = 0) ∧ (Q.2 = Q'.2) ∧
  (R.1 + R'.1 = 0) ∧ (R.2 = R'.2) :=
sorry


end reflection_line_is_x_equals_zero_l647_64772


namespace ellipse_complementary_angles_point_l647_64701

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/3 + y^2/2 = 1

-- Define the right focus of ellipse C
def right_focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the right focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - right_focus.1)

-- Define the property of complementary angles of inclination
def complementary_angles (P A B : ℝ × ℝ) : Prop :=
  (A.2 - P.2) / (A.1 - P.1) + (B.2 - P.2) / (B.1 - P.1) = 0

-- Main theorem
theorem ellipse_complementary_angles_point :
  ∃ (P : ℝ × ℝ), P.2 = 0 ∧
  ∀ (k : ℝ) (A B : ℝ × ℝ),
    k ≠ 0 →
    line_through_focus k A.1 A.2 →
    line_through_focus k B.1 B.2 →
    ellipse_C A.1 A.2 →
    ellipse_C B.1 B.2 →
    A ≠ B →
    complementary_angles P A B :=
sorry

end ellipse_complementary_angles_point_l647_64701


namespace max_share_is_18200_l647_64723

/-- Represents the profit share of a partner -/
structure PartnerShare where
  ratio : Nat
  bonus : Bool

/-- Calculates the maximum share given the total profit, bonus amount, and partner shares -/
def maxShare (totalProfit : ℚ) (bonusAmount : ℚ) (shares : List PartnerShare) : ℚ :=
  sorry

/-- The main theorem -/
theorem max_share_is_18200 :
  let shares := [
    ⟨4, false⟩,
    ⟨3, false⟩,
    ⟨2, true⟩,
    ⟨6, false⟩
  ]
  maxShare 45000 500 shares = 18200 := by sorry

end max_share_is_18200_l647_64723


namespace salary_decrease_percentage_typist_salary_problem_l647_64726

theorem salary_decrease_percentage 
  (original_salary : ℝ) 
  (increase_percentage : ℝ) 
  (final_salary : ℝ) : ℝ :=
  let increased_salary := original_salary * (1 + increase_percentage / 100)
  let decrease_percentage := (increased_salary - final_salary) / increased_salary * 100
  decrease_percentage

theorem typist_salary_problem : 
  salary_decrease_percentage 2000 10 2090 = 5 := by
  sorry

end salary_decrease_percentage_typist_salary_problem_l647_64726

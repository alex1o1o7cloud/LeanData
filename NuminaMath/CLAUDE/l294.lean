import Mathlib

namespace emily_marbles_l294_29405

/-- Emily's marble problem -/
theorem emily_marbles :
  let initial_marbles : ℕ := 6
  let megan_gives := 2 * initial_marbles
  let emily_new_total := initial_marbles + megan_gives
  let emily_gives_back := emily_new_total / 2 + 1
  let emily_final := emily_new_total - emily_gives_back
  emily_final = 8 := by sorry

end emily_marbles_l294_29405


namespace logarithmic_equation_solutions_l294_29478

theorem logarithmic_equation_solutions (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, x > 0 → x ≠ 1 →
    ((3 * (Real.log x / Real.log a) - 2) * (Real.log a / Real.log x)^2 = Real.log x / (Real.log a / 2) - 3) ↔
    (x = 1/a ∨ x = Real.sqrt a ∨ x = a^2) :=
by sorry

end logarithmic_equation_solutions_l294_29478


namespace chess_swimming_enrollment_percentage_l294_29488

theorem chess_swimming_enrollment_percentage 
  (total_students : ℕ) 
  (chess_percentage : ℚ) 
  (swimming_students : ℕ) 
  (h1 : total_students = 2000)
  (h2 : chess_percentage = 1/10)
  (h3 : swimming_students = 100) :
  (swimming_students : ℚ) / ((chess_percentage * total_students) : ℚ) = 1/2 :=
by sorry

end chess_swimming_enrollment_percentage_l294_29488


namespace age_difference_l294_29477

def age_problem (a b c : ℕ) : Prop :=
  b = 2 * c ∧ a + b + c = 27 ∧ b = 10

theorem age_difference (a b c : ℕ) (h : age_problem a b c) : a - b = 2 := by
  sorry

end age_difference_l294_29477


namespace two_carp_heavier_than_three_bream_l294_29462

/-- Represents the weight of a fish species -/
structure FishWeight where
  weight : ℝ
  weight_pos : weight > 0

/-- Given that 6 crucian carps are lighter than 5 perches and 6 crucian carps are heavier than 10 breams,
    prove that 2 crucian carp are heavier than 3 breams. -/
theorem two_carp_heavier_than_three_bream 
  (carp perch bream : FishWeight)
  (h1 : 6 * carp.weight < 5 * perch.weight)
  (h2 : 6 * carp.weight > 10 * bream.weight) :
  2 * carp.weight > 3 * bream.weight := by
sorry

end two_carp_heavier_than_three_bream_l294_29462


namespace remaining_three_digit_numbers_l294_29411

/-- The count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers where the first and last digits are the same
    and the middle digit is different -/
def excluded_numbers : ℕ := 81

/-- Theorem: The count of three-digit numbers excluding those where the first and last digits
    are the same and the middle digit is different is equal to 819 -/
theorem remaining_three_digit_numbers :
  total_three_digit_numbers - excluded_numbers = 819 := by
  sorry

end remaining_three_digit_numbers_l294_29411


namespace triangle_area_l294_29487

/-- The area of a triangle with vertices A(2, 2), B(8, 2), and C(5, 11) is 27 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (8, 2)
  let C : ℝ × ℝ := (5, 11)
  let area := (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))
  area = 27 := by sorry

end triangle_area_l294_29487


namespace erica_safari_elephants_l294_29434

/-- The number of elephants Erica saw on her safari --/
def elephants_seen (total_animals : ℕ) (lions_saturday : ℕ) (animals_sunday_monday : ℕ) : ℕ :=
  total_animals - lions_saturday - animals_sunday_monday

/-- Theorem stating the number of elephants Erica saw on Saturday --/
theorem erica_safari_elephants :
  elephants_seen 20 3 15 = 2 := by
  sorry

end erica_safari_elephants_l294_29434


namespace basketball_game_second_half_score_l294_29450

/-- Represents the points scored by a team in each quarter -/
structure QuarterlyPoints where
  q1 : ℝ
  q2 : ℝ
  q3 : ℝ
  q4 : ℝ

/-- The game between Raiders and Wildcats -/
structure BasketballGame where
  raiders : QuarterlyPoints
  wildcats : QuarterlyPoints

def BasketballGame.total_score (game : BasketballGame) : ℝ :=
  game.raiders.q1 + game.raiders.q2 + game.raiders.q3 + game.raiders.q4 +
  game.wildcats.q1 + game.wildcats.q2 + game.wildcats.q3 + game.wildcats.q4

def BasketballGame.second_half_score (game : BasketballGame) : ℝ :=
  game.raiders.q3 + game.raiders.q4 + game.wildcats.q3 + game.wildcats.q4

theorem basketball_game_second_half_score
  (a b d r : ℝ)
  (hr : r ≥ 1)
  (game : BasketballGame)
  (h1 : game.raiders = ⟨a, a*r, a*r^2, a*r^3⟩)
  (h2 : game.wildcats = ⟨b, b+d, b+2*d, b+3*d⟩)
  (h3 : game.raiders.q1 = game.wildcats.q1)
  (h4 : game.total_score = 2 * game.raiders.q1 + game.raiders.q2 + game.raiders.q3 + game.raiders.q4 +
                           game.wildcats.q2 + game.wildcats.q3 + game.wildcats.q4)
  (h5 : game.total_score = 2 * (4*b + 6*d + 3))
  (h6 : ∀ q, q ∈ [game.raiders.q1, game.raiders.q2, game.raiders.q3, game.raiders.q4,
                  game.wildcats.q1, game.wildcats.q2, game.wildcats.q3, game.wildcats.q4] → q ≤ 100) :
  game.second_half_score = 112 :=
sorry

end basketball_game_second_half_score_l294_29450


namespace complex_norm_product_l294_29423

theorem complex_norm_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end complex_norm_product_l294_29423


namespace soup_distribution_l294_29404

-- Define the total amount of soup
def total_soup : ℚ := 1

-- Define the number of grandchildren
def num_children : ℕ := 5

-- Define the amount taken by Ângela and Daniela
def angela_daniela_portion : ℚ := 2 / 5

-- Define Laura's division
def laura_division : ℕ := 5

-- Define João's division
def joao_division : ℕ := 4

-- Define the container size in ml
def container_size : ℕ := 100

-- Theorem statement
theorem soup_distribution (
  laura_portion : ℚ)
  (toni_portion : ℚ)
  (min_soup_amount : ℚ) :
  laura_portion = 3 / 25 ∧
  toni_portion = 9 / 25 ∧
  min_soup_amount = 5 / 2 := by
  sorry

-- Note: The proof is omitted as per the instructions

end soup_distribution_l294_29404


namespace unique_solution_system_l294_29424

theorem unique_solution_system (m : ℝ) : ∃! (x y : ℝ), 
  ((m + 1) * x - y - 3 * m = 0) ∧ (4 * x + (m - 1) * y + 7 = 0) := by
  sorry

end unique_solution_system_l294_29424


namespace p_sufficient_not_necessary_l294_29418

-- Define a complex number
def complex_number (a b : ℝ) := a + b * Complex.I

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Define condition p
def condition_p (a b : ℝ) : Prop := is_purely_imaginary (complex_number a b)

-- Define condition q
def condition_q (a b : ℝ) : Prop := a = 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary :
  (∀ a b : ℝ, condition_p a b → condition_q a b) ∧
  (∃ a b : ℝ, condition_q a b ∧ ¬condition_p a b) :=
sorry

end p_sufficient_not_necessary_l294_29418


namespace stating_repeating_decimal_equals_fraction_l294_29446

/-- Represents a repeating decimal where the fractional part is 0.325325325... -/
def repeating_decimal : ℚ := 3/10 + 25/990

/-- The fraction 161/495 in its lowest terms -/
def target_fraction : ℚ := 161/495

/-- 
Theorem stating that the repeating decimal 0.3̅25̅ is equal to the fraction 161/495
-/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end stating_repeating_decimal_equals_fraction_l294_29446


namespace unsold_books_percentage_l294_29486

-- Define the initial stock and daily sales
def initial_stock : ℕ := 620
def daily_sales : List ℕ := [50, 82, 60, 48, 40]

-- Define the theorem
theorem unsold_books_percentage :
  let total_sold := daily_sales.sum
  let unsold := initial_stock - total_sold
  let percentage_unsold := (unsold : ℚ) / (initial_stock : ℚ) * 100
  ∃ ε > 0, abs (percentage_unsold - 54.84) < ε :=
by
  sorry

end unsold_books_percentage_l294_29486


namespace reimbursement_difference_l294_29439

/-- The problem of reimbursement in a group activity --/
theorem reimbursement_difference (tom emma harry : ℝ) : 
  tom = 95 →
  emma = 140 →
  harry = 165 →
  let total := tom + emma + harry
  let share := total / 3
  let t := share - tom
  let e := share - emma
  e - t = -45 := by
  sorry

end reimbursement_difference_l294_29439


namespace total_animals_is_100_l294_29485

/-- Given the number of rabbits, calculates the total number of chickens, ducks, and rabbits. -/
def total_animals (num_rabbits : ℕ) : ℕ :=
  let num_ducks := num_rabbits + 12
  let num_chickens := 5 * num_ducks
  num_chickens + num_ducks + num_rabbits

/-- Theorem stating that given 4 rabbits, the total number of animals is 100. -/
theorem total_animals_is_100 : total_animals 4 = 100 := by
  sorry

end total_animals_is_100_l294_29485


namespace factorial_not_ending_19760_l294_29492

theorem factorial_not_ending_19760 (n : ℕ+) : ¬ ∃ k : ℕ, (n!:ℕ) % (10^(k+5)) = 19760 * 10^k :=
sorry

end factorial_not_ending_19760_l294_29492


namespace max_ab_squared_l294_29413

theorem max_ab_squared (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  ∃ (m : ℝ), m = (4 * Real.sqrt 6) / 9 ∧ ∀ x y : ℝ, 0 < x → 0 < y → x + y = 2 → x * y^2 ≤ m :=
sorry

end max_ab_squared_l294_29413


namespace second_stack_height_difference_l294_29453

def stack_problem (h : ℕ) : Prop :=
  let first_stack := 7
  let second_stack := h
  let third_stack := h + 7
  let fallen_blocks := first_stack + (second_stack - 2) + (third_stack - 3)
  (fallen_blocks = 33) ∧ (second_stack > first_stack)

theorem second_stack_height_difference : ∃ h : ℕ, stack_problem h ∧ (h - 7 = 5) :=
sorry

end second_stack_height_difference_l294_29453


namespace two_integers_problem_l294_29408

theorem two_integers_problem :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x - y = 8 ∧ x * y = 180 ∧ x + y = 28 := by
  sorry

end two_integers_problem_l294_29408


namespace extreme_value_and_maximum_l294_29444

-- Define the function f and its derivative
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x) + a * Real.cos x + x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (2 * x) - a * Real.sin x + 1

theorem extreme_value_and_maximum (a : ℝ) :
  f' a (π / 6) = 0 →
  a = 4 ∧
  ∀ x ∈ Set.Icc (-π / 6) (7 * π / 6), f 4 x ≤ (5 * Real.sqrt 3) / 2 + π / 6 :=
by sorry

end extreme_value_and_maximum_l294_29444


namespace eating_time_theorem_l294_29401

/-- Represents the eating rate of a character in jars per minute -/
structure EatingRate :=
  (condensed_milk : ℚ)
  (honey : ℚ)

/-- Calculates the time taken to eat a certain amount of food given the eating rate -/
def time_to_eat (rate : EatingRate) (condensed_milk : ℚ) (honey : ℚ) : ℚ :=
  (condensed_milk / rate.condensed_milk) + (honey / rate.honey)

/-- Calculates the combined eating rate of two characters -/
def combined_rate (rate1 rate2 : EatingRate) : EatingRate :=
  { condensed_milk := rate1.condensed_milk + rate2.condensed_milk,
    honey := rate1.honey + rate2.honey }

theorem eating_time_theorem (pooh_rate piglet_rate : EatingRate) : 
  (time_to_eat pooh_rate 3 1 = 25) →
  (time_to_eat piglet_rate 3 1 = 55) →
  (time_to_eat pooh_rate 1 3 = 35) →
  (time_to_eat piglet_rate 1 3 = 85) →
  (time_to_eat (combined_rate pooh_rate piglet_rate) 6 0 = 20) := by
  sorry

end eating_time_theorem_l294_29401


namespace choose_one_friend_from_ten_l294_29433

def number_of_friends : ℕ := 10
def friends_to_choose : ℕ := 1

theorem choose_one_friend_from_ten :
  Nat.choose number_of_friends friends_to_choose = 10 := by
  sorry

end choose_one_friend_from_ten_l294_29433


namespace cube_sum_symmetric_polynomials_l294_29402

theorem cube_sum_symmetric_polynomials (x y z : ℝ) :
  let σ₁ : ℝ := x + y + z
  let σ₂ : ℝ := x * y + y * z + z * x
  let σ₃ : ℝ := x * y * z
  x^3 + y^3 + z^3 = σ₁^3 - 3 * σ₁ * σ₂ + 3 * σ₃ := by
  sorry

end cube_sum_symmetric_polynomials_l294_29402


namespace inscribed_triangle_relation_l294_29483

-- Define a triangle inscribed in a unit circle
structure InscribedTriangle where
  a : Real
  b : Real
  c : Real
  α : Real
  β : Real
  γ : Real
  sum_angles : α + β + γ = Real.pi
  side_a : a = 2 * Real.sin (α / 2)
  side_b : b = 2 * Real.sin (β / 2)
  side_c : c = 2 * Real.sin (γ / 2)

-- Theorem statement
theorem inscribed_triangle_relation (t : InscribedTriangle) :
  t.a^2 + t.b^2 + t.c^2 = 8 + 4 * Real.cos t.α * Real.cos t.β * Real.cos t.γ := by
  sorry

end inscribed_triangle_relation_l294_29483


namespace quadratic_coefficient_bound_l294_29443

theorem quadratic_coefficient_bound (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc (-1) 1) →
  |f 1| ≤ 1 →
  |f (1/2)| ≤ 1 →
  |f 0| ≤ 1 →
  |a| + |b| + |c| ≤ 17 := by
sorry

end quadratic_coefficient_bound_l294_29443


namespace k_range_for_two_distinct_roots_l294_29463

/-- A quadratic equation ax^2 + bx + c = 0 has two distinct real roots if and only if its discriminant is positive -/
axiom two_distinct_roots_iff_positive_discriminant (a b c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ b^2 - 4*a*c > 0

/-- The range of k for which kx^2 - 6x + 9 = 0 has two distinct real roots -/
theorem k_range_for_two_distinct_roots :
  ∀ k : ℝ, (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 6 * x + 9 = 0 ∧ k * y^2 - 6 * y + 9 = 0) ↔ k < 1 ∧ k ≠ 0 :=
by sorry

end k_range_for_two_distinct_roots_l294_29463


namespace max_f_value_l294_29470

theorem max_f_value (a b c d e f : ℝ) 
  (sum_condition : a + b + c + d + e + f = 10)
  (square_sum_condition : (a - 1)^2 + (b - 1)^2 + (c - 1)^2 + (d - 1)^2 + (e - 1)^2 + (f - 1)^2 = 6) :
  f ≤ 2 :=
by sorry

end max_f_value_l294_29470


namespace marble_ratio_l294_29490

/-- Proves the ratio of Michael's marbles to Wolfgang and Ludo's combined marbles -/
theorem marble_ratio :
  let wolfgang_marbles : ℕ := 16
  let ludo_marbles : ℕ := wolfgang_marbles + wolfgang_marbles / 4
  let total_marbles : ℕ := 20 * 3
  let michael_marbles : ℕ := total_marbles - wolfgang_marbles - ludo_marbles
  let wolfgang_ludo_marbles : ℕ := wolfgang_marbles + ludo_marbles
  (michael_marbles : ℚ) / wolfgang_ludo_marbles = 2 / 3 :=
by
  sorry


end marble_ratio_l294_29490


namespace wire_ratio_proof_l294_29416

theorem wire_ratio_proof (total_length shorter_length : ℝ) 
  (h1 : total_length = 35)
  (h2 : shorter_length = 10)
  (h3 : shorter_length < total_length) :
  shorter_length / (total_length - shorter_length) = 2 / 5 := by
sorry

end wire_ratio_proof_l294_29416


namespace average_value_of_series_l294_29466

theorem average_value_of_series (z : ℝ) :
  let series := [4*z, 6*z, 9*z, 13.5*z, 20.25*z]
  (series.sum / series.length : ℝ) = 10.55 * z :=
by sorry

end average_value_of_series_l294_29466


namespace medicine_percentage_l294_29442

/-- Proves that the percentage of income spent on medicines is 15% --/
theorem medicine_percentage (income : ℕ) (household_percent : ℚ) (clothes_percent : ℚ) (savings : ℕ)
  (h1 : income = 90000)
  (h2 : household_percent = 50 / 100)
  (h3 : clothes_percent = 25 / 100)
  (h4 : savings = 9000) :
  (income - (household_percent * income + clothes_percent * income + savings)) / income = 15 / 100 := by
  sorry

end medicine_percentage_l294_29442


namespace xy_value_l294_29455

theorem xy_value (x y : ℝ) (h : x * (x + 2*y) = x^2 + 10) : x * y = 5 := by
  sorry

end xy_value_l294_29455


namespace symmetric_point_yoz_plane_l294_29479

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The yOz plane in 3D space -/
def yOzPlane : Set Point3D := {p : Point3D | p.x = 0}

/-- Symmetry with respect to the yOz plane -/
def symmetricPointYOz (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

/-- Theorem: The point (-1, -2, 3) is symmetric to (1, -2, 3) with respect to the yOz plane -/
theorem symmetric_point_yoz_plane :
  let p1 : Point3D := { x := 1, y := -2, z := 3 }
  let p2 : Point3D := { x := -1, y := -2, z := 3 }
  symmetricPointYOz p1 = p2 := by
  sorry

end symmetric_point_yoz_plane_l294_29479


namespace no_primes_in_range_l294_29465

theorem no_primes_in_range (n : ℕ) (hn : n > 1) : 
  ∀ k ∈ Set.Ioo (n.factorial) (n.factorial + n + 1), ¬ Nat.Prime k := by
sorry

end no_primes_in_range_l294_29465


namespace max_product_constraint_l294_29432

theorem max_product_constraint (a b : ℝ) (h : a^2 + b^2 = 6) : a * b ≤ 3 := by
  sorry

end max_product_constraint_l294_29432


namespace quadratic_solution_product_l294_29451

theorem quadratic_solution_product (p q : ℝ) : 
  (3 * p^2 - 2 * p - 8 = 0) → 
  (3 * q^2 - 2 * q - 8 = 0) → 
  p ≠ q →
  (p - 1) * (q - 1) = -7/3 := by
sorry

end quadratic_solution_product_l294_29451


namespace least_value_of_quadratic_l294_29431

theorem least_value_of_quadratic (y : ℝ) : 
  (5 * y^2 + 7 * y + 3 = 5) → y ≥ -2 :=
by sorry

end least_value_of_quadratic_l294_29431


namespace pell_like_equation_solution_l294_29498

theorem pell_like_equation_solution (n : ℤ) :
  let x := (1/4) * ((1+Real.sqrt 2)^(2*n+1) + (1-Real.sqrt 2)^(2*n+1) - 2)
  let y := (1/(2*Real.sqrt 2)) * ((1+Real.sqrt 2)^(2*n+1) - (1-Real.sqrt 2)^(2*n+1))
  (x^2 + (x+1)^2 = y^2) ∧
  (∀ (a b : ℝ), a^2 + (a+1)^2 = b^2 → ∃ (m : ℤ), 
    a = (1/4) * ((1+Real.sqrt 2)^(2*m+1) + (1-Real.sqrt 2)^(2*m+1) - 2) ∧
    b = (1/(2*Real.sqrt 2)) * ((1+Real.sqrt 2)^(2*m+1) - (1-Real.sqrt 2)^(2*m+1)))
  := by sorry

end pell_like_equation_solution_l294_29498


namespace product_of_roots_l294_29456

theorem product_of_roots (x : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 15*x^2 + 75*x - 50 = (x - r₁) * (x - r₂) * (x - r₃)) → 
  ∃ r₁ r₂ r₃ : ℝ, x^3 - 15*x^2 + 75*x - 50 = (x - r₁) * (x - r₂) * (x - r₃) ∧ r₁ * r₂ * r₃ = 50 :=
by sorry

end product_of_roots_l294_29456


namespace olivia_weekly_earnings_l294_29414

/-- Calculates the total earnings for a week given an hourly rate and hours worked on specific days -/
def weeklyEarnings (hourlyRate : ℕ) (mondayHours wednesdayHours fridayHours : ℕ) : ℕ :=
  hourlyRate * (mondayHours + wednesdayHours + fridayHours)

/-- Proves that Olivia's earnings for the week equal $117 -/
theorem olivia_weekly_earnings :
  weeklyEarnings 9 4 3 6 = 117 := by
  sorry

end olivia_weekly_earnings_l294_29414


namespace strawberry_weight_sum_l294_29448

/-- The total weight of Marco's and his dad's strawberries is 40 pounds. -/
theorem strawberry_weight_sum : 
  let marco_weight : ℕ := 8
  let dad_weight : ℕ := 32
  marco_weight + dad_weight = 40 := by sorry

end strawberry_weight_sum_l294_29448


namespace value_of_T_l294_29499

theorem value_of_T : ∃ T : ℚ, (1/2 : ℚ) * (1/7 : ℚ) * T = (1/3 : ℚ) * (1/5 : ℚ) * 90 ∧ T = 84 := by
  sorry

end value_of_T_l294_29499


namespace x_power_n_plus_inverse_l294_29429

theorem x_power_n_plus_inverse (θ : ℝ) (x : ℂ) (n : ℕ) 
  (h1 : 0 < θ) (h2 : θ < π / 2) (h3 : x + 1 / x = -2 * Real.sin θ) (h4 : n > 0) :
  x^n + 1 / x^n = -2 * Real.sin (n * θ) := by
  sorry

end x_power_n_plus_inverse_l294_29429


namespace article_original_price_l294_29419

/-- Given an article with a 25% profit margin, where the profit is 775 rupees, 
    prove that the original price of the article is 3100 rupees. -/
theorem article_original_price (profit_percentage : ℝ) (profit : ℝ) (original_price : ℝ) : 
  profit_percentage = 0.25 →
  profit = 775 →
  profit = profit_percentage * original_price →
  original_price = 3100 :=
by
  sorry

end article_original_price_l294_29419


namespace female_percentage_l294_29409

/-- Represents a classroom with double desks -/
structure Classroom where
  male_students : ℕ
  female_students : ℕ
  male_with_male : ℕ
  female_with_female : ℕ

/-- All seats are occupied and the percentages of same-gender pairings are as given -/
def valid_classroom (c : Classroom) : Prop :=
  c.male_with_male = (6 * c.male_students) / 10 ∧
  c.female_with_female = (2 * c.female_students) / 10 ∧
  c.male_students - c.male_with_male = c.female_students - c.female_with_female

theorem female_percentage (c : Classroom) (h : valid_classroom c) :
  (c.female_students : ℚ) / (c.male_students + c.female_students) = 1/3 := by
  sorry

end female_percentage_l294_29409


namespace tuesday_temperature_l294_29417

/-- Given the average temperatures for different sets of days and the temperature on Friday,
    prove the temperature on Tuesday. -/
theorem tuesday_temperature
  (avg_tues_wed_thurs : (t + w + th) / 3 = 52)
  (avg_wed_thurs_fri : (w + th + 53) / 3 = 54)
  (fri_temp : ℝ)
  (h_fri_temp : fri_temp = 53) :
  t = 47 :=
by sorry


end tuesday_temperature_l294_29417


namespace farmer_tomatoes_l294_29459

/-- Proves that if a farmer has 479 tomatoes and picks 364 of them, he will have 115 tomatoes left. -/
theorem farmer_tomatoes (initial : ℕ) (picked : ℕ) (remaining : ℕ) : 
  initial = 479 → picked = 364 → remaining = initial - picked → remaining = 115 :=
by sorry

end farmer_tomatoes_l294_29459


namespace merchant_profit_percentage_l294_29475

theorem merchant_profit_percentage (C S : ℝ) (h : 24 * C = 16 * S) : 
  (S - C) / C * 100 = 50 :=
sorry

end merchant_profit_percentage_l294_29475


namespace min_lcm_ac_is_90_l294_29437

def min_lcm_ac (a b c : ℕ) : Prop :=
  (Nat.lcm a b = 20) ∧ (Nat.lcm b c = 18) → Nat.lcm a c ≥ 90

theorem min_lcm_ac_is_90 :
  ∃ (a b c : ℕ), min_lcm_ac a b c ∧ Nat.lcm a c = 90 :=
sorry

end min_lcm_ac_is_90_l294_29437


namespace face_ratio_is_four_thirds_l294_29458

/-- A polyhedron with triangular and square faces -/
structure Polyhedron where
  triangular_faces : ℕ
  square_faces : ℕ
  no_shared_square_edges : Bool
  no_shared_triangle_edges : Bool

/-- The ratio of triangular faces to square faces in a polyhedron -/
def face_ratio (p : Polyhedron) : ℚ :=
  p.triangular_faces / p.square_faces

/-- Theorem: The ratio of triangular faces to square faces is 4:3 -/
theorem face_ratio_is_four_thirds (p : Polyhedron) 
  (h1 : p.no_shared_square_edges = true) 
  (h2 : p.no_shared_triangle_edges = true) : 
  face_ratio p = 4 / 3 := by
  sorry

end face_ratio_is_four_thirds_l294_29458


namespace max_k_value_l294_29460

theorem max_k_value (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : k * a * b * c / (a + b + c) ≤ (a + b)^2 + (a + b + 4*c)^2) : 
  k ≤ 100 :=
sorry

end max_k_value_l294_29460


namespace representatives_count_l294_29421

/-- The number of ways to select 3 representatives from 3 boys and 2 girls, 
    such that at least one girl is included. -/
def select_representatives (num_boys num_girls num_representatives : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of ways to select 3 representatives from 3 boys and 2 girls, 
    such that at least one girl is included, is equal to 9. -/
theorem representatives_count : select_representatives 3 2 3 = 9 := by
  sorry

end representatives_count_l294_29421


namespace sqrt_equality_implies_t_value_l294_29406

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, 
    (Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4)) → 
    t = 37/10 := by
sorry

end sqrt_equality_implies_t_value_l294_29406


namespace g_at_negative_two_l294_29467

-- Define the function g
def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

-- State the theorem
theorem g_at_negative_two : g (-2) = 7/3 := by sorry

end g_at_negative_two_l294_29467


namespace root_product_plus_one_l294_29452

theorem root_product_plus_one (r s t : ℂ) : 
  r^3 - 15*r^2 + 26*r - 8 = 0 →
  s^3 - 15*s^2 + 26*s - 8 = 0 →
  t^3 - 15*t^2 + 26*t - 8 = 0 →
  (1 + r) * (1 + s) * (1 + t) = 50 := by
sorry

end root_product_plus_one_l294_29452


namespace five_mondays_in_november_l294_29473

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Given that a month starts on a certain day, 
    returns the number of occurrences of a specific day in that month -/
def countDayOccurrences (month : Month) (day : DayOfWeek) : Nat :=
  sorry

/-- October of year M -/
def october : Month :=
  { days := 31, firstDay := sorry }

/-- November of year M -/
def november : Month :=
  { days := 30, firstDay := sorry }

theorem five_mondays_in_november 
  (h : countDayOccurrences october DayOfWeek.Friday = 5) :
  countDayOccurrences november DayOfWeek.Monday = 5 := by
  sorry

end five_mondays_in_november_l294_29473


namespace restaurant_group_children_l294_29482

/-- The number of children in a restaurant group -/
def num_children (num_adults : ℕ) (meal_cost : ℕ) (total_bill : ℕ) : ℕ :=
  (total_bill - num_adults * meal_cost) / meal_cost

theorem restaurant_group_children :
  num_children 2 3 21 = 5 := by
  sorry

end restaurant_group_children_l294_29482


namespace expand_and_simplify_l294_29427

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end expand_and_simplify_l294_29427


namespace circle_area_probability_l294_29464

theorem circle_area_probability (AB : ℝ) (h_AB : AB = 10) : 
  let prob := (Real.sqrt 64 - Real.sqrt 36) / AB
  prob = 1 / 5 := by sorry

end circle_area_probability_l294_29464


namespace inequality_proof_l294_29415

theorem inequality_proof (x y z : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < π/2) : 
  π/2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end inequality_proof_l294_29415


namespace brick_width_is_four_l294_29441

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: A brick with length 8 cm, height 2 cm, and surface area 112 cm² has a width of 4 cm -/
theorem brick_width_is_four :
  ∀ w : ℝ, surface_area 8 w 2 = 112 → w = 4 := by
  sorry

end brick_width_is_four_l294_29441


namespace arcsin_plus_arccos_eq_pi_half_l294_29430

theorem arcsin_plus_arccos_eq_pi_half (x : ℝ) :
  Real.arcsin x + Real.arccos (1 - x) = π / 2 → x = Real.sqrt 2 / 2 := by
  sorry

end arcsin_plus_arccos_eq_pi_half_l294_29430


namespace max_f_value_1997_l294_29412

def f : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => f (n / 2) + (n + 2) - 2 * (n / 2)

theorem max_f_value_1997 :
  (∃ (n : ℕ), n ≤ 1997 ∧ f n = 10) ∧
  (∀ (n : ℕ), n ≤ 1997 → f n ≤ 10) :=
sorry

end max_f_value_1997_l294_29412


namespace fraction_simplification_l294_29461

theorem fraction_simplification (a b : ℝ) (h1 : b ≠ 0) (h2 : a ≠ b) :
  (1/2 * a) / (1/2 * b) = a / b := by
  sorry

end fraction_simplification_l294_29461


namespace quadratic_inequality_solution_set_l294_29491

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo 1 2 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | b * x^2 + a * x + c < 0} = Set.Ioo (-2/3) 1 :=
sorry

end quadratic_inequality_solution_set_l294_29491


namespace fraction_calculation_l294_29449

theorem fraction_calculation : (16 : ℚ) / 42 * 18 / 27 - 4 / 21 = 4 / 63 := by
  sorry

end fraction_calculation_l294_29449


namespace triangle_side_lengths_l294_29403

theorem triangle_side_lengths (x : ℕ+) : 
  (9 + 12 > x^2 ∧ x^2 + 9 > 12 ∧ x^2 + 12 > 9) ↔ (x = 2 ∨ x = 3 ∨ x = 4) :=
sorry

end triangle_side_lengths_l294_29403


namespace f_extrema_l294_29496

noncomputable def f (x : ℝ) : ℝ := 1 + 3*x - x^3

theorem f_extrema :
  (∃ x : ℝ, f x = -1 ∧ ∀ y : ℝ, f y ≥ -1) ∧
  (∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≤ 3) :=
by sorry

end f_extrema_l294_29496


namespace product_real_implies_a_equals_two_l294_29471

theorem product_real_implies_a_equals_two (a : ℝ) : 
  let z₁ : ℂ := 2 + Complex.I
  let z₂ : ℂ := a - Complex.I
  (z₁ * z₂).im = 0 → a = 2 := by
sorry

end product_real_implies_a_equals_two_l294_29471


namespace salary_difference_l294_29493

theorem salary_difference (a b : ℝ) (h : b = 1.25 * a) :
  (b - a) / b * 100 = 20 := by sorry

end salary_difference_l294_29493


namespace income_comparison_l294_29436

/-- Given that Mart's income is 30% more than Tim's income and 78% of Juan's income,
    prove that Tim's income is 40% less than Juan's income. -/
theorem income_comparison (tim mart juan : ℝ) 
  (h1 : mart = 1.3 * tim) 
  (h2 : mart = 0.78 * juan) : 
  tim = 0.6 * juan := by
  sorry

end income_comparison_l294_29436


namespace gift_contribution_l294_29495

theorem gift_contribution (a b c d e : ℝ) : 
  a + b + c + d + e = 120 →
  a = 2 * b →
  b = (1/3) * (c + d) →
  c = 2 * e →
  e = 12 := by sorry

end gift_contribution_l294_29495


namespace swamp_ecosystem_l294_29480

/-- In a swamp ecosystem, prove that each gharial needs to eat 15 fish per day given the following conditions:
  * Each frog eats 30 flies per day
  * Each fish eats 8 frogs per day
  * There are 9 gharials in the swamp
  * 32,400 flies are eaten every day
-/
theorem swamp_ecosystem (flies_per_frog : ℕ) (frogs_per_fish : ℕ) (num_gharials : ℕ) (total_flies : ℕ)
  (h1 : flies_per_frog = 30)
  (h2 : frogs_per_fish = 8)
  (h3 : num_gharials = 9)
  (h4 : total_flies = 32400) :
  total_flies / (flies_per_frog * frogs_per_fish * num_gharials) = 15 := by
  sorry

end swamp_ecosystem_l294_29480


namespace nicky_running_time_l294_29447

/-- The time Nicky runs before Cristina catches up to him in a race with given conditions -/
theorem nicky_running_time (race_length : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_length = 500)
  (h2 : cristina_speed > nicky_speed)
  (h3 : head_start = 12)
  (h4 : cristina_speed = 5)
  (h5 : nicky_speed = 3) :
  head_start + (head_start * nicky_speed) / (cristina_speed - nicky_speed) = 30 := by
  sorry

#check nicky_running_time

end nicky_running_time_l294_29447


namespace geese_flew_away_l294_29494

/-- Proves that the number of geese that flew away is equal to the difference
    between the initial number of geese and the number of geese left in the field. -/
theorem geese_flew_away (initial : ℕ) (left : ℕ) (flew_away : ℕ)
    (h1 : initial = 51)
    (h2 : left = 23)
    (h3 : initial ≥ left) :
  flew_away = initial - left :=
by sorry

end geese_flew_away_l294_29494


namespace arrangements_for_six_people_l294_29422

/-- The number of people in the line -/
def n : ℕ := 6

/-- The number of arrangements of n people in a line where two specific people
    must stand next to each other and two other specific people must not stand
    next to each other -/
def arrangements (n : ℕ) : ℕ := 
  2 * (n - 2).factorial * ((n - 2) * (n - 3))

/-- Theorem stating that the number of arrangements for 6 people
    under the given conditions is 144 -/
theorem arrangements_for_six_people : arrangements n = 144 := by
  sorry

end arrangements_for_six_people_l294_29422


namespace compare_b_and_d_l294_29474

theorem compare_b_and_d (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a = b * 1.02)
  (hac : c = a * 0.99)
  (hcd : d = c * 0.99) : 
  b > d := by
sorry

end compare_b_and_d_l294_29474


namespace probability_ratio_l294_29497

def total_slips : ℕ := 50
def numbers_range : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def probability_same_number (total : ℕ) (range : ℕ) (per_num : ℕ) (drawn : ℕ) : ℚ :=
  (range : ℚ) / Nat.choose total drawn

def probability_three_and_two (total : ℕ) (range : ℕ) (per_num : ℕ) (drawn : ℕ) : ℚ :=
  (Nat.choose range 2 * Nat.choose per_num 3 * Nat.choose per_num 2 : ℚ) / Nat.choose total drawn

theorem probability_ratio :
  (probability_three_and_two total_slips numbers_range slips_per_number drawn_slips) /
  (probability_same_number total_slips numbers_range slips_per_number drawn_slips) = 450 := by
  sorry

end probability_ratio_l294_29497


namespace geometric_sum_7_terms_l294_29435

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_7_terms :
  let a : ℚ := 1/2
  let r : ℚ := -1/3
  let n : ℕ := 7
  geometric_sum a r n = 547/1458 := by sorry

end geometric_sum_7_terms_l294_29435


namespace anna_age_proof_l294_29407

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

end anna_age_proof_l294_29407


namespace simplify_square_roots_l294_29410

theorem simplify_square_roots : 
  (Real.sqrt 448 / Real.sqrt 128) + (Real.sqrt 98 / Real.sqrt 49) = 
  (Real.sqrt 14 + 2 * Real.sqrt 2) / 2 := by
  sorry

end simplify_square_roots_l294_29410


namespace total_rooms_is_260_l294_29440

/-- Represents the hotel booking scenario -/
structure HotelBooking where
  singleRooms : ℕ
  doubleRooms : ℕ
  singleRoomCost : ℕ
  doubleRoomCost : ℕ
  totalIncome : ℕ

/-- Calculates the total number of rooms booked -/
def totalRooms (booking : HotelBooking) : ℕ :=
  booking.singleRooms + booking.doubleRooms

/-- Theorem stating that the total number of rooms booked is 260 -/
theorem total_rooms_is_260 (booking : HotelBooking) 
  (h1 : booking.singleRooms = 64)
  (h2 : booking.singleRoomCost = 35)
  (h3 : booking.doubleRoomCost = 60)
  (h4 : booking.totalIncome = 14000) :
  totalRooms booking = 260 := by
  sorry

#eval totalRooms { singleRooms := 64, doubleRooms := 196, singleRoomCost := 35, doubleRoomCost := 60, totalIncome := 14000 }

end total_rooms_is_260_l294_29440


namespace quadratic_equations_solutions_l294_29425

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -1 + Real.sqrt 6 ∧ x₂ = -1 - Real.sqrt 6 ∧ 
    x₁^2 + 2*x₁ = 5 ∧ x₂^2 + 2*x₂ = 5) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧ 
    x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -5/2 ∧ x₂ = 1 ∧ 
    2*x₁^2 + 3*x₁ - 5 = 0 ∧ 2*x₂^2 + 3*x₂ - 5 = 0) :=
by sorry

end quadratic_equations_solutions_l294_29425


namespace sum_of_dimensions_l294_29481

/-- A rectangular box with dimensions A, B, and C, where AB = 40, AC = 90, and BC = 360 -/
structure RectangularBox where
  A : ℝ
  B : ℝ
  C : ℝ
  ab_area : A * B = 40
  ac_area : A * C = 90
  bc_area : B * C = 360

/-- The sum of dimensions A, B, and C of the rectangular box is 45 -/
theorem sum_of_dimensions (box : RectangularBox) : box.A + box.B + box.C = 45 := by
  sorry

end sum_of_dimensions_l294_29481


namespace rotation_result_l294_29428

/-- Given a point A(3, -4) rotated counterclockwise by π/2 around the origin,
    the resulting point B has a y-coordinate of 3. -/
theorem rotation_result : ∃ (B : ℝ × ℝ), 
  let A : ℝ × ℝ := (3, -4)
  let angle : ℝ := π / 2
  B.1 = A.1 * Real.cos angle - A.2 * Real.sin angle ∧
  B.2 = A.1 * Real.sin angle + A.2 * Real.cos angle ∧
  B.2 = 3 := by
  sorry

end rotation_result_l294_29428


namespace quadratic_equation_properties_l294_29484

theorem quadratic_equation_properties : ∃ (x y : ℝ),
  x^2 + 1984513*x + 3154891 = 0 ∧
  y^2 + 1984513*y + 3154891 = 0 ∧
  x ≠ y ∧
  (∀ z : ℤ, z^2 + 1984513*z + 3154891 ≠ 0) ∧
  x ≤ 0 ∧
  y ≤ 0 ∧
  1/x + 1/y ≥ -1 :=
by sorry

end quadratic_equation_properties_l294_29484


namespace solution_implies_value_l294_29420

theorem solution_implies_value (a b : ℝ) : 
  (-a * 3 - b = 5 - 2 * 3) → (3 - 6 * a - 2 * b = 1) := by
  sorry

end solution_implies_value_l294_29420


namespace line_plane_parallelism_l294_29472

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallelism relation between planes
variable (parallel_plane : Plane → Plane → Prop)

-- Define the intersection relation between lines
variable (intersect : Line → Line → Prop)

-- Define the relation for a line being outside a plane
variable (outside : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_parallelism 
  (m n : Line) (α β : Plane) :
  intersect m n ∧ 
  outside m α ∧ outside m β ∧ 
  outside n α ∧ outside n β ∧
  parallel_line_plane m α ∧ parallel_line_plane m β ∧ 
  parallel_line_plane n α ∧ parallel_line_plane n β →
  parallel_plane α β :=
sorry

end line_plane_parallelism_l294_29472


namespace exists_unsolvable_grid_l294_29476

/-- Represents a 9x9 grid with values either 1 or -1 -/
def Grid := Fin 9 → Fin 9 → Int

/-- Defines a valid grid where all values are either 1 or -1 -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j, g i j = 1 ∨ g i j = -1

/-- Defines the neighbors of a cell in the grid -/
def neighbors (i j : Fin 9) : List (Fin 9 × Fin 9) :=
  [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]

/-- Computes the new value of a cell after a move -/
def move (g : Grid) (i j : Fin 9) : Int :=
  (neighbors i j).foldl (λ acc (ni, nj) => acc * g ni nj) 1

/-- Applies a move to the entire grid -/
def apply_move (g : Grid) : Grid :=
  λ i j => move g i j

/-- Checks if all cells in the grid are 1 -/
def all_ones (g : Grid) : Prop :=
  ∀ i j, g i j = 1

/-- The main theorem: there exists a valid grid that cannot be transformed to all ones -/
theorem exists_unsolvable_grid :
  ∃ (g : Grid), valid_grid g ∧ 
    ∀ (n : ℕ), ¬(all_ones ((apply_move^[n]) g)) :=
  sorry

end exists_unsolvable_grid_l294_29476


namespace total_interest_received_l294_29468

/-- Simple interest calculation function -/
def simple_interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time / 100

theorem total_interest_received (loan_b_principal loan_c_principal : ℕ) 
  (loan_b_time loan_c_time : ℕ) (interest_rate : ℚ) : 
  loan_b_principal = 5000 →
  loan_c_principal = 3000 →
  loan_b_time = 2 →
  loan_c_time = 4 →
  interest_rate = 15 →
  simple_interest loan_b_principal interest_rate loan_b_time + 
  simple_interest loan_c_principal interest_rate loan_c_time = 3300 := by
sorry

end total_interest_received_l294_29468


namespace triangle_angle_c_l294_29426

theorem triangle_angle_c (A B C : Real) (perimeter area : Real) :
  perimeter = Real.sqrt 2 + 1 →
  Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C →
  area = (1 / 6) * Real.sin C →
  C = π / 3 := by
  sorry

end triangle_angle_c_l294_29426


namespace cards_given_to_miguel_l294_29454

/-- Represents the card distribution problem --/
def card_distribution (total_cards : ℕ) (kept_cards : ℕ) (friends : ℕ) (cards_per_friend : ℕ) 
  (sisters : ℕ) (cards_per_sister : ℕ) : ℕ :=
  let remaining_after_keeping := total_cards - kept_cards
  let given_to_friends := friends * cards_per_friend
  let remaining_after_friends := remaining_after_keeping - given_to_friends
  let given_to_sisters := sisters * cards_per_sister
  remaining_after_friends - given_to_sisters

/-- Theorem stating the number of cards Rick gave to Miguel --/
theorem cards_given_to_miguel : 
  card_distribution 250 25 12 15 4 7 = 17 := by
  sorry


end cards_given_to_miguel_l294_29454


namespace imaginary_part_of_complex_fraction_l294_29489

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (1 - Complex.I) / (1 + 3 * Complex.I)
  Complex.im z = -2/5 := by
sorry

end imaginary_part_of_complex_fraction_l294_29489


namespace tangent_line_properties_l294_29445

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 1

-- Define the tangent line condition
def tangent_condition (x₁ : ℝ) (a : ℝ) : Prop :=
  ∃ x₂ : ℝ, f' x₁ * (x₂ - x₁) + f x₁ = g a x₂ ∧ f' x₁ = 2 * x₂

-- State the theorem
theorem tangent_line_properties :
  (∀ x₁ a : ℝ, tangent_condition x₁ a → (x₁ = -1 → a = 3)) ∧
  (∀ a : ℝ, (∃ x₁ : ℝ, tangent_condition x₁ a) → a ≥ -1) :=
sorry

end tangent_line_properties_l294_29445


namespace subset_iff_range_l294_29400

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x - 4 ≤ 0}

-- State the theorem
theorem subset_iff_range (a : ℝ) : B a ⊆ A ↔ 0 ≤ a ∧ a < 3 :=
sorry

end subset_iff_range_l294_29400


namespace parabola_vertex_l294_29438

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 18

/-- The x-coordinate of the vertex -/
def p : ℝ := -2

/-- The y-coordinate of the vertex -/
def q : ℝ := 10

/-- Theorem: The vertex of the parabola y = 2x^2 + 8x + 18 is at (-2, 10) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola p) ∧ parabola p = q := by sorry

end parabola_vertex_l294_29438


namespace e₁_e₂_divisibility_l294_29457

def e₁ (a : ℕ) : ℕ := a^2 + 3^a + a * 3^((a + 1) / 2)
def e₂ (a : ℕ) : ℕ := a^2 + 3^a - a * 3^((a + 1) / 2)

theorem e₁_e₂_divisibility (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 25) :
  (e₁ a * e₂ a) % 3 = 0 ↔ a % 3 = 0 :=
sorry

end e₁_e₂_divisibility_l294_29457


namespace complex_sum_eighth_power_l294_29469

open Complex

theorem complex_sum_eighth_power :
  ((-1 + I) / 2) ^ 8 + ((-1 - I) / 2) ^ 8 = (1 : ℂ) / 8 := by sorry

end complex_sum_eighth_power_l294_29469

import Mathlib

namespace book_club_picks_l1828_182854

theorem book_club_picks (total_members : ℕ) (meeting_weeks : ℕ) (guest_picks : ℕ) :
  total_members = 13 →
  meeting_weeks = 48 →
  guest_picks = 12 →
  (meeting_weeks - guest_picks) / total_members = 2 :=
by sorry

end book_club_picks_l1828_182854


namespace inequality_proof_l1828_182889

theorem inequality_proof (a b : ℝ) (h : a > b) : -3 * a < -3 * b := by
  sorry

end inequality_proof_l1828_182889


namespace sum_of_cubes_l1828_182899

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : 
  a^3 + b^3 = 26 := by
sorry

end sum_of_cubes_l1828_182899


namespace min_circle_area_l1828_182855

/-- Given a line ax + by = 1 passing through point A(b, a), where O is the origin (0, 0),
    the minimum area of the circle with center O and radius OA is π. -/
theorem min_circle_area (a b : ℝ) (h : a * b = 1 / 2) :
  (π : ℝ) ≤ π * (a^2 + b^2) ∧ ∃ (a₀ b₀ : ℝ), a₀ * b₀ = 1 / 2 ∧ π * (a₀^2 + b₀^2) = π :=
by sorry

end min_circle_area_l1828_182855


namespace negation_of_proposition_l1828_182876

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 0 → x^3 / (x - 2) > 0)) ↔ (∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x ≤ 2) :=
by sorry

end negation_of_proposition_l1828_182876


namespace product_remainder_mod_17_l1828_182882

theorem product_remainder_mod_17 : (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 2 := by
  sorry

end product_remainder_mod_17_l1828_182882


namespace some_students_not_club_members_l1828_182878

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (ClubMember : U → Prop)
variable (Dishonest : U → Prop)

-- Define the conditions
variable (some_students_dishonest : ∃ x, Student x ∧ Dishonest x)
variable (all_club_members_honest : ∀ x, ClubMember x → ¬Dishonest x)

-- Theorem to prove
theorem some_students_not_club_members :
  ∃ x, Student x ∧ ¬ClubMember x :=
sorry

end some_students_not_club_members_l1828_182878


namespace radical_conjugate_sum_product_l1828_182824

/-- Given that c + √d and its radical conjugate have a sum of 0 and a product of 9, prove that c + d = -9 -/
theorem radical_conjugate_sum_product (c d : ℝ) : 
  ((c + Real.sqrt d) + (c - Real.sqrt d) = 0) ∧ 
  ((c + Real.sqrt d) * (c - Real.sqrt d) = 9) → 
  c + d = -9 := by sorry

end radical_conjugate_sum_product_l1828_182824


namespace triangle_inequality_l1828_182892

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_l1828_182892


namespace quadratic_root_conditions_l1828_182831

theorem quadratic_root_conditions (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 + (a^2 - 1)*x + a - 2 = 0 ∧ 
    y^2 + (a^2 - 1)*y + a - 2 = 0 ∧ 
    x > 1 ∧ y < 1) ↔ 
  -2 < a ∧ a < 1 := by sorry

end quadratic_root_conditions_l1828_182831


namespace inequality_solution_range_l1828_182897

theorem inequality_solution_range (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - m * x + (m - 1) ≥ 0) → m ≥ 2 * Real.sqrt 3 / 3 := by
  sorry

end inequality_solution_range_l1828_182897


namespace arianna_sleep_hours_l1828_182875

/-- Represents the number of hours in a day. -/
def hours_in_day : ℕ := 24

/-- Represents the number of hours Arianna spends at work. -/
def work_hours : ℕ := 6

/-- Represents the number of hours Arianna spends in class. -/
def class_hours : ℕ := 3

/-- Represents the number of hours Arianna spends at the gym. -/
def gym_hours : ℕ := 2

/-- Represents the number of hours Arianna spends on other daily chores. -/
def chore_hours : ℕ := 5

/-- Represents the number of hours Arianna sleeps. -/
def sleep_hours : ℕ := hours_in_day - (work_hours + class_hours + gym_hours + chore_hours)

/-- Theorem stating that Arianna sleeps for 8 hours a day. -/
theorem arianna_sleep_hours : sleep_hours = 8 := by
  sorry

end arianna_sleep_hours_l1828_182875


namespace factorization_equality_l1828_182801

theorem factorization_equality (a b : ℝ) : a * b^2 - a = a * (b + 1) * (b - 1) := by
  sorry

end factorization_equality_l1828_182801


namespace tomato_plants_l1828_182870

theorem tomato_plants (first_plant : ℕ) : 
  (∃ (second_plant third_plant fourth_plant : ℕ),
    second_plant = first_plant + 4 ∧
    third_plant = 3 * (first_plant + second_plant) ∧
    fourth_plant = 3 * (first_plant + second_plant) ∧
    first_plant + second_plant + third_plant + fourth_plant = 140) →
  first_plant = 8 := by
  sorry

end tomato_plants_l1828_182870


namespace one_true_proposition_l1828_182815

-- Define the basic concepts
def Point : Type := ℝ × ℝ
def Triangle (A B C : Point) : Prop := True  -- Simplified definition
def Isosceles (A B C : Point) : Prop := True  -- Simplified definition

-- Define the original proposition
def original_prop (A B C : Point) : Prop :=
  A.1 = B.1 ∧ A.2 = B.2 → Isosceles A B C

-- Define the converse proposition
def converse_prop (A B C : Point) : Prop :=
  Isosceles A B C → A.1 = B.1 ∧ A.2 = B.2

-- Define the inverse proposition
def inverse_prop (A B C : Point) : Prop :=
  ¬(A.1 = B.1 ∧ A.2 = B.2) → ¬(Isosceles A B C)

-- Define the contrapositive proposition
def contrapositive_prop (A B C : Point) : Prop :=
  ¬(Isosceles A B C) → ¬(A.1 = B.1 ∧ A.2 = B.2)

-- The theorem to be proved
theorem one_true_proposition (A B C : Point) :
  (original_prop A B C) ∧
  (¬(converse_prop A B C) ∨ ¬(inverse_prop A B C)) ∧
  (contrapositive_prop A B C) :=
sorry

end one_true_proposition_l1828_182815


namespace square_difference_l1828_182853

theorem square_difference (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : b^2 - a^2 = -15 := by
  sorry

end square_difference_l1828_182853


namespace benny_total_spent_l1828_182864

def soft_drink_quantity : ℕ := 2
def soft_drink_price : ℕ := 4
def candy_bar_quantity : ℕ := 5
def candy_bar_price : ℕ := 4

theorem benny_total_spent :
  soft_drink_quantity * soft_drink_price + candy_bar_quantity * candy_bar_price = 28 := by
  sorry

end benny_total_spent_l1828_182864


namespace right_angled_complex_roots_l1828_182888

open Complex

theorem right_angled_complex_roots (a b : ℂ) (z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 → 
  z₂^2 + a*z₂ + b = 0 → 
  z₁ ≠ 0 → 
  z₂ ≠ 0 → 
  z₁ ≠ z₂ → 
  (z₁.re * z₂.re + z₁.im * z₂.im = 0) → 
  a^2 / b = 2 := by
sorry

end right_angled_complex_roots_l1828_182888


namespace complex_equation_solution_count_l1828_182842

theorem complex_equation_solution_count : 
  ∃! (c : ℝ), Complex.abs (2/3 - c * Complex.I) = 5/6 ∧ Complex.im (3 + c * Complex.I) > 0 :=
by sorry

end complex_equation_solution_count_l1828_182842


namespace probability_closer_to_point1_l1828_182840

/-- The rectangular region from which point P is selected -/
def Rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- The area of the rectangular region -/
def RectangleArea : ℝ := 6

/-- The point (1,1) -/
def Point1 : ℝ × ℝ := (1, 1)

/-- The point (4,1) -/
def Point2 : ℝ × ℝ := (4, 1)

/-- The region where points are closer to (1,1) than to (4,1) -/
def CloserRegion : Set (ℝ × ℝ) :=
  {p ∈ Rectangle | dist p Point1 < dist p Point2}

/-- The area of the region closer to (1,1) -/
def CloserRegionArea : ℝ := 5

/-- The probability of a randomly selected point being closer to (1,1) than to (4,1) -/
theorem probability_closer_to_point1 :
  CloserRegionArea / RectangleArea = 5 / 6 :=
sorry

end probability_closer_to_point1_l1828_182840


namespace second_candidate_votes_l1828_182883

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℚ) :
  total_votes = 1200 →
  first_candidate_percentage = 60 / 100 →
  (total_votes : ℚ) * (1 - first_candidate_percentage) = 480 :=
by sorry

end second_candidate_votes_l1828_182883


namespace brazilian_coffee_price_l1828_182860

/-- Proves that the price of Brazilian coffee is $3.75 per pound given the conditions of the coffee mix problem. -/
theorem brazilian_coffee_price
  (total_mix : ℝ)
  (columbian_price : ℝ)
  (final_mix_price : ℝ)
  (columbian_amount : ℝ)
  (h_total_mix : total_mix = 100)
  (h_columbian_price : columbian_price = 8.75)
  (h_final_mix_price : final_mix_price = 6.35)
  (h_columbian_amount : columbian_amount = 52) :
  let brazilian_amount : ℝ := total_mix - columbian_amount
  let brazilian_price : ℝ := (total_mix * final_mix_price - columbian_amount * columbian_price) / brazilian_amount
  brazilian_price = 3.75 := by
sorry


end brazilian_coffee_price_l1828_182860


namespace line_ratio_sum_l1828_182848

/-- Given two lines l₁ and l₂, and points P₁ and P₂ on these lines respectively,
    prove that the sum of certain ratios of the line coefficients equals 3. -/
theorem line_ratio_sum (a₁ b₁ c₁ a₂ b₂ c₂ x₁ y₁ x₂ y₂ : ℝ) : 
  a₁ * x₁ + b₁ * y₁ = c₁ →
  a₂ * x₂ + b₂ * y₂ = c₂ →
  a₁ + b₁ = c₁ →
  a₂ + b₂ = 2 * c₂ →
  (∀ x₁ y₁ x₂ y₂, (x₁ - x₂)^2 + (y₁ - y₂)^2 ≥ 1/2) →
  c₁ / a₁ + a₂ / c₂ = 3 := by
sorry

end line_ratio_sum_l1828_182848


namespace probability_of_strong_l1828_182823

def word_train : Finset Char := {'T', 'R', 'A', 'I', 'N'}
def word_shield : Finset Char := {'S', 'H', 'I', 'E', 'L', 'D'}
def word_grow : Finset Char := {'G', 'R', 'O', 'W'}
def word_strong : Finset Char := {'S', 'T', 'R', 'O', 'N', 'G'}

def prob_train : ℚ := 1 / (word_train.card.choose 3)
def prob_shield : ℚ := 3 / (word_shield.card.choose 4)
def prob_grow : ℚ := 1 / (word_grow.card.choose 2)

theorem probability_of_strong :
  prob_train * prob_shield * prob_grow = 1 / 300 :=
sorry

end probability_of_strong_l1828_182823


namespace five_candies_three_kids_l1828_182858

/-- The number of ways to distribute n candies among k kids, with each kid getting at least one candy -/
def distribute_candies (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 6 ways to distribute 5 candies among 3 kids, with each kid getting at least one candy -/
theorem five_candies_three_kids : distribute_candies 5 3 = 6 := by
  sorry

end five_candies_three_kids_l1828_182858


namespace ball_drawing_theorem_l1828_182844

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat

/-- Represents the random variable ξ (number of yellow balls drawn) -/
def Xi := Nat

/-- The initial state of the box -/
def initialBox : BallCounts := { red := 1, green := 1, yellow := 2 }

/-- The probability of drawing no yellow balls before drawing the red ball -/
def probXiZero (box : BallCounts) : Real :=
  sorry

/-- The expected value of ξ -/
def expectedXi (box : BallCounts) : Real :=
  sorry

/-- The main theorem stating the probability and expectation results -/
theorem ball_drawing_theorem (box : BallCounts) 
  (h : box = initialBox) : 
  probXiZero box = 1/3 ∧ expectedXi box = 1 := by
  sorry

end ball_drawing_theorem_l1828_182844


namespace sqrt_sin_identity_l1828_182850

theorem sqrt_sin_identity : Real.sqrt (1 - Real.sin 2) + Real.sqrt (1 + Real.sin 2) = 2 * Real.sin 1 := by
  sorry

end sqrt_sin_identity_l1828_182850


namespace sams_price_per_sheet_l1828_182895

/-- Represents the pricing structure of a photo company -/
structure PhotoCompany where
  price_per_sheet : ℝ
  sitting_fee : ℝ

/-- Calculates the total cost for a given number of sheets -/
def total_cost (company : PhotoCompany) (sheets : ℝ) : ℝ :=
  company.price_per_sheet * sheets + company.sitting_fee

/-- Proves that Sam's Picture Emporium charges $1.50 per sheet -/
theorem sams_price_per_sheet :
  let johns := PhotoCompany.mk 2.75 125
  let sams := PhotoCompany.mk x 140
  total_cost johns 12 = total_cost sams 12 →
  x = 1.50 := by
  sorry

end sams_price_per_sheet_l1828_182895


namespace even_quadratic_function_l1828_182813

/-- A quadratic function f(x) = ax^2 + (2a^2 - a)x + 1 is even if and only if a = 1/2 -/
theorem even_quadratic_function (a : ℝ) :
  (∀ x, (a * x^2 + (2 * a^2 - a) * x + 1) = (a * (-x)^2 + (2 * a^2 - a) * (-x) + 1)) ↔
  a = 1/2 := by
sorry

end even_quadratic_function_l1828_182813


namespace certain_number_problem_l1828_182816

theorem certain_number_problem (x : ℝ) : 
  3 + x + 333 + 33.3 = 399.6 → x = 30.3 := by
sorry

end certain_number_problem_l1828_182816


namespace rectangle_area_problem_l1828_182896

theorem rectangle_area_problem (a b : ℝ) :
  (∀ (a b : ℝ), 
    ((a + 3) * b - a * b = 12) ∧
    ((a + 3) * (b + 3) - (a + 3) * b = 24)) →
  a * b = 20 :=
by sorry

end rectangle_area_problem_l1828_182896


namespace parallel_vectors_xy_value_l1828_182884

/-- Given two parallel vectors a and b in R³, prove that xy = -1/4 --/
theorem parallel_vectors_xy_value (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (2*x, 1, 3)
  let b : ℝ × ℝ × ℝ := (1, -2*y, 9)
  (∃ (k : ℝ), a = k • b) → x * y = -1/4 := by
  sorry

end parallel_vectors_xy_value_l1828_182884


namespace levels_ratio_l1828_182843

def total_levels : ℕ := 32
def beaten_levels : ℕ := 24

theorem levels_ratio :
  let not_beaten := total_levels - beaten_levels
  (beaten_levels : ℚ) / not_beaten = 3 / 1 := by sorry

end levels_ratio_l1828_182843


namespace parabola_c_value_l1828_182893

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_c_value (p : Parabola) :
  p.y_at 1 = 3 →  -- vertex at (1, 3)
  p.y_at 0 = 2 →  -- passes through (0, 2)
  p.c = 2 := by
sorry


end parabola_c_value_l1828_182893


namespace unique_solution_conditions_l1828_182863

/-- The system has a unique solution if and only if a = arctan(4) + πk or a = -arctan(2) + πk, where k is an integer -/
theorem unique_solution_conditions (a : ℝ) : 
  (∃! x y : ℝ, x * Real.cos a + y * Real.sin a = 5 * Real.cos a + 2 * Real.sin a ∧ 
                -3 ≤ x + 2*y ∧ x + 2*y ≤ 7 ∧ 
                -9 ≤ 3*x - 4*y ∧ 3*x - 4*y ≤ 1) ↔ 
  (∃ k : ℤ, a = Real.arctan 4 + k * Real.pi ∨ a = -Real.arctan 2 + k * Real.pi) :=
by sorry

end unique_solution_conditions_l1828_182863


namespace square_area_error_l1828_182807

theorem square_area_error (a : ℝ) (h : a > 0) :
  let measured_side := a * (1 + 0.08)
  let actual_area := a^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.1664 := by
sorry

end square_area_error_l1828_182807


namespace fraction_simplification_l1828_182868

theorem fraction_simplification (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 := by
  sorry

end fraction_simplification_l1828_182868


namespace wheel_circumference_proof_l1828_182821

/-- The circumference of the front wheel -/
def front_wheel_circumference : ℝ := 24

/-- The circumference of the rear wheel -/
def rear_wheel_circumference : ℝ := 18

/-- The distance traveled -/
def distance : ℝ := 360

theorem wheel_circumference_proof :
  (distance / front_wheel_circumference = distance / rear_wheel_circumference + 4) ∧
  (distance / (front_wheel_circumference - 3) = distance / (rear_wheel_circumference - 3) + 6) →
  (front_wheel_circumference = 24 ∧ rear_wheel_circumference = 18) :=
by sorry

end wheel_circumference_proof_l1828_182821


namespace expand_expression_l1828_182881

theorem expand_expression (x : ℝ) : 25 * (3 * x - 4) = 75 * x - 100 := by
  sorry

end expand_expression_l1828_182881


namespace arbitrarily_large_power_l1828_182856

theorem arbitrarily_large_power (a : ℝ) (h : a > 1) :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, a^x > y :=
by sorry

end arbitrarily_large_power_l1828_182856


namespace symmetry_axis_of_sine_function_l1828_182836

/-- Given that cos(2π/3 - φ) = cosφ, prove that x = 5π/6 is a symmetry axis of f(x) = sin(x - φ) -/
theorem symmetry_axis_of_sine_function (φ : ℝ) 
  (h : Real.cos (2 * Real.pi / 3 - φ) = Real.cos φ) :
  ∀ x : ℝ, Real.sin (x - φ) = Real.sin ((5 * Real.pi / 3 - x) - φ) :=
by sorry

end symmetry_axis_of_sine_function_l1828_182836


namespace inequality_proof_binomial_inequality_l1828_182898

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a / Real.sqrt b + b / Real.sqrt a > Real.sqrt a + Real.sqrt b :=
by sorry

theorem binomial_inequality (x : ℝ) (m : ℕ) (hx : x > -1) (hm : m > 0) :
  (1 + x)^m ≥ 1 + m * x :=
by sorry

end inequality_proof_binomial_inequality_l1828_182898


namespace mobile_phone_purchase_price_l1828_182828

/-- The purchase price of the refrigerator in rupees -/
def refrigerator_price : ℝ := 15000

/-- The loss percentage on the refrigerator sale -/
def refrigerator_loss_percent : ℝ := 3

/-- The profit percentage on the mobile phone sale -/
def mobile_profit_percent : ℝ := 10

/-- The overall profit in rupees -/
def overall_profit : ℝ := 350

/-- The purchase price of the mobile phone in rupees -/
def mobile_price : ℝ := 8000

theorem mobile_phone_purchase_price :
  ∃ (x : ℝ),
    x = mobile_price ∧
    refrigerator_price * (1 - refrigerator_loss_percent / 100) +
    x * (1 + mobile_profit_percent / 100) =
    refrigerator_price + x + overall_profit :=
by sorry

end mobile_phone_purchase_price_l1828_182828


namespace exponential_equation_solution_l1828_182839

theorem exponential_equation_solution : ∃ x : ℝ, (9 : ℝ)^x * (9 : ℝ)^x * (9 : ℝ)^x = (27 : ℝ)^4 ∧ x = 2 := by
  sorry

end exponential_equation_solution_l1828_182839


namespace quadratic_function_property_l1828_182872

-- Define a quadratic function with integer coefficients
def QuadraticFunction (a b c : ℤ) : ℤ → ℤ := fun x ↦ a * x^2 + b * x + c

-- Define the set of possible values for f(0), f(3), and f(4)
def PossibleValues : Set ℤ := {2, 20, 202, 2022}

-- Theorem statement
theorem quadratic_function_property (a b c : ℤ) :
  let f := QuadraticFunction a b c
  (f 0 ∈ PossibleValues) ∧
  (f 3 ∈ PossibleValues) ∧
  (f 4 ∈ PossibleValues) ∧
  (f 0 ≠ f 3) ∧
  (f 0 ≠ f 4) ∧
  (f 3 ≠ f 4) →
  (f 1 = -80) ∨ (f 1 = -990) := by
sorry

end quadratic_function_property_l1828_182872


namespace limes_given_correct_l1828_182865

/-- The number of limes Dan initially picked -/
def initial_limes : ℕ := 9

/-- The number of limes Dan has now -/
def current_limes : ℕ := 5

/-- The number of limes Dan gave to Sara -/
def limes_given : ℕ := initial_limes - current_limes

theorem limes_given_correct : limes_given = 4 := by sorry

end limes_given_correct_l1828_182865


namespace madeline_leisure_hours_l1828_182818

def total_hours_in_week : ℕ := 24 * 7

def class_hours : ℕ := 18
def homework_hours : ℕ := 4 * 7
def extracurricular_hours : ℕ := 3 * 3
def tutoring_hours : ℕ := 1 * 2
def work_hours : ℕ := 5 + 4 + 4 + 7
def sleep_hours : ℕ := 8 * 7

def total_scheduled_hours : ℕ := 
  class_hours + homework_hours + extracurricular_hours + tutoring_hours + work_hours + sleep_hours

theorem madeline_leisure_hours : 
  total_hours_in_week - total_scheduled_hours = 35 := by sorry

end madeline_leisure_hours_l1828_182818


namespace train_length_l1828_182805

/-- Given a train that crosses a bridge and passes a lamp post, calculate its length. -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ)
  (h1 : bridge_length = 2500)
  (h2 : bridge_time = 120)
  (h3 : post_time = 30) :
  bridge_length * (post_time / bridge_time) / (1 - post_time / bridge_time) = 2500 * (1/4) / (1 - 1/4) :=
by sorry

end train_length_l1828_182805


namespace gym_floor_area_per_person_l1828_182891

theorem gym_floor_area_per_person :
  ∀ (base height : ℝ) (num_students : ℕ),
    base = 9 →
    height = 8 →
    num_students = 24 →
    (base * height) / num_students = 3 := by
  sorry

end gym_floor_area_per_person_l1828_182891


namespace lecture_room_seating_l1828_182877

theorem lecture_room_seating (m n : ℕ) : 
  (∃ boys_per_row girls_per_column unoccupied : ℕ,
    boys_per_row = 6 ∧ 
    girls_per_column = 8 ∧ 
    unoccupied = 15 ∧
    m * n = boys_per_row * m + girls_per_column * n + unoccupied) →
  (m - 8) * (n - 6) = 63 :=
by sorry

end lecture_room_seating_l1828_182877


namespace trapezoid_side_length_l1828_182819

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  base1 : ℝ
  base2 : ℝ

/-- Theorem: In a trapezoid with given properties, the length of the other side is 10 -/
theorem trapezoid_side_length (t : Trapezoid) 
  (h_area : t.area = 164)
  (h_altitude : t.altitude = 8)
  (h_base1 : t.base1 = 10)
  (h_base2 : t.base2 = 17) :
  t.base2 - t.base1 = 10 := by
  sorry

#check trapezoid_side_length

end trapezoid_side_length_l1828_182819


namespace circle_rooted_polynomial_ab_neq_nine_l1828_182830

/-- A polynomial of degree 4 with four distinct roots on a circle in the complex plane -/
structure CircleRootedPolynomial where
  a : ℂ
  b : ℂ
  roots_distinct : True  -- Placeholder for the distinctness condition
  roots_on_circle : True -- Placeholder for the circle condition

/-- The theorem stating that for a polynomial with four distinct roots on a circle, ab ≠ 9 -/
theorem circle_rooted_polynomial_ab_neq_nine (P : CircleRootedPolynomial) : P.a * P.b ≠ 9 := by
  sorry

end circle_rooted_polynomial_ab_neq_nine_l1828_182830


namespace aquarium_purchase_cost_l1828_182829

/-- Calculates the total cost of an aquarium purchase with given discounts and tax rates -/
theorem aquarium_purchase_cost 
  (original_price : ℝ)
  (aquarium_discount : ℝ)
  (coupon_discount : ℝ)
  (additional_items_cost : ℝ)
  (aquarium_tax_rate : ℝ)
  (other_items_tax_rate : ℝ)
  (h1 : original_price = 120)
  (h2 : aquarium_discount = 0.5)
  (h3 : coupon_discount = 0.1)
  (h4 : additional_items_cost = 75)
  (h5 : aquarium_tax_rate = 0.05)
  (h6 : other_items_tax_rate = 0.08) :
  let discounted_price := original_price * (1 - aquarium_discount)
  let final_aquarium_price := discounted_price * (1 - coupon_discount)
  let aquarium_tax := final_aquarium_price * aquarium_tax_rate
  let other_items_tax := additional_items_cost * other_items_tax_rate
  let total_cost := final_aquarium_price + aquarium_tax + additional_items_cost + other_items_tax
  total_cost = 137.70 := by
sorry


end aquarium_purchase_cost_l1828_182829


namespace equation_solution_l1828_182800

theorem equation_solution :
  ∀ t : ℂ, (2 / (t + 3) + 3 * t / (t + 3) - 5 / (t + 3) = t + 2) ↔ 
  (t = -1 + 2 * Complex.I * Real.sqrt 2 ∨ t = -1 - 2 * Complex.I * Real.sqrt 2) :=
by sorry

end equation_solution_l1828_182800


namespace polynomial_identity_l1828_182867

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end polynomial_identity_l1828_182867


namespace sin_squared_plus_cos_squared_equals_one_l1828_182841

-- Define a point on a unit circle
def PointOnUnitCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the relationship between x, y, and θ on the unit circle
def UnitCirclePoint (θ : ℝ) (x y : ℝ) : Prop :=
  x = Real.cos θ ∧ y = Real.sin θ

-- Theorem statement
theorem sin_squared_plus_cos_squared_equals_one (θ : ℝ) :
  ∃ x y : ℝ, UnitCirclePoint θ x y → (Real.sin θ)^2 + (Real.cos θ)^2 = 1 := by
  sorry

end sin_squared_plus_cos_squared_equals_one_l1828_182841


namespace difference_of_squares_l1828_182886

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end difference_of_squares_l1828_182886


namespace chloe_david_distance_difference_l1828_182838

-- Define the speeds and time
def chloe_speed : ℝ := 18
def david_speed : ℝ := 15
def bike_time : ℝ := 5

-- Define the theorem
theorem chloe_david_distance_difference :
  chloe_speed * bike_time - david_speed * bike_time = 15 := by
  sorry

end chloe_david_distance_difference_l1828_182838


namespace max_profit_price_l1828_182834

/-- Represents the profit function for a store selling items -/
def profit_function (purchase_price : ℝ) (base_price : ℝ) (base_quantity : ℝ) (price_sensitivity : ℝ) (x : ℝ) : ℝ :=
  (x - purchase_price) * (base_quantity - price_sensitivity * (x - base_price))

theorem max_profit_price (purchase_price : ℝ) (base_price : ℝ) (base_quantity : ℝ) (price_sensitivity : ℝ) 
    (h1 : purchase_price = 20)
    (h2 : base_price = 30)
    (h3 : base_quantity = 400)
    (h4 : price_sensitivity = 20) : 
  ∃ (max_price : ℝ), max_price = 35 ∧ 
    ∀ (x : ℝ), profit_function purchase_price base_price base_quantity price_sensitivity x ≤ 
               profit_function purchase_price base_price base_quantity price_sensitivity max_price :=
by sorry

#check max_profit_price

end max_profit_price_l1828_182834


namespace modulus_of_z_l1828_182847

theorem modulus_of_z (z : ℂ) (h : (z + Complex.I) * (1 + Complex.I) = 1 - Complex.I) :
  Complex.abs z = 2 := by
  sorry

end modulus_of_z_l1828_182847


namespace ellipse_foci_coordinates_l1828_182835

/-- The coordinates of the foci of an ellipse given by the equation mx^2 + ny^2 + mn = 0,
    where m < n < 0 -/
theorem ellipse_foci_coordinates (m n : ℝ) (h1 : m < n) (h2 : n < 0) :
  let equation := fun (x y : ℝ) => m * x^2 + n * y^2 + m * n
  ∃ c : ℝ, c > 0 ∧ 
    (∀ x y : ℝ, equation x y = 0 → 
      ((x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)) ↔ 
      (x, y) ∈ {p : ℝ × ℝ | p.1^2 / (-n) + p.2^2 / (-m) = 1 ∧ p.1^2 + p.2^2 > 1}) ∧
    c^2 = n - m :=
by sorry

end ellipse_foci_coordinates_l1828_182835


namespace expansion_without_x_squared_l1828_182890

theorem expansion_without_x_squared (n : ℕ+) (h1 : 5 ≤ n) (h2 : n ≤ 8) :
  (∀ (r : ℕ), r ≤ n → n - 4 * r ≠ 0 ∧ n - 4 * r ≠ 1 ∧ n - 4 * r ≠ 2) ↔ n = 7 := by
  sorry

end expansion_without_x_squared_l1828_182890


namespace percentage_increase_l1828_182803

theorem percentage_increase (original_earnings new_earnings : ℝ) :
  original_earnings = 60 →
  new_earnings = 68 →
  (new_earnings - original_earnings) / original_earnings * 100 = (68 - 60) / 60 * 100 := by
sorry

end percentage_increase_l1828_182803


namespace smallest_multiple_of_5_and_711_l1828_182822

theorem smallest_multiple_of_5_and_711 :
  ∀ n : ℕ, n > 0 ∧ 5 ∣ n ∧ 711 ∣ n → n ≥ 3555 :=
by
  sorry

end smallest_multiple_of_5_and_711_l1828_182822


namespace rose_flyers_count_l1828_182862

def total_flyers : ℕ := 1236
def jack_flyers : ℕ := 120
def left_flyers : ℕ := 796

theorem rose_flyers_count : total_flyers - jack_flyers - left_flyers = 320 := by
  sorry

end rose_flyers_count_l1828_182862


namespace subtracted_value_proof_l1828_182806

theorem subtracted_value_proof (n : ℕ) (x : ℕ) : 
  n = 36 → 
  ((n + 10) * 2) / 2 - x = 88 / 2 ↔ 
  x = 2 := by
sorry

end subtracted_value_proof_l1828_182806


namespace percent_of_a_is_4b_l1828_182857

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.2 * b) : (4 * b) / a = 10/3 := by
  sorry

end percent_of_a_is_4b_l1828_182857


namespace geometric_sequence_first_term_l1828_182825

/-- Given a geometric sequence where the fifth term is 48 and the sixth term is 72,
    the first term of the sequence is 768/81. -/
theorem geometric_sequence_first_term :
  ∀ (a r : ℚ),
    a * r^4 = 48 →
    a * r^5 = 72 →
    a = 768/81 := by
  sorry

end geometric_sequence_first_term_l1828_182825


namespace arithmetic_sequence_property_l1828_182808

/-- An arithmetic sequence with the given properties has the general term a_n = n. -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) : 
  d ≠ 0 ∧ 
  (∀ n, a (n + 1) = a n + d) ∧ 
  a 2 ^ 2 = a 1 * a 4 ∧ 
  a 5 + a 6 = 11 → 
  ∀ n, a n = n := by
  sorry

end arithmetic_sequence_property_l1828_182808


namespace ring_toss_earnings_l1828_182832

/-- The ring toss game earnings problem -/
theorem ring_toss_earnings (total_earnings : ℕ) (num_days : ℕ) (daily_earnings : ℕ) : 
  total_earnings = 165 → num_days = 5 → total_earnings = num_days * daily_earnings → daily_earnings = 33 := by
  sorry

end ring_toss_earnings_l1828_182832


namespace unique_first_degree_polynomial_l1828_182827

/-- The polynomial p(x) = 2x + 1 -/
def p (x : ℝ) : ℝ := 2 * x + 1

/-- The polynomial q(x) = x -/
def q (x : ℝ) : ℝ := x

theorem unique_first_degree_polynomial :
  ∀ (x : ℝ), p (p (q x)) = q (p (p x)) ∧
  ∀ (r : ℝ → ℝ), (∃ (a b : ℝ), ∀ (x : ℝ), r x = a * x + b) →
  (∀ (x : ℝ), p (p (r x)) = r (p (p x))) →
  r = q :=
sorry

end unique_first_degree_polynomial_l1828_182827


namespace min_value_expression_l1828_182804

theorem min_value_expression (x y z : ℝ) (h : x - 2*y + 2*z = 5) :
  ∃ (min : ℝ), min = 36 ∧ ∀ (x' y' z' : ℝ), x' - 2*y' + 2*z' = 5 → 
    (x' + 5)^2 + (y' - 1)^2 + (z' + 3)^2 ≥ min :=
sorry

end min_value_expression_l1828_182804


namespace complex_imaginary_part_l1828_182880

theorem complex_imaginary_part (a : ℝ) (z : ℂ) : 
  z = 1 + a * I →  -- z is of the form 1 + ai
  a > 0 →  -- z is in the first quadrant
  Complex.abs z = Real.sqrt 5 →  -- |z| = √5
  z.im = 2 :=  -- The imaginary part of z is 2
by sorry

end complex_imaginary_part_l1828_182880


namespace sugar_consumption_reduction_l1828_182859

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h1 : initial_price = 6)
  (h2 : new_price = 7.50) :
  let price_increase_ratio := new_price / initial_price
  let consumption_reduction_percentage := (1 - 1 / price_increase_ratio) * 100
  consumption_reduction_percentage = 25 := by sorry

end sugar_consumption_reduction_l1828_182859


namespace leadership_structure_count_15_l1828_182810

/-- The number of ways to select a leadership structure from a group of people. -/
def leadershipStructureCount (n : ℕ) : ℕ :=
  n * (n - 1).choose 2 * (n - 3).choose 3 * (n - 6).choose 3

/-- Theorem stating that the number of ways to select a leadership structure
    from 15 people is 2,717,880. -/
theorem leadership_structure_count_15 :
  leadershipStructureCount 15 = 2717880 := by
  sorry

end leadership_structure_count_15_l1828_182810


namespace birthday_celebration_attendance_l1828_182826

/-- The number of people who stayed at a birthday celebration --/
def people_stayed (total_guests : ℕ) (men : ℕ) (children_left : ℕ) : ℕ :=
  let women := total_guests / 2
  let children := total_guests - women - men
  let men_left := men / 3
  total_guests - men_left - children_left

/-- Theorem about the number of people who stayed at the birthday celebration --/
theorem birthday_celebration_attendance :
  people_stayed 60 15 5 = 50 :=
by
  sorry

end birthday_celebration_attendance_l1828_182826


namespace complement_of_A_l1828_182879

-- Define the set A
def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 ≥ 0}

-- State the theorem
theorem complement_of_A :
  (Set.univ : Set ℝ) \ A = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end complement_of_A_l1828_182879


namespace mode_of_throws_l1828_182811

def throw_results : List Float := [7.6, 8.5, 8.6, 8.5, 9.1, 8.5, 8.4, 8.6, 9.2, 7.3]

def mode (l : List Float) : Float :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_throws :
  mode throw_results = 8.5 := by
  sorry

end mode_of_throws_l1828_182811


namespace complex_sum_and_product_l1828_182809

theorem complex_sum_and_product : ∃ (z₁ z₂ : ℂ),
  z₁ = 2 + 5*I ∧ z₂ = 3 - 7*I ∧ z₁ + z₂ = 5 - 2*I ∧ z₁ * z₂ = -29 + I :=
by
  sorry

end complex_sum_and_product_l1828_182809


namespace min_n_for_cuboid_sum_l1828_182861

theorem min_n_for_cuboid_sum (n : ℕ) : (∀ m : ℕ, m > 0 ∧ 128 * m > 2011 → n ≤ m) ∧ n > 0 ∧ 128 * n > 2011 ↔ n = 16 := by
  sorry

end min_n_for_cuboid_sum_l1828_182861


namespace new_year_markup_percentage_l1828_182837

/-- Proves that given specific markups and profit, the New Year season markup is 25% -/
theorem new_year_markup_percentage
  (initial_markup : ℝ)
  (february_discount : ℝ)
  (final_profit : ℝ)
  (h1 : initial_markup = 0.20)
  (h2 : february_discount = 0.10)
  (h3 : final_profit = 0.35)
  : ∃ (new_year_markup : ℝ),
    (1 + initial_markup) * (1 + new_year_markup) * (1 - february_discount) = 1 + final_profit ∧
    new_year_markup = 0.25 :=
sorry

end new_year_markup_percentage_l1828_182837


namespace artwork_transaction_l1828_182887

/-- Converts a number from base s to base 10 -/
def to_base_10 (digits : List Nat) (s : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * s^i) 0

theorem artwork_transaction (s : Nat) : 
  s > 1 →
  to_base_10 [0, 3, 5] s + to_base_10 [0, 3, 2, 1] s = to_base_10 [0, 0, 0, 2] s →
  s = 8 := by
sorry

end artwork_transaction_l1828_182887


namespace negative_two_cubed_l1828_182814

theorem negative_two_cubed : (-2 : ℤ)^3 = -8 := by
  sorry

end negative_two_cubed_l1828_182814


namespace range_of_a_l1828_182852

/-- The equation x^2 + 2ax + 1 = 0 has two real roots greater than -1 -/
def p (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > -1 ∧ x₂ > -1 ∧ x₁^2 + 2*a*x₁ + 1 = 0 ∧ x₂^2 + 2*a*x₂ + 1 = 0

/-- The solution set to the inequality ax^2 - ax + 1 > 0 is ℝ -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem range_of_a :
  (∀ a : ℝ, p a ∨ q a) →
  (∀ a : ℝ, ¬q a) →
  {a : ℝ | a ≤ -1} = {a : ℝ | p a} :=
sorry

end range_of_a_l1828_182852


namespace modulus_of_complex_fraction_l1828_182820

theorem modulus_of_complex_fraction : 
  let z : ℂ := (2 * Complex.I) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end modulus_of_complex_fraction_l1828_182820


namespace video_recorder_markup_l1828_182812

theorem video_recorder_markup (wholesale_cost : ℝ) (employee_discount : ℝ) (employee_paid : ℝ) :
  wholesale_cost = 200 →
  employee_discount = 0.30 →
  employee_paid = 168 →
  ∃ (markup : ℝ), 
    employee_paid = (1 - employee_discount) * (wholesale_cost * (1 + markup)) ∧
    markup = 0.20 := by
  sorry

end video_recorder_markup_l1828_182812


namespace total_cakes_is_fifteen_l1828_182817

/-- The number of cakes served during lunch -/
def lunch_cakes : ℕ := 6

/-- The number of cakes served during dinner -/
def dinner_cakes : ℕ := 9

/-- The total number of cakes served today -/
def total_cakes : ℕ := lunch_cakes + dinner_cakes

/-- Proof that the total number of cakes served today is 15 -/
theorem total_cakes_is_fifteen : total_cakes = 15 := by
  sorry

end total_cakes_is_fifteen_l1828_182817


namespace square_remainder_l1828_182833

theorem square_remainder (n x : ℤ) (h : n % x = 3) : (n^2) % x = 9 % x := by
  sorry

end square_remainder_l1828_182833


namespace area_quadrilateral_OBEC_l1828_182885

/-- A line with slope -3 passing through (3,6) -/
def line1 (x y : ℝ) : Prop := y - 6 = -3 * (x - 3)

/-- The x-coordinate of point A where line1 intersects the x-axis -/
def point_A : ℝ := 5

/-- The y-coordinate of point B where line1 intersects the y-axis -/
def point_B : ℝ := 15

/-- A line passing through points (6,0) and (3,6) -/
def line2 (x y : ℝ) : Prop := y = 2 * x - 12

/-- The area of quadrilateral OBEC -/
def area_OBEC : ℝ := 72

theorem area_quadrilateral_OBEC :
  line1 3 6 →
  line1 point_A 0 →
  line1 0 point_B →
  line2 3 6 →
  line2 6 0 →
  area_OBEC = 72 := by
  sorry

end area_quadrilateral_OBEC_l1828_182885


namespace mystery_book_shelves_l1828_182851

/-- Proves that the number of mystery book shelves is 8 --/
theorem mystery_book_shelves :
  let books_per_shelf : ℕ := 7
  let picture_book_shelves : ℕ := 2
  let total_books : ℕ := 70
  let mystery_book_shelves : ℕ := (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf
  mystery_book_shelves = 8 := by
sorry

end mystery_book_shelves_l1828_182851


namespace equal_money_time_l1828_182866

/-- 
Proves that Carol and Mike will have the same amount of money after 5 weeks,
given their initial amounts and weekly savings rates.
-/
theorem equal_money_time (carol_initial : ℕ) (mike_initial : ℕ) 
  (carol_weekly : ℕ) (mike_weekly : ℕ) :
  carol_initial = 60 →
  mike_initial = 90 →
  carol_weekly = 9 →
  mike_weekly = 3 →
  ∃ w : ℕ, w = 5 ∧ carol_initial + w * carol_weekly = mike_initial + w * mike_weekly :=
by
  sorry

#check equal_money_time

end equal_money_time_l1828_182866


namespace cookie_circle_properties_l1828_182869

/-- Given a circle described by the equation x^2 + y^2 + 10 = 6x + 12y,
    this theorem proves its radius, circumference, and area. -/
theorem cookie_circle_properties :
  let equation := fun (x y : ℝ) => x^2 + y^2 + 10 = 6*x + 12*y
  ∃ (center : ℝ × ℝ) (r : ℝ),
    (∀ x y, equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = r^2) ∧
    r = Real.sqrt 35 ∧
    2 * Real.pi * r = 2 * Real.pi * Real.sqrt 35 ∧
    Real.pi * r^2 = 35 * Real.pi :=
by sorry

end cookie_circle_properties_l1828_182869


namespace different_parrot_extra_toes_l1828_182802

/-- Represents the nail trimming scenario for Cassie's pets -/
structure PetNails where
  num_dogs : Nat
  num_parrots : Nat
  dog_nails_per_foot : Nat
  dog_feet : Nat
  parrot_claws_per_leg : Nat
  parrot_legs : Nat
  total_nails_to_cut : Nat

/-- Calculates the number of extra toes on the different parrot -/
def extra_toes (p : PetNails) : Nat :=
  let standard_dog_nails := p.num_dogs * p.dog_nails_per_foot * p.dog_feet
  let standard_parrot_claws := (p.num_parrots - 1) * p.parrot_claws_per_leg * p.parrot_legs
  let standard_nails := standard_dog_nails + standard_parrot_claws
  p.total_nails_to_cut - standard_nails - (p.parrot_claws_per_leg * p.parrot_legs)

/-- Theorem stating that the number of extra toes on the different parrot is 7 -/
theorem different_parrot_extra_toes :
  ∃ (p : PetNails), 
    p.num_dogs = 4 ∧ 
    p.num_parrots = 8 ∧ 
    p.dog_nails_per_foot = 4 ∧ 
    p.dog_feet = 4 ∧ 
    p.parrot_claws_per_leg = 3 ∧ 
    p.parrot_legs = 2 ∧ 
    p.total_nails_to_cut = 113 ∧ 
    extra_toes p = 7 := by
  sorry

end different_parrot_extra_toes_l1828_182802


namespace horse_catches_dog_l1828_182846

/-- Represents the relative speed and step distance of animals -/
structure AnimalData where
  steps_per_time_unit : ℕ
  distance_per_steps : ℕ

/-- Calculates the distance an animal covers in one time unit -/
def speed (a : AnimalData) : ℕ := a.steps_per_time_unit * a.distance_per_steps

theorem horse_catches_dog (dog : AnimalData) (horse : AnimalData) 
  (h1 : dog.steps_per_time_unit = 5)
  (h2 : horse.steps_per_time_unit = 3)
  (h3 : 4 * horse.distance_per_steps = 7 * dog.distance_per_steps)
  (initial_distance : ℕ)
  (h4 : initial_distance = 30) :
  (speed horse - speed dog) * 600 = initial_distance * (speed horse) :=
sorry

end horse_catches_dog_l1828_182846


namespace triangle_acute_from_angle_ratio_l1828_182873

/-- Theorem: In a triangle ABC where the ratio of angles A:B:C is 2:3:4, all angles are less than 90 degrees. -/
theorem triangle_acute_from_angle_ratio (A B C : ℝ) (h_ratio : ∃ (x : ℝ), A = 2*x ∧ B = 3*x ∧ C = 4*x) 
  (h_sum : A + B + C = 180) : A < 90 ∧ B < 90 ∧ C < 90 := by
  sorry

end triangle_acute_from_angle_ratio_l1828_182873


namespace angle_bisector_length_l1828_182871

/-- Given a triangle PQR with side lengths PQ and PR, and the cosine of angle P,
    calculate the length of the angle bisector PS. -/
theorem angle_bisector_length (PQ PR : ℝ) (cos_P : ℝ) (h_PQ : PQ = 4) (h_PR : PR = 8) (h_cos_P : cos_P = 1/9) :
  ∃ (PS : ℝ), PS = Real.sqrt ((43280 - 128 * Real.sqrt 41) / 81) :=
sorry

end angle_bisector_length_l1828_182871


namespace david_presents_l1828_182849

theorem david_presents (christmas_presents : ℕ) (birthday_presents : ℕ) : 
  christmas_presents = 2 * birthday_presents →
  christmas_presents = 60 →
  christmas_presents + birthday_presents = 90 := by
sorry

end david_presents_l1828_182849


namespace kite_only_always_perpendicular_diagonals_l1828_182894

-- Define the types of quadrilaterals
inductive Quadrilateral
  | Rhombus
  | Rectangle
  | Square
  | Kite
  | IsoscelesTrapezoid

-- Define a property for perpendicular diagonals
def has_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.Kite => true
  | _ => false

-- Theorem statement
theorem kite_only_always_perpendicular_diagonals :
  ∀ q : Quadrilateral, has_perpendicular_diagonals q ↔ q = Quadrilateral.Kite :=
by sorry

end kite_only_always_perpendicular_diagonals_l1828_182894


namespace red_balls_count_l1828_182874

theorem red_balls_count (total_balls : ℕ) (prob_red : ℚ) : 
  total_balls = 20 → prob_red = 1/4 → (prob_red * total_balls : ℚ) = 5 := by
  sorry

end red_balls_count_l1828_182874


namespace complex_equation_solution_l1828_182845

theorem complex_equation_solution :
  ∃ x : ℂ, (5 : ℂ) - 3 * Complex.I * x = (7 : ℂ) - Complex.I * x ∧ x = -Complex.I := by
  sorry

end complex_equation_solution_l1828_182845

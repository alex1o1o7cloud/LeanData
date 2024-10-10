import Mathlib

namespace geometric_sequence_minimum_value_l1664_166496

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a7 : a 7 = Real.sqrt 2 / 2) :
  (1 / a 3 + 2 / a 11) ≥ 4 ∧
  ∃ x : ℝ, (1 / a 3 + 2 / a 11) = x → x ≥ 4 :=
sorry

end geometric_sequence_minimum_value_l1664_166496


namespace least_number_divisibility_l1664_166405

theorem least_number_divisibility (n : ℕ) (h1 : (n + 6) % 24 = 0) (h2 : (n + 6) % 32 = 0)
  (h3 : (n + 6) % 36 = 0) (h4 : n + 6 = 858) :
  ∃ p : ℕ, Nat.Prime p ∧ p ≠ 2 ∧ p ≠ 3 ∧ n % p = 0 := by
  sorry

end least_number_divisibility_l1664_166405


namespace greg_adam_marble_difference_l1664_166407

/-- Given that Adam has 29 marbles, Greg has 43 marbles, and Greg has more marbles than Adam,
    prove that Greg has 14 more marbles than Adam. -/
theorem greg_adam_marble_difference :
  ∀ (adam_marbles greg_marbles : ℕ),
    adam_marbles = 29 →
    greg_marbles = 43 →
    greg_marbles > adam_marbles →
    greg_marbles - adam_marbles = 14 := by
  sorry

end greg_adam_marble_difference_l1664_166407


namespace no_solution_exists_l1664_166467

theorem no_solution_exists : ¬ ∃ (x : ℕ), 
  (18 + x = 2 * (5 + x)) ∧ 
  (18 + x = 3 * (2 + x)) ∧ 
  ((18 + x) + (5 + x) + (2 + x) = 50) := by
  sorry

end no_solution_exists_l1664_166467


namespace probability_vowel_in_mathematics_l1664_166482

def english_alphabet : Finset Char := sorry

def vowels : Finset Char := {'a', 'e', 'i', 'o', 'u'}

def mathematics : List Char := ['m', 'a', 't', 'h', 'e', 'm', 'a', 't', 'i', 'c', 's']

theorem probability_vowel_in_mathematics :
  (Finset.filter (fun c => c ∈ vowels) mathematics.toFinset).card / mathematics.length = 4 / 11 :=
by sorry

end probability_vowel_in_mathematics_l1664_166482


namespace farm_animals_l1664_166429

theorem farm_animals (total_legs : ℕ) (total_animals : ℕ) (duck_legs : ℕ) (cow_legs : ℕ) :
  total_legs = 42 →
  total_animals = 15 →
  duck_legs = 2 →
  cow_legs = 4 →
  ∃ (num_ducks : ℕ) (num_cows : ℕ),
    num_ducks + num_cows = total_animals ∧
    num_ducks * duck_legs + num_cows * cow_legs = total_legs ∧
    num_cows = 6 :=
by sorry

end farm_animals_l1664_166429


namespace order_of_powers_l1664_166485

theorem order_of_powers : (3/5)^(1/5 : ℝ) > (1/5 : ℝ)^(1/5 : ℝ) ∧ (1/5 : ℝ)^(1/5 : ℝ) > (1/5 : ℝ)^(3/5 : ℝ) := by
  sorry

end order_of_powers_l1664_166485


namespace circle_area_outside_triangle_l1664_166478

-- Define the right triangle ABC
structure RightTriangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  right_angle : AC^2 = AB^2 + BC^2

-- Define the circle
structure TangentCircle (t : RightTriangle) where
  radius : ℝ
  tangent_AB : radius = t.AB / 2
  diametric_point_on_BC : radius * 2 ≤ t.BC

-- Main theorem
theorem circle_area_outside_triangle (t : RightTriangle) (c : TangentCircle t)
  (h1 : t.AB = 8)
  (h2 : t.BC = 10) :
  (π * c.radius^2 / 4) - (c.radius^2 / 2) = 4*π - 8 := by
  sorry


end circle_area_outside_triangle_l1664_166478


namespace extra_oil_amount_l1664_166480

-- Define the given conditions
def price_reduction : ℚ := 25 / 100
def reduced_price : ℚ := 40
def total_money : ℚ := 800

-- Define the function to calculate the original price
def original_price : ℚ := reduced_price / (1 - price_reduction)

-- Define the function to calculate the amount of oil that can be bought
def oil_amount (price : ℚ) : ℚ := total_money / price

-- State the theorem
theorem extra_oil_amount : 
  oil_amount reduced_price - oil_amount original_price = 5 := by
  sorry

end extra_oil_amount_l1664_166480


namespace vasya_clock_problem_l1664_166451

theorem vasya_clock_problem :
  ¬ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 12 ∧ b ≤ 59 ∧ (100 * a + b) % (60 * a + b) = 0 :=
by sorry

end vasya_clock_problem_l1664_166451


namespace smallest_solution_quartic_equation_l1664_166402

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 50*x^2 + 625 = 0 ∧ x = -5 ∧ ∀ y : ℝ, y^4 - 50*y^2 + 625 = 0 → y ≥ x := by
  sorry

end smallest_solution_quartic_equation_l1664_166402


namespace maximum_marks_calculation_l1664_166449

/-- Given a student who needs to obtain 40% to pass, got 150 marks, and failed by 50 marks, 
    the maximum possible marks are 500. -/
theorem maximum_marks_calculation (passing_threshold : ℝ) (marks_obtained : ℝ) (marks_short : ℝ) :
  passing_threshold = 0.4 →
  marks_obtained = 150 →
  marks_short = 50 →
  ∃ (max_marks : ℝ), max_marks = 500 ∧ passing_threshold * max_marks = marks_obtained + marks_short :=
by sorry

end maximum_marks_calculation_l1664_166449


namespace h_of_two_equals_fifteen_l1664_166409

theorem h_of_two_equals_fifteen (h : ℝ → ℝ) 
  (h_def : ∀ x : ℝ, h (3 * x - 4) = 4 * x + 7) : 
  h 2 = 15 := by
sorry

end h_of_two_equals_fifteen_l1664_166409


namespace square_of_linear_expression_l1664_166456

/-- F is a quadratic function of x with parameter m -/
def F (x m : ℚ) : ℚ := (6 * x^2 + 16 * x + 3 * m) / 6

/-- A linear function of x -/
def linear (a b x : ℚ) : ℚ := a * x + b

theorem square_of_linear_expression (m : ℚ) :
  (∃ a b : ℚ, ∀ x : ℚ, F x m = (linear a b x)^2) → m = 32/9 := by
  sorry

end square_of_linear_expression_l1664_166456


namespace arcade_tickets_l1664_166492

/-- The number of tickets Dave and Alex had combined at the start -/
def total_tickets (dave_spent dave_left alex_spent alex_left : ℕ) : ℕ :=
  (dave_spent + dave_left) + (alex_spent + alex_left)

/-- Theorem stating the total number of tickets Dave and Alex had at the start -/
theorem arcade_tickets : total_tickets 43 55 65 42 = 205 := by
  sorry

end arcade_tickets_l1664_166492


namespace largest_k_for_right_triangle_inequality_l1664_166463

theorem largest_k_for_right_triangle_inequality :
  ∃ (k : ℝ), k = (3 * Real.sqrt 2 - 4) / 2 ∧
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 → a^2 + b^2 = c^2 →
    a^3 + b^3 + c^3 ≥ k * (a + b + c)^3) ∧
  (∀ (k' : ℝ), k' > k →
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧
      a^3 + b^3 + c^3 < k' * (a + b + c)^3) :=
by sorry

end largest_k_for_right_triangle_inequality_l1664_166463


namespace largest_odd_digit_multiple_of_11_l1664_166435

def is_odd_digit (d : ℕ) : Prop := d % 2 = 1 ∧ d ≤ 9

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, (n / 10^d % 10 ≠ 0 → is_odd_digit (n / 10^d % 10))

def alternating_sum (n : ℕ) : ℤ :=
  let digits := List.reverse (List.map (λ i => n / 10^i % 10) (List.range 4))
  List.foldl (λ sum (i, d) => sum + ((-1)^i : ℤ) * d) 0 (List.enumFrom 0 digits)

theorem largest_odd_digit_multiple_of_11 :
  9393 < 10000 ∧
  has_only_odd_digits 9393 ∧
  alternating_sum 9393 % 11 = 0 ∧
  ∀ n : ℕ, n < 10000 → has_only_odd_digits n → alternating_sum n % 11 = 0 → n ≤ 9393 :=
sorry

end largest_odd_digit_multiple_of_11_l1664_166435


namespace quadratic_sum_constrained_l1664_166404

theorem quadratic_sum_constrained (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 10) 
  (sum_squares_condition : p^2 + q^2 + r^2 + s^2 = 20) : 
  3 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) = 64 := by
  sorry

end quadratic_sum_constrained_l1664_166404


namespace factorization_equalities_l1664_166411

theorem factorization_equalities (x y a : ℝ) : 
  (x^4 - 9*x^2 = x^2*(x+3)*(x-3)) ∧ 
  (25*x^2*y + 20*x*y^2 + 4*y^3 = y*(5*x+2*y)^2) ∧ 
  (x^2*(a-1) + y^2*(1-a) = (a-1)*(x+y)*(x-y)) := by
  sorry

end factorization_equalities_l1664_166411


namespace abs_five_minus_e_l1664_166440

-- Define e as a real number approximately equal to 2.718
def e : ℝ := 2.718

-- State the theorem
theorem abs_five_minus_e : |5 - e| = 2.282 := by
  sorry

end abs_five_minus_e_l1664_166440


namespace candy_cost_per_pack_l1664_166416

theorem candy_cost_per_pack (number_of_packs : ℕ) (amount_paid : ℕ) (change_received : ℕ) :
  number_of_packs = 3 →
  amount_paid = 20 →
  change_received = 11 →
  (amount_paid - change_received) / number_of_packs = 3 :=
by
  sorry

end candy_cost_per_pack_l1664_166416


namespace parabola_translation_theorem_l1664_166450

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a 2D translation -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The original parabola y = x² -/
def original_parabola : Parabola :=
  { a := 1, b := 0, c := 0 }

/-- The translation of 1 unit left and 2 units down -/
def translation : Translation :=
  { dx := -1, dy := -2 }

/-- The resulting parabola after translation -/
def translated_parabola (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a
    b := -2 * p.a * t.dx
    c := p.a * t.dx^2 + p.b * t.dx + p.c + t.dy }

theorem parabola_translation_theorem :
  let p := original_parabola
  let t := translation
  let result := translated_parabola p t
  result.a = 1 ∧ result.b = 2 ∧ result.c = -2 :=
by sorry

end parabola_translation_theorem_l1664_166450


namespace diophantine_equation_solvable_l1664_166498

theorem diophantine_equation_solvable (n : ℤ) :
  ∃ (x y z : ℤ), 10 * x * y + 17 * y * z + 27 * z * x = n :=
by sorry

end diophantine_equation_solvable_l1664_166498


namespace square_area_decrease_l1664_166483

theorem square_area_decrease (s : ℝ) (h : s = 12) :
  let new_s := s * (1 - 0.125)
  (s^2 - new_s^2) / s^2 = 0.25 := by sorry

end square_area_decrease_l1664_166483


namespace height_side_relation_l1664_166494

/-- Triangle with sides and heights -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  H_A : ℝ
  H_B : ℝ
  H_C : ℝ

/-- Theorem: In a triangle, if one height is greater than another, the side opposite to the greater height is shorter than the side opposite to the smaller height -/
theorem height_side_relation (t : Triangle) :
  t.H_A > t.H_B → t.B < t.A :=
by sorry

end height_side_relation_l1664_166494


namespace units_digit_of_power_l1664_166422

theorem units_digit_of_power (n : ℕ) : n > 0 → (7^(7 * (13^13))) % 10 = 3 := by
  sorry

end units_digit_of_power_l1664_166422


namespace circle_equations_l1664_166447

/-- A circle in the Cartesian coordinate system -/
structure Circle where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ α, x α = 2 + 2 * Real.cos α
  h_y : ∀ α, y α = 2 * Real.sin α

/-- The Cartesian equation of the circle -/
def cartesian_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 4

/-- The polar equation of the circle -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.cos θ

theorem circle_equations (c : Circle) :
  (∀ x y, cartesian_equation c x y ↔ ∃ α, c.x α = x ∧ c.y α = y) ∧
  (∀ ρ θ, polar_equation ρ θ ↔ cartesian_equation c (ρ * Real.cos θ) (ρ * Real.sin θ)) := by
  sorry

end circle_equations_l1664_166447


namespace speech_arrangement_count_l1664_166453

/-- The number of ways to arrange speeches for 3 boys and 2 girls chosen from a group of 4 boys and 3 girls, where the girls do not give consecutive speeches. -/
def speech_arrangements (total_boys : ℕ) (total_girls : ℕ) (chosen_boys : ℕ) (chosen_girls : ℕ) : ℕ :=
  (Nat.choose total_boys chosen_boys) * 
  (Nat.choose total_girls chosen_girls) * 
  (Nat.factorial chosen_boys) * 
  (Nat.factorial (chosen_boys + 1))

theorem speech_arrangement_count :
  speech_arrangements 4 3 3 2 = 864 := by
  sorry

end speech_arrangement_count_l1664_166453


namespace max_product_on_circle_l1664_166459

/-- The maximum product of xy for integer points on x^2 + y^2 = 100 is 48 -/
theorem max_product_on_circle : 
  (∃ (a b : ℤ), a^2 + b^2 = 100 ∧ a * b = 48) ∧ 
  (∀ (x y : ℤ), x^2 + y^2 = 100 → x * y ≤ 48) := by
  sorry

#check max_product_on_circle

end max_product_on_circle_l1664_166459


namespace correct_contributions_l1664_166460

/-- Represents the business contribution problem -/
structure BusinessContribution where
  total : ℝ
  a_months : ℝ
  b_months : ℝ
  a_received : ℝ
  b_received : ℝ

/-- Theorem stating the correct contributions of A and B -/
theorem correct_contributions (bc : BusinessContribution)
  (h_total : bc.total = 3400)
  (h_a_months : bc.a_months = 12)
  (h_b_months : bc.b_months = 16)
  (h_a_received : bc.a_received = 2070)
  (h_b_received : bc.b_received = 1920) :
  ∃ (a_contribution b_contribution : ℝ),
    a_contribution = 1800 ∧
    b_contribution = 1600 ∧
    a_contribution + b_contribution = bc.total ∧
    (bc.a_received - a_contribution) / (bc.b_received - (bc.total - a_contribution)) =
      (bc.a_months * a_contribution) / (bc.b_months * (bc.total - a_contribution)) :=
by sorry

end correct_contributions_l1664_166460


namespace power_property_iff_square_property_l1664_166413

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, a * b ≠ 0 → f (a * b) ≥ f a + f b

/-- The property that f(a^n) = n * f(a) for all non-zero a and natural n -/
def SatisfiesPowerProperty (f : ℤ → ℤ) : Prop :=
  ∀ a : ℤ, a ≠ 0 → ∀ n : ℕ, f (a ^ n) = n * f a

/-- The property that f(a^2) = 2 * f(a) for all non-zero a -/
def SatisfiesSquareProperty (f : ℤ → ℤ) : Prop :=
  ∀ a : ℤ, a ≠ 0 → f (a ^ 2) = 2 * f a

theorem power_property_iff_square_property (f : ℤ → ℤ) (h : SatisfiesInequality f) :
  SatisfiesPowerProperty f ↔ SatisfiesSquareProperty f :=
sorry

end power_property_iff_square_property_l1664_166413


namespace number_wall_solution_l1664_166466

/-- Represents a number wall with four levels -/
structure NumberWall :=
  (bottom_left : ℕ)
  (bottom_middle_left : ℕ)
  (bottom_middle_right : ℕ)
  (bottom_right : ℕ)

/-- Calculates the value of the top block in the number wall -/
def top_block (wall : NumberWall) : ℕ :=
  wall.bottom_left + wall.bottom_middle_left + wall.bottom_middle_right + wall.bottom_right + 30

/-- Theorem: In a number wall where the top block is 42, and the bottom row contains m, 5, 3, and 6 from left to right, the value of m is 12 -/
theorem number_wall_solution (wall : NumberWall) 
  (h1 : wall.bottom_middle_left = 5)
  (h2 : wall.bottom_middle_right = 3)
  (h3 : wall.bottom_right = 6)
  (h4 : top_block wall = 42) : 
  wall.bottom_left = 12 := by
  sorry

end number_wall_solution_l1664_166466


namespace butterfat_solution_l1664_166444

def butterfat_problem (x : ℝ) : Prop :=
  let initial_volume : ℝ := 8
  let added_volume : ℝ := 20
  let initial_butterfat : ℝ := x / 100
  let added_butterfat : ℝ := 10 / 100
  let final_butterfat : ℝ := 20 / 100
  let total_volume : ℝ := initial_volume + added_volume
  (initial_volume * initial_butterfat + added_volume * added_butterfat) / total_volume = final_butterfat

theorem butterfat_solution : butterfat_problem 45 := by
  sorry

end butterfat_solution_l1664_166444


namespace sum_seven_to_ten_l1664_166469

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = a n * q
  sum_first_two : a 1 + a 2 = 2
  sum_third_fourth : a 3 + a 4 = 4

/-- The sum of the 7th to 10th terms of the geometric sequence is 48 -/
theorem sum_seven_to_ten (seq : GeometricSequence) :
  seq.a 7 + seq.a 8 + seq.a 9 + seq.a 10 = 48 := by
  sorry

end sum_seven_to_ten_l1664_166469


namespace fejes_toth_inequality_l1664_166427

/-- A convex function on [-1, 1] with absolute value at most 1 -/
structure ConvexBoundedFunction :=
  (f : ℝ → ℝ)
  (convex : ConvexOn ℝ (Set.Icc (-1) 1) f)
  (bounded : ∀ x ∈ Set.Icc (-1) 1, |f x| ≤ 1)

/-- The theorem statement -/
theorem fejes_toth_inequality (F : ConvexBoundedFunction) :
  ∃ (a b : ℝ), ∫ x in Set.Icc (-1) 1, |F.f x - (a * x + b)| ≤ 4 - Real.sqrt 8 := by
  sorry

end fejes_toth_inequality_l1664_166427


namespace puzzle_completion_l1664_166439

/-- Puzzle completion problem -/
theorem puzzle_completion 
  (total_pieces : ℕ) 
  (num_children : ℕ) 
  (time_limit : ℕ) 
  (reyn_rate : ℚ) :
  total_pieces = 500 →
  num_children = 4 →
  time_limit = 120 →
  reyn_rate = 25 / 30 →
  (reyn_rate * time_limit + 
   2 * reyn_rate * time_limit + 
   3 * reyn_rate * time_limit + 
   4 * reyn_rate * time_limit) ≥ total_pieces := by
  sorry


end puzzle_completion_l1664_166439


namespace triangle_angle_calculation_l1664_166495

theorem triangle_angle_calculation (D E F : ℝ) : 
  D = 90 →
  E = 2 * F + 15 →
  D + E + F = 180 →
  F = 25 := by
sorry

end triangle_angle_calculation_l1664_166495


namespace kim_hard_round_correct_l1664_166452

/-- A math contest with three rounds of questions --/
structure MathContest where
  easy_points : ℕ
  average_points : ℕ
  hard_points : ℕ
  easy_correct : ℕ
  average_correct : ℕ
  total_points : ℕ

/-- Kim's performance in the math contest --/
def kim_contest : MathContest :=
  { easy_points := 2
  , average_points := 3
  , hard_points := 5
  , easy_correct := 6
  , average_correct := 2
  , total_points := 38 }

/-- The number of correct answers in the hard round --/
def hard_round_correct (contest : MathContest) : ℕ :=
  (contest.total_points - (contest.easy_points * contest.easy_correct + contest.average_points * contest.average_correct)) / contest.hard_points

theorem kim_hard_round_correct :
  hard_round_correct kim_contest = 4 := by
  sorry

end kim_hard_round_correct_l1664_166452


namespace perpendicular_planes_l1664_166426

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (non_coincident : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (l : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : parallel l β)
  (h3 : contained_in l α)
  (h4 : non_coincident α β) :
  plane_perpendicular α β :=
sorry

end perpendicular_planes_l1664_166426


namespace income_savings_theorem_l1664_166418

def income_savings_problem (income : ℝ) (savings : ℝ) : Prop :=
  let income_year2 : ℝ := income * 1.25
  let savings_year2 : ℝ := savings * 2
  let expenditure_year1 : ℝ := income - savings
  let expenditure_year2 : ℝ := income_year2 - savings_year2
  (expenditure_year1 + expenditure_year2 = 2 * expenditure_year1) ∧
  (savings / income = 0.25)

theorem income_savings_theorem (income : ℝ) (savings : ℝ) 
  (h : income > 0) : income_savings_problem income savings :=
by
  sorry

#check income_savings_theorem

end income_savings_theorem_l1664_166418


namespace vector_sum_proof_l1664_166433

/-- Given two vectors a and b in ℝ², prove that their sum is (2, 4) -/
theorem vector_sum_proof :
  let a : ℝ × ℝ := (-1, 6)
  let b : ℝ × ℝ := (3, -2)
  a + b = (2, 4) := by
  sorry

end vector_sum_proof_l1664_166433


namespace soup_tasting_equivalent_to_sample_estimation_l1664_166445

/-- Represents the entire soup -/
def Soup : Type := Unit

/-- Represents a small portion of the soup -/
def SoupSample : Type := Unit

/-- Represents the action of tasting a small portion of soup -/
def TasteSoup : SoupSample → Bool := fun _ => true

/-- Represents a population in a statistical survey -/
def Population : Type := Unit

/-- Represents a sample from a population -/
def PopulationSample : Type := Unit

/-- Represents the process of sample estimation in statistics -/
def SampleEstimation : PopulationSample → Population → Prop := fun _ _ => true

/-- Theorem stating that tasting a small portion of soup is mathematically equivalent
    to using sample estimation in statistical surveys -/
theorem soup_tasting_equivalent_to_sample_estimation :
  ∀ (soup : Soup) (sample : SoupSample) (pop : Population) (pop_sample : PopulationSample),
  TasteSoup sample ↔ SampleEstimation pop_sample pop :=
sorry

end soup_tasting_equivalent_to_sample_estimation_l1664_166445


namespace school_classes_count_l1664_166423

theorem school_classes_count (sheets_per_class_per_day : ℕ) 
                              (total_sheets_per_week : ℕ) 
                              (school_days_per_week : ℕ) :
  sheets_per_class_per_day = 200 →
  total_sheets_per_week = 9000 →
  school_days_per_week = 5 →
  (total_sheets_per_week / (sheets_per_class_per_day * school_days_per_week) : ℕ) = 9 := by
sorry

end school_classes_count_l1664_166423


namespace parabola_vertex_l1664_166414

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 4*y + 3*x + 8 = 0

-- Define the vertex of a parabola
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x' y', eq x' y' → x' ≥ x

-- Theorem stating that (-4/3, -2) is the vertex of the parabola
theorem parabola_vertex :
  is_vertex (-4/3) (-2) parabola_equation :=
sorry

end parabola_vertex_l1664_166414


namespace problem_statement_l1664_166472

theorem problem_statement (x : ℝ) (h : Real.exp (x * Real.log 9) + Real.exp (x * Real.log 3) = 6) :
  Real.exp ((1 / x) * Real.log 16) + Real.exp ((1 / x) * Real.log 4) = 90 := by
  sorry

end problem_statement_l1664_166472


namespace obtuse_triangle_side_length_range_l1664_166458

theorem obtuse_triangle_side_length_range (a : ℝ) :
  (∃ (x y z : ℝ), x = a ∧ y = a + 3 ∧ z = a + 6 ∧
   x + y > z ∧ y + z > x ∧ z + x > y ∧  -- triangle inequality
   z^2 > x^2 + y^2)  -- obtuse triangle condition
  ↔ 3 < a ∧ a < 9 := by
sorry

end obtuse_triangle_side_length_range_l1664_166458


namespace paint_cans_problem_l1664_166436

theorem paint_cans_problem (original_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) :
  original_rooms = 50 →
  lost_cans = 5 →
  remaining_rooms = 35 →
  (∃ (cans_per_room : ℚ), 
    cans_per_room * (original_rooms - remaining_rooms) = lost_cans ∧
    cans_per_room * remaining_rooms = 12) :=
by sorry

end paint_cans_problem_l1664_166436


namespace single_elimination_tournament_games_l1664_166441

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- Calculates the number of games needed to declare a winner in a single-elimination tournament -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams and no ties, 
    the number of games required to declare a winner is 22 -/
theorem single_elimination_tournament_games :
  ∀ (t : Tournament), t.num_teams = 23 → t.no_ties = true → 
  games_to_winner t = 22 :=
by
  sorry

end single_elimination_tournament_games_l1664_166441


namespace f_is_quadratic_l1664_166470

/-- Definition of a quadratic equation in one variable x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l1664_166470


namespace area_inside_circle_outside_rectangle_l1664_166461

/-- The area inside a circle but outside a rectangle with shared center --/
theorem area_inside_circle_outside_rectangle (π : Real) :
  let circle_radius : Real := 1 / 3
  let rectangle_length : Real := 3
  let rectangle_width : Real := 1.5
  let circle_area : Real := π * circle_radius ^ 2
  let rectangle_area : Real := rectangle_length * rectangle_width
  let rectangle_diagonal : Real := (rectangle_length ^ 2 + rectangle_width ^ 2).sqrt
  circle_radius < rectangle_diagonal / 2 →
  circle_area = π / 9 := by
  sorry

end area_inside_circle_outside_rectangle_l1664_166461


namespace blueberry_muffin_percentage_is_fifty_percent_l1664_166428

/-- Calculates the percentage of blueberry muffins given the number of blueberry cartons,
    blueberries per carton, blueberries per muffin, and number of cinnamon muffins. -/
def blueberry_muffin_percentage (
  cartons : ℕ
  ) (blueberries_per_carton : ℕ
  ) (blueberries_per_muffin : ℕ
  ) (cinnamon_muffins : ℕ
  ) : ℚ :=
  let total_blueberries := cartons * blueberries_per_carton
  let blueberry_muffins := total_blueberries / blueberries_per_muffin
  let total_muffins := blueberry_muffins + cinnamon_muffins
  (blueberry_muffins : ℚ) / (total_muffins : ℚ) * 100

/-- Proves that given 3 cartons of 200 blueberries, making muffins with 10 blueberries each,
    and 60 additional cinnamon muffins, the percentage of blueberry muffins is 50% of the total muffins. -/
theorem blueberry_muffin_percentage_is_fifty_percent :
  blueberry_muffin_percentage 3 200 10 60 = 50 := by
  sorry

end blueberry_muffin_percentage_is_fifty_percent_l1664_166428


namespace simplify_fraction_division_l1664_166443

theorem simplify_fraction_division (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  ((1 - x) / x) / ((1 - x) / x^2) = x := by
  sorry

end simplify_fraction_division_l1664_166443


namespace other_diagonal_length_l1664_166438

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ
  d2 : ℝ
  area : ℝ

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.d1 * r.d2) / 2

/-- Given a rhombus with one diagonal of 17 cm and area of 170 cm², 
    the other diagonal is 20 cm -/
theorem other_diagonal_length :
  ∀ (r : Rhombus), r.d1 = 17 ∧ r.area = 170 → r.d2 = 20 := by
  sorry

end other_diagonal_length_l1664_166438


namespace range_of_y_over_x_l1664_166421

theorem range_of_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 3) :
  ∃ (k : ℝ), y / x = k ∧ -Real.sqrt 3 ≤ k ∧ k ≤ Real.sqrt 3 :=
sorry

end range_of_y_over_x_l1664_166421


namespace gary_remaining_money_l1664_166420

/-- The amount of money Gary has left after buying a pet snake -/
def money_left (initial_amount spent_amount : ℕ) : ℕ :=
  initial_amount - spent_amount

/-- Theorem stating that Gary has 18 dollars left after buying a pet snake -/
theorem gary_remaining_money :
  money_left 73 55 = 18 := by
  sorry

end gary_remaining_money_l1664_166420


namespace unique_function_f_l1664_166481

/-- A function from [1,+∞) to [1,+∞) satisfying given conditions -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, x ≥ 1 → f x ≥ 1) ∧ 
  (∀ x : ℝ, x ≥ 1 → f x ≤ 2 * (x + 1)) ∧
  (∀ x : ℝ, x ≥ 1 → f (x + 1) = (1 / x) * ((f x)^2 - 1))

/-- The unique function satisfying the conditions is f(x) = x + 1 -/
theorem unique_function_f :
  ∃! f : ℝ → ℝ, FunctionF f ∧ ∀ x : ℝ, x ≥ 1 → f x = x + 1 :=
sorry

end unique_function_f_l1664_166481


namespace ice_cream_theorem_l1664_166493

/-- The number of permutations of n distinct elements -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- There are 5 distinct flavors of ice cream -/
def num_flavors : ℕ := 5

/-- The number of ways to arrange ice cream scoops -/
def ice_cream_arrangements : ℕ := permutations num_flavors

theorem ice_cream_theorem : ice_cream_arrangements = 120 := by
  sorry

end ice_cream_theorem_l1664_166493


namespace number_fraction_problem_l1664_166499

theorem number_fraction_problem (n : ℝ) : 
  (1/3) * (1/4) * (1/5) * n = 15 → (3/10) * n = 270 := by
sorry

end number_fraction_problem_l1664_166499


namespace arithmetic_geometric_sequence_l1664_166475

/-- 
Given an arithmetic sequence {a_n} with common difference 3,
where a_1, a_3, and a_4 form a geometric sequence,
prove that a_2 = -9
-/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = 3) →  -- arithmetic sequence with common difference 3
  (a 3)^2 = a 1 * a 4 →         -- a_1, a_3, a_4 form a geometric sequence
  a 2 = -9 := by
sorry

end arithmetic_geometric_sequence_l1664_166475


namespace roots_sum_powers_l1664_166488

theorem roots_sum_powers (c d : ℝ) : 
  c^2 - 6*c + 10 = 0 → 
  d^2 - 6*d + 10 = 0 → 
  c^3 + c^5 * d^3 + c^3 * d^5 + d^3 = 16036 := by
sorry

end roots_sum_powers_l1664_166488


namespace circle_circumference_irrational_l1664_166415

/-- A circle with rational diameter has irrational circumference -/
theorem circle_circumference_irrational (d : ℚ) :
  ∃ (C : ℝ), C = π * (d : ℝ) ∧ Irrational C := by
  sorry

end circle_circumference_irrational_l1664_166415


namespace tan_negative_240_degrees_l1664_166448

theorem tan_negative_240_degrees : Real.tan (-(240 * π / 180)) = -Real.sqrt 3 := by
  sorry

end tan_negative_240_degrees_l1664_166448


namespace min_value_squared_sum_l1664_166479

theorem min_value_squared_sum (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) 
  (h2 : t * u * v * w = 25) : 
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 400 := by
  sorry

end min_value_squared_sum_l1664_166479


namespace power_multiplication_l1664_166473

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end power_multiplication_l1664_166473


namespace largest_integer_in_interval_l1664_166490

theorem largest_integer_in_interval (x : ℤ) : 
  (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 3/5 → x ≤ 4 ∧ 
  ∃ y : ℤ, (1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 3/5 ∧ y = 4 :=
by sorry

end largest_integer_in_interval_l1664_166490


namespace simon_practice_requirement_l1664_166408

def week1_hours : ℝ := 12
def week2_hours : ℝ := 16
def week3_hours : ℝ := 14
def total_weeks : ℝ := 4
def required_average : ℝ := 15

def fourth_week_hours : ℝ := 18

theorem simon_practice_requirement :
  (week1_hours + week2_hours + week3_hours + fourth_week_hours) / total_weeks = required_average :=
by sorry

end simon_practice_requirement_l1664_166408


namespace pascal_row_12_left_half_sum_l1664_166425

/-- The sum of the left half of a row in Pascal's Triangle -/
def pascal_left_half_sum (n : ℕ) : ℕ :=
  2^n

/-- Row 12 of Pascal's Triangle -/
def pascal_row_12 : ℕ := 12

theorem pascal_row_12_left_half_sum :
  pascal_left_half_sum pascal_row_12 = 2048 := by
  sorry

end pascal_row_12_left_half_sum_l1664_166425


namespace intersection_correct_l1664_166484

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℚ
  y : ℚ

/-- Represents a parametric line in 2D -/
structure ParametricLine where
  origin : Vector2D
  direction : Vector2D

/-- The first line -/
def line1 : ParametricLine :=
  { origin := { x := 2, y := 2 },
    direction := { x := 3, y := -4 } }

/-- The second line -/
def line2 : ParametricLine :=
  { origin := { x := 4, y := -6 },
    direction := { x := 5, y := 3 } }

/-- Calculates the point on a parametric line given a parameter value -/
def pointOnLine (line : ParametricLine) (t : ℚ) : Vector2D :=
  { x := line.origin.x + t * line.direction.x,
    y := line.origin.y + t * line.direction.y }

/-- The intersection point of the two lines -/
def intersectionPoint : Vector2D :=
  { x := 160 / 29, y := -160 / 29 }

/-- Theorem stating that the calculated intersection point is correct -/
theorem intersection_correct :
  ∃ (t u : ℚ), pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint :=
by
  sorry


end intersection_correct_l1664_166484


namespace sufficient_condition_l1664_166400

theorem sufficient_condition (a : ℝ) : a > 0 → a^2 + a ≥ 0 := by
  sorry

end sufficient_condition_l1664_166400


namespace no_real_roots_quadratic_l1664_166464

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - 2*k + 3 ≠ 0) → k < 1 := by
  sorry

end no_real_roots_quadratic_l1664_166464


namespace right_triangle_and_perimeter_range_l1664_166430

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the right triangle and its perimeter range -/
theorem right_triangle_and_perimeter_range (t : Triangle) (h : t.a * (Real.cos t.B + Real.cos t.C) = t.b + t.c) :
  t.A = π / 2 ∧ 
  (∀ (r : ℝ), r = 1 → 4 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 2 + 2 * Real.sqrt 2) :=
by sorry

end right_triangle_and_perimeter_range_l1664_166430


namespace madeline_money_l1664_166491

theorem madeline_money (madeline_money : ℝ) (brother_money : ℝ) : 
  brother_money = madeline_money / 2 →
  madeline_money + brother_money = 72 →
  madeline_money = 48 := by
sorry

end madeline_money_l1664_166491


namespace f_composition_value_l1664_166455

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem f_composition_value :
  f (f (π / 12)) = (1 / 2) * Real.sin (1 / 2) := by
  sorry

end f_composition_value_l1664_166455


namespace power_sum_equality_l1664_166406

theorem power_sum_equality : (-2)^23 + 5^(2^4 + 3^3 - 4^2) = -8388608 + 5^27 := by
  sorry

end power_sum_equality_l1664_166406


namespace distance_between_points_l1664_166412

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (6, 5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 34 :=
by sorry

end distance_between_points_l1664_166412


namespace smallest_N_for_P_less_than_four_fifths_l1664_166431

/-- The probability function P(N) as described in the problem -/
def P (N : ℕ) : ℚ :=
  (2 * N * N) / (9 * (N + 2) * (N + 3))

/-- Predicate to check if a number is a multiple of 6 -/
def isMultipleOf6 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 6 * k

theorem smallest_N_for_P_less_than_four_fifths :
  ∀ N : ℕ, isMultipleOf6 N → N < 600 → P N ≥ 4/5 ∧
  isMultipleOf6 600 ∧ P 600 < 4/5 := by
  sorry

#eval P 600 -- To verify that P(600) is indeed less than 4/5

end smallest_N_for_P_less_than_four_fifths_l1664_166431


namespace arithmetic_mean_of_scores_l1664_166487

def scores : List ℕ := [87, 90, 85, 93, 89, 92]

theorem arithmetic_mean_of_scores :
  (scores.sum : ℚ) / scores.length = 268 / 3 := by
  sorry

end arithmetic_mean_of_scores_l1664_166487


namespace m_range_l1664_166401

def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*(m+1)*x + m*(m+1) > 0

theorem m_range (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) :
  m > 2 ∨ (-2 ≤ m ∧ m < -1) :=
sorry

end m_range_l1664_166401


namespace e_squared_f_2_gt_e_cubed_f_3_l1664_166457

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the condition that f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Define the given condition
axiom condition : ∀ x, f x + f' x < 0

-- State the theorem to be proved
theorem e_squared_f_2_gt_e_cubed_f_3 : e^2 * f 2 > e^3 * f 3 := by sorry

end e_squared_f_2_gt_e_cubed_f_3_l1664_166457


namespace complement_A_intersect_B_l1664_166403

-- Define the sets A and B
def A : Set ℝ := {x | x + 1 < 0}
def B : Set ℝ := {x | x - 3 < 0}

-- Define the complement of A in the universal set ℝ
def C_U_A : Set ℝ := {x | ¬ (x ∈ A)}

-- State the theorem
theorem complement_A_intersect_B :
  (C_U_A ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 3} := by sorry

end complement_A_intersect_B_l1664_166403


namespace expression_value_l1664_166489

theorem expression_value : 
  1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1200 := by
  sorry

end expression_value_l1664_166489


namespace sum_of_specific_primes_l1664_166446

def smallest_odd_prime : ℕ := 3

def largest_prime_less_than_50 : ℕ := 47

def smallest_prime_greater_than_60 : ℕ := 61

theorem sum_of_specific_primes :
  smallest_odd_prime + largest_prime_less_than_50 + smallest_prime_greater_than_60 = 111 :=
by sorry

end sum_of_specific_primes_l1664_166446


namespace cans_collected_l1664_166442

theorem cans_collected (total_cans : ℕ) (ladonna_cans : ℕ) (prikya_cans : ℕ) (yoki_cans : ℕ) :
  total_cans = 85 →
  ladonna_cans = 25 →
  prikya_cans = 2 * ladonna_cans →
  yoki_cans = total_cans - (ladonna_cans + prikya_cans) →
  yoki_cans = 10 := by
  sorry

end cans_collected_l1664_166442


namespace college_students_count_l1664_166424

theorem college_students_count :
  ∀ (students professors : ℕ),
  students = 15 * professors →
  students + professors = 40000 →
  students = 37500 :=
by
  sorry

end college_students_count_l1664_166424


namespace committee_probability_l1664_166477

def total_members : ℕ := 30
def boys : ℕ := 12
def girls : ℕ := 18
def committee_size : ℕ := 5

theorem committee_probability :
  let total_combinations := Nat.choose total_members committee_size
  let all_boys_combinations := Nat.choose boys committee_size
  let all_girls_combinations := Nat.choose girls committee_size
  let favorable_combinations := total_combinations - (all_boys_combinations + all_girls_combinations)
  (favorable_combinations : ℚ) / total_combinations = 59 / 63 := by
  sorry

end committee_probability_l1664_166477


namespace cos_minus_sin_special_angle_l1664_166476

/-- An angle whose initial side coincides with the non-negative x-axis
    and whose terminal side lies on the ray 4x - 3y = 0 (x ≤ 0) -/
def special_angle (α : Real) : Prop :=
  ∃ (x y : Real), x ≤ 0 ∧ 4 * x - 3 * y = 0 ∧
  Real.cos α = x / Real.sqrt (x^2 + y^2) ∧
  Real.sin α = y / Real.sqrt (x^2 + y^2)

/-- Theorem: For a special angle α, cos α - sin α = 1/5 -/
theorem cos_minus_sin_special_angle (α : Real) (h : special_angle α) :
  Real.cos α - Real.sin α = 1/5 := by
  sorry

end cos_minus_sin_special_angle_l1664_166476


namespace find_n_l1664_166419

/-- Definition of S_n -/
def S (n : ℕ) : ℚ := n / (n + 1)

/-- Theorem stating that n = 6 satisfies the given conditions -/
theorem find_n : ∃ (n : ℕ), S n * S (n + 1) = 3/4 ∧ n = 6 := by
  sorry

end find_n_l1664_166419


namespace polar_equation_is_circle_l1664_166497

/-- The curve represented by the polar equation ρ = sin θ + cos θ is a circle. -/
theorem polar_equation_is_circle :
  ∀ (ρ θ : ℝ), ρ = Real.sin θ + Real.cos θ →
  ∃ (x₀ y₀ r : ℝ), ∀ (x y : ℝ),
    (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
    (x - x₀)^2 + (y - y₀)^2 = r^2 :=
by sorry

end polar_equation_is_circle_l1664_166497


namespace systematic_sampling_theorem_l1664_166417

def systematic_sampling (total_members : ℕ) (num_groups : ℕ) (group_number : ℕ) (number_in_group : ℕ) : ℕ :=
  number_in_group - (group_number - 1) * (total_members / num_groups)

theorem systematic_sampling_theorem (total_members num_groups group_5 group_3 : ℕ) 
  (h1 : total_members = 200)
  (h2 : num_groups = 40)
  (h3 : group_5 = 5)
  (h4 : group_3 = 3)
  (h5 : systematic_sampling total_members num_groups group_5 22 = 22) :
  systematic_sampling total_members num_groups group_3 22 = 12 :=
by
  sorry

end systematic_sampling_theorem_l1664_166417


namespace sequence_bound_l1664_166462

def is_valid_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ (a n)^2 ≤ a n - a (n + 1)

theorem sequence_bound (a : ℕ → ℝ) (h : is_valid_sequence a) :
  ∀ n : ℕ, a n < 1 / n :=
sorry

end sequence_bound_l1664_166462


namespace rohans_age_puzzle_l1664_166465

/-- 
Given that Rohan is currently 25 years old, this theorem proves that the number of years 
into the future when Rohan will be 4 times as old as he was the same number of years ago is 15.
-/
theorem rohans_age_puzzle : 
  ∃ x : ℕ, (25 + x = 4 * (25 - x)) ∧ x = 15 :=
by sorry

end rohans_age_puzzle_l1664_166465


namespace discounted_price_l1664_166486

theorem discounted_price (original_price : ℝ) : 
  original_price * (1 - 0.20) * (1 - 0.10) * (1 - 0.05) = 6840 → 
  original_price = 10000 := by
sorry

end discounted_price_l1664_166486


namespace no_solution_exists_l1664_166437

theorem no_solution_exists : 
  ¬ ∃ (a b : ℕ+), a * b + 75 = 15 * Nat.lcm a b + 10 * Nat.gcd a b :=
sorry

end no_solution_exists_l1664_166437


namespace complex_number_range_l1664_166434

/-- The range of y/x for a complex number (x-2) + yi with modulus 1 -/
theorem complex_number_range (x y : ℝ) : 
  (x - 2)^2 + y^2 = 1 → 
  y ≠ 0 → 
  ∃ k : ℝ, y = k * x ∧ 
    (-Real.sqrt 3 / 3 ≤ k ∧ k < 0) ∨ (0 < k ∧ k ≤ Real.sqrt 3 / 3) :=
sorry

end complex_number_range_l1664_166434


namespace edwards_summer_earnings_l1664_166474

/-- Given Edward's lawn mowing business earnings and expenses, prove the amount he made in the summer. -/
theorem edwards_summer_earnings (spring_earnings : ℕ) (supplies_cost : ℕ) (final_amount : ℕ) :
  spring_earnings = 2 →
  supplies_cost = 5 →
  final_amount = 24 →
  ∃ summer_earnings : ℕ, spring_earnings + summer_earnings - supplies_cost = final_amount ∧ summer_earnings = 27 :=
by sorry

end edwards_summer_earnings_l1664_166474


namespace xy_yz_zx_geq_3_l1664_166468

theorem xy_yz_zx_geq_3 (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (heq : x + y + z = 1/x + 1/y + 1/z) : x*y + y*z + z*x ≥ 3 := by
  sorry

end xy_yz_zx_geq_3_l1664_166468


namespace sin_cos_tan_product_l1664_166410

theorem sin_cos_tan_product : 
  Real.sin (4/3 * Real.pi) * Real.cos (5/6 * Real.pi) * Real.tan (-4/3 * Real.pi) = -(3 * Real.sqrt 3) / 4 := by
  sorry

end sin_cos_tan_product_l1664_166410


namespace parabola_focus_directrix_distance_l1664_166432

/-- Theorem: For a parabola y = ax^2 where a > 0, if the distance from the focus to the directrix is 1, then a = 1/2 -/
theorem parabola_focus_directrix_distance (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, y = a * x^2) → -- Parabola equation
  (∃ p : ℝ, p = 1 ∧ p = 1 / (2 * a)) → -- Distance from focus to directrix is 1
  a = 1 / 2 := by sorry

end parabola_focus_directrix_distance_l1664_166432


namespace college_student_count_l1664_166454

theorem college_student_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 200) :
  boys + girls = 520 := by
  sorry

end college_student_count_l1664_166454


namespace only_expr3_not_factorable_l1664_166471

-- Define the expressions
def expr1 (a b : ℝ) := a^2 - b^2
def expr2 (x y z : ℝ) := 49*x^2 - y^2*z^2
def expr3 (x y : ℝ) := -x^2 - y^2
def expr4 (m n p : ℝ) := 16*m^2*n^2 - 25*p^2

-- Define the difference of squares formula
def diff_of_squares (a b : ℝ) := (a + b) * (a - b)

-- Theorem statement
theorem only_expr3_not_factorable :
  (∃ (a b : ℝ), expr1 a b = diff_of_squares a b) ∧
  (∃ (x y z : ℝ), expr2 x y z = diff_of_squares (7*x) (y*z)) ∧
  (∀ (x y : ℝ), ¬∃ (a b : ℝ), expr3 x y = diff_of_squares a b) ∧
  (∃ (m n p : ℝ), expr4 m n p = diff_of_squares (4*m*n) (5*p)) :=
sorry

end only_expr3_not_factorable_l1664_166471

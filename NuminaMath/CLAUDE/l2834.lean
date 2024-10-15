import Mathlib

namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2834_283462

-- Define the conditions
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x - a ≤ 0
def q (a : ℝ) : Prop := a > 0 ∨ a < -1

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ a : ℝ, q a → p a) ∧ 
  (∃ a : ℝ, p a ∧ ¬(q a)) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2834_283462


namespace NUMINAMATH_CALUDE_smallest_constant_for_triangle_sides_l2834_283412

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem smallest_constant_for_triangle_sides (t : Triangle) :
  (t.a^2 + t.b^2) / (t.a * t.b) ≥ 2 ∧
  ∀ N, (∀ t' : Triangle, (t'.a^2 + t'.b^2) / (t'.a * t'.b) < N) → N ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_constant_for_triangle_sides_l2834_283412


namespace NUMINAMATH_CALUDE_jack_barbecue_sauce_l2834_283472

/-- The amount of vinegar used in Jack's barbecue sauce recipe -/
def vinegar_amount : ℚ → Prop :=
  fun v =>
    let ketchup : ℚ := 3
    let honey : ℚ := 1
    let burger_sauce : ℚ := 1/4
    let sandwich_sauce : ℚ := 1/6
    let num_burgers : ℚ := 8
    let num_sandwiches : ℚ := 18
    let total_sauce : ℚ := num_burgers * burger_sauce + num_sandwiches * sandwich_sauce
    ketchup + v + honey = total_sauce

theorem jack_barbecue_sauce :
  vinegar_amount 1 := by sorry

end NUMINAMATH_CALUDE_jack_barbecue_sauce_l2834_283472


namespace NUMINAMATH_CALUDE_endocrine_cells_synthesize_both_l2834_283400

structure Cell :=
  (canSynthesizeEnzymes : Bool)
  (canSynthesizeHormones : Bool)

structure Hormone :=
  (producedByEndocrine : Bool)
  (directlyParticipateInCells : Bool)

structure Enzyme :=
  (producedByLivingCells : Bool)

def EndocrineCell := {c : Cell // c.canSynthesizeHormones = true}

theorem endocrine_cells_synthesize_both :
  ∀ (h : Hormone) (e : Enzyme) (ec : EndocrineCell),
    h.directlyParticipateInCells = false →
    e.producedByLivingCells = true →
    h.producedByEndocrine = true →
    ec.val.canSynthesizeEnzymes = true ∧ ec.val.canSynthesizeHormones = true :=
by sorry

end NUMINAMATH_CALUDE_endocrine_cells_synthesize_both_l2834_283400


namespace NUMINAMATH_CALUDE_percentage_female_officers_on_duty_l2834_283441

/-- Given a police force with female officers, calculate the percentage on duty. -/
theorem percentage_female_officers_on_duty
  (total_on_duty : ℕ)
  (half_on_duty_female : ℕ)
  (total_female_officers : ℕ)
  (h1 : total_on_duty = 300)
  (h2 : half_on_duty_female = total_on_duty / 2)
  (h3 : total_female_officers = 1000) :
  (half_on_duty_female : ℚ) / total_female_officers * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_percentage_female_officers_on_duty_l2834_283441


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_6241_l2834_283433

theorem largest_prime_factor_of_6241 : ∃ (p : ℕ), p.Prime ∧ p ∣ 6241 ∧ ∀ (q : ℕ), q.Prime → q ∣ 6241 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_6241_l2834_283433


namespace NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l2834_283448

/-- Definition of the sequence x_n -/
def x : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 6 * x (n + 1) - x n

/-- Theorem stating that no term in the sequence is a perfect square -/
theorem no_perfect_squares_in_sequence : ∀ n : ℕ, ¬∃ m : ℤ, x n = m * m :=
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l2834_283448


namespace NUMINAMATH_CALUDE_complex_polygon_area_l2834_283413

/-- A complex polygon with specific properties -/
structure ComplexPolygon where
  sides : Nat
  side_length : ℝ
  perimeter : ℝ
  is_perpendicular : Bool
  is_congruent : Bool

/-- The area of the complex polygon -/
noncomputable def polygon_area (p : ComplexPolygon) : ℝ :=
  96

/-- Theorem stating the area of the specific complex polygon -/
theorem complex_polygon_area 
  (p : ComplexPolygon) 
  (h1 : p.sides = 32) 
  (h2 : p.perimeter = 64) 
  (h3 : p.is_perpendicular = true) 
  (h4 : p.is_congruent = true) : 
  polygon_area p = 96 := by
  sorry


end NUMINAMATH_CALUDE_complex_polygon_area_l2834_283413


namespace NUMINAMATH_CALUDE_tourist_contact_probability_l2834_283408

/-- The probability that at least one tourist from the first group can contact at least one tourist from the second group -/
def contact_probability (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 42

/-- Theorem stating the probability of contact between two groups of tourists -/
theorem tourist_contact_probability 
  (group1_size : ℕ) 
  (group2_size : ℕ) 
  (p : ℝ) 
  (h1 : group1_size = 6) 
  (h2 : group2_size = 7) 
  (h3 : 0 ≤ p ∧ p ≤ 1) : 
  contact_probability p = 1 - (1 - p) ^ (group1_size * group2_size) :=
by sorry

end NUMINAMATH_CALUDE_tourist_contact_probability_l2834_283408


namespace NUMINAMATH_CALUDE_abc_subtraction_problem_l2834_283495

theorem abc_subtraction_problem (a b c : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (100 * b + 10 * c + a) - (100 * a + 10 * b + c) = 682 →
  a = 3 ∧ b = 7 ∧ c = 5 := by
sorry

end NUMINAMATH_CALUDE_abc_subtraction_problem_l2834_283495


namespace NUMINAMATH_CALUDE_system_solution_l2834_283470

theorem system_solution :
  ∃! (x y : ℚ), (4 * x - 3 * y = -7) ∧ (5 * x + 4 * y = -2) ∧ x = -34/31 ∧ y = 27/31 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2834_283470


namespace NUMINAMATH_CALUDE_age_ratio_constant_l2834_283465

/-- Given two people p and q, where the ratio of their present ages is 3:4 and their total age is 28,
    prove that p's age was always 3/4 of q's age at any point in the past. -/
theorem age_ratio_constant
  (p q : ℕ) -- present ages of p and q
  (h1 : p * 4 = q * 3) -- ratio of present ages is 3:4
  (h2 : p + q = 28) -- total present age is 28
  (t : ℕ) -- time in the past
  (h3 : t ≤ min p q) -- ensure t is not greater than either age
  : (p - t) * 4 = (q - t) * 3 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_constant_l2834_283465


namespace NUMINAMATH_CALUDE_only_b_q_rotationally_symmetric_l2834_283445

/-- Represents an English letter -/
inductive Letter
| B
| D
| P
| Q

/-- Defines rotational symmetry between two letters -/
def rotationallySymmetric (l1 l2 : Letter) : Prop :=
  match l1, l2 with
  | Letter.B, Letter.Q => True
  | Letter.Q, Letter.B => True
  | _, _ => False

/-- Theorem stating that only B and Q are rotationally symmetric -/
theorem only_b_q_rotationally_symmetric :
  ∀ (l1 l2 : Letter),
    rotationallySymmetric l1 l2 ↔ (l1 = Letter.B ∧ l2 = Letter.Q) ∨ (l1 = Letter.Q ∧ l2 = Letter.B) :=
by sorry

#check only_b_q_rotationally_symmetric

end NUMINAMATH_CALUDE_only_b_q_rotationally_symmetric_l2834_283445


namespace NUMINAMATH_CALUDE_equivalent_discount_l2834_283438

theorem equivalent_discount (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.15
  let second_discount := 0.25
  let price_after_first := original_price * (1 - first_discount)
  let price_after_second := price_after_first * (1 - second_discount)
  let equivalent_discount := 1 - (price_after_second / original_price)
  equivalent_discount = 0.3625 := by
sorry

end NUMINAMATH_CALUDE_equivalent_discount_l2834_283438


namespace NUMINAMATH_CALUDE_cats_owners_percentage_l2834_283458

/-- The percentage of students who own cats, given 75 out of 450 students own cats. -/
def percentage_cats_owners : ℚ :=
  75 / 450 * 100

/-- Theorem: The percentage of students who own cats is 16.6% (recurring). -/
theorem cats_owners_percentage :
  percentage_cats_owners = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cats_owners_percentage_l2834_283458


namespace NUMINAMATH_CALUDE_four_term_expression_l2834_283432

theorem four_term_expression (x : ℝ) : 
  ∃ (a b c d : ℝ), (x^3 - 2)^2 + (x^2 + 2*x)^2 = a*x^6 + b*x^4 + c*x^2 + d ∧ 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_four_term_expression_l2834_283432


namespace NUMINAMATH_CALUDE_cloth_selling_price_l2834_283409

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter -/
def totalSellingPrice (quantity : ℕ) (profitPerMeter : ℕ) (costPricePerMeter : ℕ) : ℕ :=
  quantity * (costPricePerMeter + profitPerMeter)

/-- Proves that the total selling price for 85 meters of cloth with a profit of Rs. 25 per meter 
    and a cost price of Rs. 80 per meter is Rs. 8925 -/
theorem cloth_selling_price :
  totalSellingPrice 85 25 80 = 8925 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l2834_283409


namespace NUMINAMATH_CALUDE_divisibility_of_polynomial_l2834_283490

theorem divisibility_of_polynomial (x : ℕ) (h_prime : Nat.Prime x) (h_gt3 : x > 3) :
  (∃ n : ℤ, x = 3 * n + 1 ∧ (x^6 - x^3 - x^2 + x) % 12 = 0) ∨
  (∃ n : ℤ, x = 3 * n - 1 ∧ (x^6 - x^3 - x^2 + x) % 36 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_polynomial_l2834_283490


namespace NUMINAMATH_CALUDE_abs_neg_three_l2834_283488

theorem abs_neg_three : abs (-3 : ℤ) = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_l2834_283488


namespace NUMINAMATH_CALUDE_f_satisfies_equation_l2834_283460

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem f_satisfies_equation : ∀ x : ℝ, 2 * (f x) + f (-x) = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_equation_l2834_283460


namespace NUMINAMATH_CALUDE_minimum_dresses_for_six_colors_one_style_l2834_283455

theorem minimum_dresses_for_six_colors_one_style 
  (num_colors : ℕ) 
  (num_styles : ℕ) 
  (max_extraction_time : ℕ) 
  (h1 : num_colors = 10)
  (h2 : num_styles = 9)
  (h3 : max_extraction_time = 60) :
  ∃ (min_dresses : ℕ),
    (∀ (n : ℕ), n < min_dresses → 
      ¬(∃ (style : ℕ), style < num_styles ∧ 
        (∃ (colors : Finset ℕ), colors.card = 6 ∧ 
          (∀ (c : ℕ), c ∈ colors → c < num_colors) ∧
          (∀ (c1 c2 : ℕ), c1 ∈ colors → c2 ∈ colors → c1 ≠ c2 → 
            ∃ (t1 t2 : ℕ), t1 < max_extraction_time ∧ t2 < max_extraction_time ∧ t1 ≠ t2)))) ∧
    (∃ (style : ℕ), style < num_styles ∧ 
      (∃ (colors : Finset ℕ), colors.card = 6 ∧ 
        (∀ (c : ℕ), c ∈ colors → c < num_colors) ∧
        (∀ (c1 c2 : ℕ), c1 ∈ colors → c2 ∈ colors → c1 ≠ c2 → 
          ∃ (t1 t2 : ℕ), t1 < max_extraction_time ∧ t2 < max_extraction_time ∧ t1 ≠ t2))) ∧
    min_dresses = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_dresses_for_six_colors_one_style_l2834_283455


namespace NUMINAMATH_CALUDE_equation_solutions_l2834_283454

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 36 = 0 ↔ x = 6 ∨ x = -6) ∧
  (∀ x : ℝ, (x+1)^3 + 27 = 0 ↔ x = -4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2834_283454


namespace NUMINAMATH_CALUDE_shifted_parabola_equation_l2834_283414

/-- Represents a parabola in the form y = (x - h)^2 + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { h := 0, k := 0 }

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (units : ℝ) : Parabola :=
  { h := p.h - units, k := p.k }

/-- The equation of a parabola in terms of x -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  (x - p.h)^2 + p.k

theorem shifted_parabola_equation :
  let shifted := shift_parabola original_parabola 2
  ∀ x, parabola_equation shifted x = (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_shifted_parabola_equation_l2834_283414


namespace NUMINAMATH_CALUDE_quadratic_roots_l2834_283476

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_roots (b c : ℝ) :
  (f b c (-2) = 5) →
  (f b c (-1) = 0) →
  (f b c 0 = -3) →
  (f b c 1 = -4) →
  (f b c 2 = -3) →
  (f b c 4 = 5) →
  (∃ x, f b c x = 0) →
  (∀ x, f b c x = 0 ↔ (x = -1 ∨ x = 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2834_283476


namespace NUMINAMATH_CALUDE_M_is_hypersquared_l2834_283496

def n : ℕ := 1000

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y^2

def first_n_digits (x : ℕ) (n : ℕ) : ℕ := x / 10^n

def last_n_digits (x : ℕ) (n : ℕ) : ℕ := x % 10^n

def is_hypersquared (x : ℕ) : Prop :=
  ∃ n : ℕ,
    (x ≥ 10^(2*n - 1)) ∧
    (x < 10^(2*n)) ∧
    is_perfect_square x ∧
    is_perfect_square (first_n_digits x n) ∧
    is_perfect_square (last_n_digits x n) ∧
    (last_n_digits x n ≥ 10^(n-1))

def M : ℕ := ((5 * 10^(n-1) - 1) * 10^n + (10^n - 1))^2

theorem M_is_hypersquared : is_hypersquared M := by
  sorry

end NUMINAMATH_CALUDE_M_is_hypersquared_l2834_283496


namespace NUMINAMATH_CALUDE_new_person_weight_l2834_283439

theorem new_person_weight 
  (n : ℕ) 
  (initial_weight : ℝ) 
  (weight_increase : ℝ) 
  (replaced_weight : ℝ) :
  n = 8 →
  weight_increase = 2.5 →
  replaced_weight = 75 →
  initial_weight + (n : ℝ) * weight_increase = initial_weight - replaced_weight + 95 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2834_283439


namespace NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l2834_283403

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop :=
  n ≥ 10000000 ∧ n < 100000000

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k, n / (10^k) % 10 = d

theorem largest_eight_digit_with_even_digits :
  ∀ n : Nat, is_eight_digit n → contains_all_even_digits n →
  n ≤ 99986420 :=
sorry

end NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l2834_283403


namespace NUMINAMATH_CALUDE_cloth_sold_meters_l2834_283424

/-- Proves that the number of meters of cloth sold is 85 given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sold_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
    (h1 : total_selling_price = 8925)
    (h2 : profit_per_meter = 25)
    (h3 : cost_price_per_meter = 80) :
    (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 85 := by
  sorry

#eval (8925 / (80 + 25) : ℕ)  -- Should output 85

end NUMINAMATH_CALUDE_cloth_sold_meters_l2834_283424


namespace NUMINAMATH_CALUDE_average_age_when_youngest_born_l2834_283440

/-- Proves that given a group of 7 people with an average age of 50 years,
    where the youngest is 5 years old, the average age of the remaining 6 people
    5 years ago was 57.5 years. -/
theorem average_age_when_youngest_born
  (total_people : ℕ)
  (average_age : ℝ)
  (youngest_age : ℝ)
  (total_age : ℝ)
  (h1 : total_people = 7)
  (h2 : average_age = 50)
  (h3 : youngest_age = 5)
  (h4 : total_age = average_age * total_people)
  : (total_age - youngest_age) / (total_people - 1) = 57.5 := by
  sorry

#check average_age_when_youngest_born

end NUMINAMATH_CALUDE_average_age_when_youngest_born_l2834_283440


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l2834_283474

theorem circle_area_from_circumference (C : ℝ) (h : C = 36) :
  let r := C / (2 * Real.pi)
  (Real.pi * r^2) = 324 / Real.pi := by sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l2834_283474


namespace NUMINAMATH_CALUDE_composite_evaluation_l2834_283461

/-- A polynomial with coefficients either 0 or 1 -/
def BinaryPolynomial (P : Polynomial ℤ) : Prop :=
  ∀ i, P.coeff i = 0 ∨ P.coeff i = 1

/-- A polynomial is nonconstant -/
def IsNonconstant (P : Polynomial ℤ) : Prop :=
  ∃ i > 0, P.coeff i ≠ 0

theorem composite_evaluation
  (P : Polynomial ℤ)
  (h_binary : BinaryPolynomial P)
  (h_factorizable : ∃ (f g : Polynomial ℤ), P = f * g ∧ IsNonconstant f ∧ IsNonconstant g) :
  ∃ (a b : ℤ), a > 1 ∧ b > 1 ∧ P.eval 2 = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_evaluation_l2834_283461


namespace NUMINAMATH_CALUDE_no_prime_satisfies_equation_l2834_283411

theorem no_prime_satisfies_equation : ¬∃ (p : ℕ), 
  Nat.Prime p ∧ 
  (2 * p^2 + 5 * p + 3) + (5 * p^2 + p + 2) + (p^2 + 1) + (2 * p^2 + 4 * p + 3) + (p^2 + 6) = 
  (7 * p^2 + 6 * p + 5) + (4 * p^2 + 3 * p + 2) + (p^2 + 2 * p) := by
  sorry

#check no_prime_satisfies_equation

end NUMINAMATH_CALUDE_no_prime_satisfies_equation_l2834_283411


namespace NUMINAMATH_CALUDE_mechanic_parts_cost_l2834_283477

/-- A problem about calculating the cost of parts in a mechanic's bill -/
theorem mechanic_parts_cost
  (hourly_rate : ℝ)
  (job_duration : ℝ)
  (total_bill : ℝ)
  (h1 : hourly_rate = 45)
  (h2 : job_duration = 5)
  (h3 : total_bill = 450) :
  total_bill - hourly_rate * job_duration = 225 := by
  sorry

end NUMINAMATH_CALUDE_mechanic_parts_cost_l2834_283477


namespace NUMINAMATH_CALUDE_iris_shopping_cost_l2834_283452

-- Define the quantities and prices
def num_jackets : ℕ := 3
def price_jacket : ℕ := 10
def num_shorts : ℕ := 2
def price_shorts : ℕ := 6
def num_pants : ℕ := 4
def price_pants : ℕ := 12

-- Define the total cost function
def total_cost : ℕ :=
  num_jackets * price_jacket +
  num_shorts * price_shorts +
  num_pants * price_pants

-- Theorem stating the total cost is $90
theorem iris_shopping_cost : total_cost = 90 := by
  sorry

end NUMINAMATH_CALUDE_iris_shopping_cost_l2834_283452


namespace NUMINAMATH_CALUDE_sum_sqrt_inequality_l2834_283464

theorem sum_sqrt_inequality (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_sqrt_inequality_l2834_283464


namespace NUMINAMATH_CALUDE_minute_hand_rotation_l2834_283422

-- Define the constants
def full_rotation_minutes : ℝ := 60
def full_rotation_degrees : ℝ := 360
def minutes_moved : ℝ := 10

-- Define the theorem
theorem minute_hand_rotation : 
  -(minutes_moved / full_rotation_minutes * full_rotation_degrees * (π / 180)) = -π/3 := by
  sorry

end NUMINAMATH_CALUDE_minute_hand_rotation_l2834_283422


namespace NUMINAMATH_CALUDE_sampling_theorem_l2834_283497

/-- Staff distribution in departments A and B -/
structure StaffDistribution where
  maleA : ℕ
  femaleA : ℕ
  maleB : ℕ
  femaleB : ℕ

/-- Sampling method for selecting staff members -/
inductive SamplingMethod
  | Stratified : SamplingMethod

/-- Result of the sampling process -/
structure SamplingResult where
  fromA : ℕ
  fromB : ℕ
  totalSelected : ℕ

/-- Theorem stating the probability of selecting at least one female from A
    and the expectation of the number of males selected -/
theorem sampling_theorem (sd : StaffDistribution) (sm : SamplingMethod) (sr : SamplingResult) :
  sd.maleA = 6 ∧ sd.femaleA = 4 ∧ sd.maleB = 3 ∧ sd.femaleB = 2 ∧
  sm = SamplingMethod.Stratified ∧
  sr.fromA = 2 ∧ sr.fromB = 1 ∧ sr.totalSelected = 3 →
  (ProbabilityAtLeastOneFemaleFromA = 2/3) ∧
  (ExpectationOfMalesSelected = 9/5) := by
  sorry

end NUMINAMATH_CALUDE_sampling_theorem_l2834_283497


namespace NUMINAMATH_CALUDE_common_divisors_9240_10080_l2834_283467

theorem common_divisors_9240_10080 : Nat.card {d : ℕ | d ∣ 9240 ∧ d ∣ 10080} = 48 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_9240_10080_l2834_283467


namespace NUMINAMATH_CALUDE_four_pencils_per_child_l2834_283437

/-- Given a group of children and pencils, calculate the number of pencils per child. -/
def pencils_per_child (num_children : ℕ) (total_pencils : ℕ) : ℕ :=
  total_pencils / num_children

/-- Theorem stating that with 8 children and 32 pencils, each child has 4 pencils. -/
theorem four_pencils_per_child :
  pencils_per_child 8 32 = 4 := by
  sorry

#eval pencils_per_child 8 32

end NUMINAMATH_CALUDE_four_pencils_per_child_l2834_283437


namespace NUMINAMATH_CALUDE_solution_pair_l2834_283487

theorem solution_pair : ∃ (x y : ℤ), 
  Real.sqrt (4 - 3 * Real.sin (30 * π / 180)) = x + y * (1 / Real.sin (30 * π / 180)) ∧ 
  x = 0 ∧ y = 1 := by
  sorry

#check solution_pair

end NUMINAMATH_CALUDE_solution_pair_l2834_283487


namespace NUMINAMATH_CALUDE_function_properties_l2834_283426

/-- The function f(x) = x^2 - 6x + 10 -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 10

/-- The interval [2, 5) -/
def I : Set ℝ := Set.Icc 2 5

theorem function_properties :
  (∃ (m : ℝ), m = 1 ∧ ∀ x ∈ I, f x ≥ m) ∧
  (¬∃ (M : ℝ), ∀ x ∈ I, f x ≤ M) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2834_283426


namespace NUMINAMATH_CALUDE_even_digits_in_base7_of_315_l2834_283498

/-- Converts a natural number from base 10 to base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of digits --/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a digit is even in base 7 --/
def isEvenInBase7 (digit : ℕ) : Bool :=
  sorry

theorem even_digits_in_base7_of_315 :
  let base7Repr := toBase7 315
  countEvenDigits (base7Repr.filter isEvenInBase7) = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_in_base7_of_315_l2834_283498


namespace NUMINAMATH_CALUDE_spider_dressing_8_pairs_l2834_283427

/-- The number of ways a spider can put on n pairs of socks and shoes -/
def spiderDressingWays (n : ℕ) : ℕ :=
  Nat.factorial (2 * n) / (2^n)

/-- Theorem: For 8 pairs of socks and shoes, the number of ways is 81729648000 -/
theorem spider_dressing_8_pairs :
  spiderDressingWays 8 = 81729648000 := by
  sorry

end NUMINAMATH_CALUDE_spider_dressing_8_pairs_l2834_283427


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_1_condition_l2834_283428

-- Define the condition for the square root to be meaningful
def is_meaningful (x : ℝ) : Prop := 2 * x - 1 ≥ 0

-- State the theorem
theorem sqrt_2x_minus_1_condition (x : ℝ) :
  is_meaningful x ↔ x ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_1_condition_l2834_283428


namespace NUMINAMATH_CALUDE_product_of_brackets_l2834_283486

def bracket_a (a : ℕ) : ℕ := a^2 + 3

def bracket_b (b : ℕ) : ℕ := 2*b - 4

theorem product_of_brackets (p q : ℕ) (h1 : p = 7) (h2 : q = 10) :
  bracket_a p * bracket_b q = 832 := by
  sorry

end NUMINAMATH_CALUDE_product_of_brackets_l2834_283486


namespace NUMINAMATH_CALUDE_duck_pond_problem_l2834_283447

theorem duck_pond_problem :
  let small_pond_total : ℕ := 30
  let small_pond_green_ratio : ℚ := 1/5
  let large_pond_green_ratio : ℚ := 3/25
  let total_green_ratio : ℚ := 3/20
  ∃ (large_pond_total : ℕ),
    (small_pond_green_ratio * small_pond_total + large_pond_green_ratio * large_pond_total : ℚ) = 
    total_green_ratio * (small_pond_total + large_pond_total) ∧
    large_pond_total = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_duck_pond_problem_l2834_283447


namespace NUMINAMATH_CALUDE_smallest_sum_is_14_l2834_283436

/-- Represents a pentagon arrangement of numbers 1 through 10 -/
structure PentagonArrangement where
  vertices : Fin 5 → Fin 10
  sides : Fin 5 → Fin 10
  all_used : ∀ n : Fin 10, (n ∈ Set.range vertices) ∨ (n ∈ Set.range sides)
  distinct : Function.Injective vertices ∧ Function.Injective sides

/-- The sum along each side of the pentagon -/
def side_sum (arr : PentagonArrangement) : ℕ → ℕ
| 0 => arr.vertices 0 + arr.sides 0 + arr.vertices 1
| 1 => arr.vertices 1 + arr.sides 1 + arr.vertices 2
| 2 => arr.vertices 2 + arr.sides 2 + arr.vertices 3
| 3 => arr.vertices 3 + arr.sides 3 + arr.vertices 4
| 4 => arr.vertices 4 + arr.sides 4 + arr.vertices 0
| _ => 0

/-- The arrangement is valid if all side sums are equal -/
def is_valid_arrangement (arr : PentagonArrangement) : Prop :=
  ∀ i j : Fin 5, side_sum arr i = side_sum arr j

/-- The main theorem: the smallest possible sum is 14 -/
theorem smallest_sum_is_14 :
  ∃ (arr : PentagonArrangement), is_valid_arrangement arr ∧
  (∀ i : Fin 5, side_sum arr i = 14) ∧
  (∀ arr' : PentagonArrangement, is_valid_arrangement arr' →
    ∀ i : Fin 5, side_sum arr' i ≥ 14) :=
  sorry

end NUMINAMATH_CALUDE_smallest_sum_is_14_l2834_283436


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l2834_283420

/-- Given a mixture of milk and water with an initial ratio of 4:1,
    adding 3 litres of water results in a new ratio of 3:1.
    This theorem proves that the initial volume of the mixture was 45 litres. -/
theorem initial_mixture_volume
  (initial_milk : ℝ)
  (initial_water : ℝ)
  (h1 : initial_milk / initial_water = 4)
  (h2 : initial_milk / (initial_water + 3) = 3) :
  initial_milk + initial_water = 45 := by
sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l2834_283420


namespace NUMINAMATH_CALUDE_tournament_balls_count_l2834_283417

def tournament_rounds : ℕ := 7

def games_per_round : List ℕ := [64, 32, 16, 8, 4, 2, 1]

def cans_per_game : ℕ := 6

def balls_per_can : ℕ := 4

def total_balls : ℕ := (games_per_round.sum * cans_per_game * balls_per_can)

theorem tournament_balls_count :
  total_balls = 3048 :=
by sorry

end NUMINAMATH_CALUDE_tournament_balls_count_l2834_283417


namespace NUMINAMATH_CALUDE_katy_summer_reading_l2834_283451

/-- The number of books Katy read in June -/
def june_books : ℕ := 8

/-- The number of books Katy read in July -/
def july_books : ℕ := 2 * june_books

/-- The number of books Katy read in August -/
def august_books : ℕ := july_books - 3

/-- The total number of books Katy read during the summer -/
def total_summer_books : ℕ := june_books + july_books + august_books

/-- Theorem stating that Katy read 37 books during the summer -/
theorem katy_summer_reading : total_summer_books = 37 := by
  sorry

end NUMINAMATH_CALUDE_katy_summer_reading_l2834_283451


namespace NUMINAMATH_CALUDE_symmetric_circle_correct_l2834_283483

/-- The equation of circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

/-- The equation of the symmetry line -/
def symmetry_line (x y : ℝ) : Prop := y = -x - 4

/-- The equation of the symmetric circle -/
def symmetric_circle (x y : ℝ) : Prop := (x + 4)^2 + (y + 6)^2 = 1

/-- Theorem stating that the given symmetric circle is correct -/
theorem symmetric_circle_correct :
  ∀ (x y : ℝ), (∃ (x₀ y₀ : ℝ), circle_C x₀ y₀ ∧ symmetry_line ((x + x₀)/2) ((y + y₀)/2)) →
  symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_symmetric_circle_correct_l2834_283483


namespace NUMINAMATH_CALUDE_min_abs_z_complex_l2834_283406

theorem min_abs_z_complex (z : ℂ) (h : Complex.abs (z - 3*I) + Complex.abs (z - 4) = 5) :
  ∃ (min_abs : ℝ), min_abs = 12/5 ∧ ∀ w : ℂ, Complex.abs (w - 3*I) + Complex.abs (w - 4) = 5 → Complex.abs w ≥ min_abs :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_complex_l2834_283406


namespace NUMINAMATH_CALUDE_photo_count_proof_l2834_283457

def final_photo_count (initial_photos : ℕ) (deleted_bad_shots : ℕ) (cat_photos : ℕ) (friend_photos : ℕ) (deleted_after_edit : ℕ) : ℕ :=
  initial_photos - deleted_bad_shots + cat_photos + friend_photos - deleted_after_edit

theorem photo_count_proof (x : ℕ) : 
  final_photo_count 63 7 15 x 3 = 68 + x := by
  sorry

end NUMINAMATH_CALUDE_photo_count_proof_l2834_283457


namespace NUMINAMATH_CALUDE_cost_price_percentage_l2834_283471

theorem cost_price_percentage (selling_price cost_price : ℝ) :
  selling_price > 0 →
  cost_price > 0 →
  selling_price - cost_price = (1 / 3) * cost_price →
  (cost_price / selling_price) * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l2834_283471


namespace NUMINAMATH_CALUDE_tv_screen_height_l2834_283404

/-- The height of a rectangular TV screen given its area and width -/
theorem tv_screen_height (area width : ℝ) (h_area : area = 21) (h_width : width = 3) :
  area / width = 7 := by
  sorry

end NUMINAMATH_CALUDE_tv_screen_height_l2834_283404


namespace NUMINAMATH_CALUDE_garage_spokes_count_l2834_283442

/-- The number of bicycles in the garage -/
def num_bicycles : ℕ := 4

/-- The number of wheels per bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of spokes per wheel -/
def spokes_per_wheel : ℕ := 10

/-- The total number of spokes in the garage -/
def total_spokes : ℕ := num_bicycles * wheels_per_bicycle * spokes_per_wheel

theorem garage_spokes_count : total_spokes = 80 := by
  sorry

end NUMINAMATH_CALUDE_garage_spokes_count_l2834_283442


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2834_283469

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2834_283469


namespace NUMINAMATH_CALUDE_overlapping_sticks_length_l2834_283459

/-- The total length of overlapping wooden sticks -/
def total_length (n : ℕ) (stick_length overlap : ℝ) : ℝ :=
  stick_length + (n - 1) * (stick_length - overlap)

/-- Theorem: The total length of 30 wooden sticks, each 25 cm long, 
    when overlapped by 6 cm, is equal to 576 cm -/
theorem overlapping_sticks_length :
  total_length 30 25 6 = 576 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_sticks_length_l2834_283459


namespace NUMINAMATH_CALUDE_sequence_property_l2834_283494

theorem sequence_property (a : ℕ → ℝ) (h_nonzero : ∀ n, a n ≠ 0) 
  (h_arith1 : a 2 - a 1 = a 3 - a 2)
  (h_geom : a 3 / a 2 = a 4 / a 3)
  (h_arith2 : 1 / a 4 - 1 / a 3 = 1 / a 5 - 1 / a 4) :
  a 3 ^ 2 = a 1 * a 5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l2834_283494


namespace NUMINAMATH_CALUDE_cafeteria_extra_apples_l2834_283492

/-- The number of extra apples in the cafeteria -/
def extra_apples (red_apples green_apples students_wanting_fruit : ℕ) : ℕ :=
  red_apples + green_apples - students_wanting_fruit

/-- Theorem: The cafeteria ends up with 73 extra apples -/
theorem cafeteria_extra_apples :
  ∀ (red_apples green_apples students_wanting_fruit : ℕ),
    red_apples = 43 →
    green_apples = 32 →
    students_wanting_fruit = 2 →
    extra_apples red_apples green_apples students_wanting_fruit = 73 :=
by
  sorry


end NUMINAMATH_CALUDE_cafeteria_extra_apples_l2834_283492


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2834_283493

theorem cube_root_equation_solution :
  ∃ y : ℝ, (5 - 2 / y)^(1/3 : ℝ) = -3 ↔ y = 1/16 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2834_283493


namespace NUMINAMATH_CALUDE_a_less_than_one_necessary_not_sufficient_for_a_squared_less_than_one_l2834_283466

theorem a_less_than_one_necessary_not_sufficient_for_a_squared_less_than_one :
  ∃ (a : ℝ), (a^2 < 1 → a < 1) ∧ ¬(a < 1 → a^2 < 1) :=
by sorry

end NUMINAMATH_CALUDE_a_less_than_one_necessary_not_sufficient_for_a_squared_less_than_one_l2834_283466


namespace NUMINAMATH_CALUDE_min_a_value_l2834_283446

/-- The minimum value of a that satisfies the given inequality for all positive x -/
theorem min_a_value (a : ℝ) : 
  (∀ x > 0, Real.log (2 * x) - (a * Real.exp x) / 2 ≤ Real.log a) → 
  a ≥ 2 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l2834_283446


namespace NUMINAMATH_CALUDE_avery_donation_total_l2834_283405

/-- Proves that the total number of clothes Avery donates is 16 -/
theorem avery_donation_total (shirts : ℕ) (pants : ℕ) (shorts : ℕ) : 
  shirts = 4 →
  pants = 2 * shirts →
  shorts = pants / 2 →
  shirts + pants + shorts = 16 := by
  sorry

end NUMINAMATH_CALUDE_avery_donation_total_l2834_283405


namespace NUMINAMATH_CALUDE_avg_f_value_l2834_283491

/-- A function that counts the number of multiples of p in the partial sums of a permutation -/
def f (p : ℕ) (π : Fin p → Fin p) : ℕ := sorry

/-- The average value of f over all permutations -/
def avg_f (p : ℕ) : ℚ := sorry

theorem avg_f_value (p : ℕ) (h : p.Prime) (h2 : p > 2) :
  avg_f p = 2 - 1 / p := by sorry

end NUMINAMATH_CALUDE_avg_f_value_l2834_283491


namespace NUMINAMATH_CALUDE_car_average_speed_l2834_283421

/-- Proves that the average speed of a car is 40 km/h given the specified conditions -/
theorem car_average_speed : ∀ (s : ℝ), s > 0 →
  ∃ (v : ℝ), v > 0 ∧
  (s / (s / (2 * (v + 30)) + s / (1.4 * v)) = v) ∧
  v = 40 := by
sorry

end NUMINAMATH_CALUDE_car_average_speed_l2834_283421


namespace NUMINAMATH_CALUDE_defined_implies_continuous_but_not_conversely_l2834_283468

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)
-- Define a point x₀
variable (x₀ : ℝ)

-- Statement: If f is defined at x₀, then f is continuous at x₀,
-- but the converse is not always true
theorem defined_implies_continuous_but_not_conversely :
  (∃ y, f x₀ = y) → ContinuousAt f x₀ ∧ 
  ¬(∀ g : ℝ → ℝ, ContinuousAt g x₀ → ∃ y, g x₀ = y) :=
sorry

end NUMINAMATH_CALUDE_defined_implies_continuous_but_not_conversely_l2834_283468


namespace NUMINAMATH_CALUDE_english_only_enrollment_l2834_283444

theorem english_only_enrollment (total : ℕ) (both : ℕ) (german : ℕ) 
  (h1 : total = 60)
  (h2 : both = 18)
  (h3 : german = 36)
  (h4 : total ≥ german)
  (h5 : german ≥ both) :
  total - (german - both) - both = 24 :=
by sorry

end NUMINAMATH_CALUDE_english_only_enrollment_l2834_283444


namespace NUMINAMATH_CALUDE_lcm_product_hcf_l2834_283479

theorem lcm_product_hcf (a b : ℕ+) (h1 : Nat.lcm a b = 750) (h2 : a * b = 18750) :
  Nat.gcd a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_lcm_product_hcf_l2834_283479


namespace NUMINAMATH_CALUDE_cube_root_sum_theorem_l2834_283425

theorem cube_root_sum_theorem :
  ∃ (x : ℝ), (x^(1/3) + (27 - x)^(1/3) = 3) ∧
  (∀ (y : ℝ), (y^(1/3) + (27 - y)^(1/3) = 3) → x ≤ y) →
  ∃ (r s : ℤ), (x = r - Real.sqrt s) ∧ (r + s = 0) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sum_theorem_l2834_283425


namespace NUMINAMATH_CALUDE_cupcake_price_is_one_fifty_l2834_283482

/-- Represents the daily production and prices of bakery items -/
structure BakeryProduction where
  cupcakes_per_day : ℕ
  cookie_packets_per_day : ℕ
  biscuit_packets_per_day : ℕ
  cookie_price_per_packet : ℚ
  biscuit_price_per_packet : ℚ

/-- Calculates the price of a cupcake given the bakery production and total earnings -/
def calculate_cupcake_price (prod : BakeryProduction) (days : ℕ) (total_earnings : ℚ) : ℚ :=
  let total_cookies_earnings := prod.cookie_packets_per_day * days * prod.cookie_price_per_packet
  let total_biscuits_earnings := prod.biscuit_packets_per_day * days * prod.biscuit_price_per_packet
  let cupcakes_earnings := total_earnings - total_cookies_earnings - total_biscuits_earnings
  cupcakes_earnings / (prod.cupcakes_per_day * days)

/-- Theorem stating that the cupcake price is $1.50 given the specified conditions -/
theorem cupcake_price_is_one_fifty :
  let prod : BakeryProduction := {
    cupcakes_per_day := 20,
    cookie_packets_per_day := 10,
    biscuit_packets_per_day := 20,
    cookie_price_per_packet := 2,
    biscuit_price_per_packet := 1
  }
  let days : ℕ := 5
  let total_earnings : ℚ := 350
  calculate_cupcake_price prod days total_earnings = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_cupcake_price_is_one_fifty_l2834_283482


namespace NUMINAMATH_CALUDE_donated_area_is_108_45_l2834_283435

/-- Calculates the total area of cloth donated given the areas and percentages of three cloths. -/
def total_donated_area (cloth1_area cloth2_area cloth3_area : ℝ)
  (cloth1_keep_percent cloth2_keep_percent cloth3_keep_percent : ℝ) : ℝ :=
  let cloth1_donate := cloth1_area * (1 - cloth1_keep_percent)
  let cloth2_donate := cloth2_area * (1 - cloth2_keep_percent)
  let cloth3_donate := cloth3_area * (1 - cloth3_keep_percent)
  cloth1_donate + cloth2_donate + cloth3_donate

/-- Theorem stating that the total donated area is 108.45 square inches. -/
theorem donated_area_is_108_45 :
  total_donated_area 100 65 48 0.4 0.55 0.6 = 108.45 := by
  sorry

end NUMINAMATH_CALUDE_donated_area_is_108_45_l2834_283435


namespace NUMINAMATH_CALUDE_triangle_side_length_l2834_283430

/-- An equilateral triangle divided into three congruent trapezoids -/
structure TriangleDivision where
  /-- The side length of the equilateral triangle -/
  triangle_side : ℝ
  /-- The length of the shorter base of each trapezoid -/
  trapezoid_short_base : ℝ
  /-- The length of the longer base of each trapezoid -/
  trapezoid_long_base : ℝ
  /-- The length of the legs of each trapezoid -/
  trapezoid_leg : ℝ
  /-- The trapezoids are congruent -/
  congruent_trapezoids : trapezoid_long_base = 2 * trapezoid_short_base
  /-- The triangle is divided into three trapezoids -/
  triangle_composition : triangle_side = trapezoid_short_base + 2 * trapezoid_leg
  /-- The perimeter of each trapezoid is 10 + 5√3 -/
  trapezoid_perimeter : trapezoid_short_base + trapezoid_long_base + 2 * trapezoid_leg = 10 + 5 * Real.sqrt 3

/-- Theorem: The side length of the equilateral triangle is 6 + 3√3 -/
theorem triangle_side_length (td : TriangleDivision) : td.triangle_side = 6 + 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2834_283430


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l2834_283484

theorem complex_exponential_sum (α β : ℝ) : 
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1/4 : ℂ) + (3/7 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1/4 : ℂ) - (3/7 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l2834_283484


namespace NUMINAMATH_CALUDE_minsu_age_proof_l2834_283485

/-- Minsu's current age in years -/
def minsu_current_age : ℕ := 8

/-- Years in the future when Minsu's age will be four times his current age -/
def years_in_future : ℕ := 24

/-- Theorem stating that Minsu's current age is 8, given the condition -/
theorem minsu_age_proof :
  minsu_current_age = 8 ∧
  minsu_current_age + years_in_future = 4 * minsu_current_age :=
by sorry

end NUMINAMATH_CALUDE_minsu_age_proof_l2834_283485


namespace NUMINAMATH_CALUDE_complex_magnitude_l2834_283449

theorem complex_magnitude (z : ℂ) : z = (2 - Complex.I) / Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2834_283449


namespace NUMINAMATH_CALUDE_equation_solution_l2834_283478

theorem equation_solution (x y k z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0)
  (h : 1/x + 1/y = k/z) : z = x*y / (k*(y+x)) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2834_283478


namespace NUMINAMATH_CALUDE_number_relationship_l2834_283450

theorem number_relationship (a b c d : ℝ) 
  (h1 : a < b) 
  (h2 : d < c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b :=
by sorry

end NUMINAMATH_CALUDE_number_relationship_l2834_283450


namespace NUMINAMATH_CALUDE_power_of_two_triples_l2834_283443

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def satisfies_condition (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  is_power_of_two (a * b - c) ∧
  is_power_of_two (b * c - a) ∧
  is_power_of_two (c * a - b)

theorem power_of_two_triples :
  ∀ a b c : ℕ, satisfies_condition a b c ↔
    ((a, b, c) = (2, 2, 2) ∨
     (a, b, c) = (2, 2, 3) ∨
     (a, b, c) = (3, 5, 7) ∨
     (a, b, c) = (2, 6, 11)) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_triples_l2834_283443


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l2834_283453

theorem imaginary_power_sum : Complex.I ^ 22 + Complex.I ^ 222 = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l2834_283453


namespace NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l2834_283481

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8195 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l2834_283481


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2834_283407

theorem smallest_number_divisible (n : ℕ) : n = 257 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) % 8 = 0 ∧ (m + 7) % 11 = 0 ∧ (m + 7) % 24 = 0)) ∧
  (n + 7) % 8 = 0 ∧ (n + 7) % 11 = 0 ∧ (n + 7) % 24 = 0 :=
by sorry

#check smallest_number_divisible

end NUMINAMATH_CALUDE_smallest_number_divisible_l2834_283407


namespace NUMINAMATH_CALUDE_publishing_profit_inequality_minimum_sets_answer_is_four_thousand_l2834_283456

/-- Represents the number of thousands of sets -/
def x : ℝ := 4

/-- Fixed cost in yuan -/
def fixed_cost : ℝ := 80000

/-- Cost increase per set in yuan -/
def cost_increase : ℝ := 20

/-- Price per set in yuan -/
def price : ℝ := 100

/-- Underwriter's share of sales -/
def underwriter_share : ℝ := 0.3

/-- Publishing house's desired profit margin -/
def profit_margin : ℝ := 0.1

/-- The inequality that must be satisfied for the publishing house to achieve its desired profit -/
theorem publishing_profit_inequality :
  fixed_cost + cost_increase * 1000 * x ≤ price * (1 - underwriter_share - profit_margin) * 1000 * x :=
sorry

/-- The minimum number of sets (in thousands) that satisfies the inequality -/
theorem minimum_sets :
  x = ⌈(fixed_cost / (price * (1 - underwriter_share - profit_margin) * 1000 - cost_increase * 1000))⌉ :=
sorry

/-- Proof that 4,000 sets is the correct answer when rounded to the nearest thousand -/
theorem answer_is_four_thousand :
  ⌊x * 1000 / 1000 + 0.5⌋ * 1000 = 4000 :=
sorry

end NUMINAMATH_CALUDE_publishing_profit_inequality_minimum_sets_answer_is_four_thousand_l2834_283456


namespace NUMINAMATH_CALUDE_plane_relationships_l2834_283423

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (in_plane : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- State the theorem
theorem plane_relationships 
  (α β : Plane) 
  (h_different : α ≠ β) :
  (∀ l : Line, in_plane l α → 
    (∀ m : Line, in_plane m β → perpendicular l m) → 
    plane_perpendicular α β) ∧
  ((∀ l : Line, in_plane l α → line_parallel_to_plane l β) → 
    plane_parallel α β) ∧
  (plane_parallel α β → 
    ∀ l : Line, in_plane l α → line_parallel_to_plane l β) :=
sorry

end NUMINAMATH_CALUDE_plane_relationships_l2834_283423


namespace NUMINAMATH_CALUDE_two_digit_number_theorem_l2834_283415

/-- Represents a two-digit number with specific properties -/
structure TwoDigitNumber where
  x : ℕ  -- tens digit
  -- Ensure x is a single digit
  h1 : x ≥ 1 ∧ x ≤ 9
  -- Ensure the units digit is non-negative
  h2 : 2 * x ≥ 3

/-- The value of the two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : ℕ :=
  10 * n.x + (2 * n.x - 3)

theorem two_digit_number_theorem (n : TwoDigitNumber) :
  n.value = 12 * n.x - 3 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_theorem_l2834_283415


namespace NUMINAMATH_CALUDE_shortest_leg_of_smallest_triangle_l2834_283401

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  short_leg : ℝ
  long_leg : ℝ
  hypotenuse : ℝ
  short_leg_prop : short_leg = hypotenuse / 2
  long_leg_prop : long_leg = short_leg * Real.sqrt 3

/-- Represents a series of three 30-60-90 triangles -/
structure TriangleSeries where
  large : Triangle30_60_90
  medium : Triangle30_60_90
  small : Triangle30_60_90
  large_medium_relation : large.short_leg = medium.hypotenuse
  medium_small_relation : medium.short_leg = small.hypotenuse
  largest_hypotenuse : large.hypotenuse = 12

theorem shortest_leg_of_smallest_triangle (series : TriangleSeries) :
  series.small.short_leg = 1.5 := by sorry

end NUMINAMATH_CALUDE_shortest_leg_of_smallest_triangle_l2834_283401


namespace NUMINAMATH_CALUDE_cat_groupings_count_l2834_283475

/-- The number of ways to divide 12 cats into groups of 4, 6, and 2,
    with Whiskers in the 4-cat group and Paws in the 6-cat group. -/
def cat_groupings : ℕ :=
  Nat.choose 10 3 * Nat.choose 7 5

theorem cat_groupings_count : cat_groupings = 2520 := by
  sorry

end NUMINAMATH_CALUDE_cat_groupings_count_l2834_283475


namespace NUMINAMATH_CALUDE_vikas_questions_l2834_283418

theorem vikas_questions (total : ℕ) (r v a : ℕ) : 
  total = 24 →
  r + v + a = total →
  7 * v = 3 * r →
  3 * a = 2 * v →
  v = 6 := by
sorry

end NUMINAMATH_CALUDE_vikas_questions_l2834_283418


namespace NUMINAMATH_CALUDE_marys_cake_recipe_l2834_283463

/-- Mary's cake recipe problem -/
theorem marys_cake_recipe 
  (total_flour : ℕ) 
  (sugar : ℕ) 
  (flour_to_add : ℕ) 
  (h1 : total_flour = 9)
  (h2 : flour_to_add = sugar + 1)
  (h3 : sugar = 6) :
  total_flour - flour_to_add = 2 := by
  sorry

end NUMINAMATH_CALUDE_marys_cake_recipe_l2834_283463


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l2834_283431

theorem sum_of_roots_quadratic_equation :
  let a : ℝ := -3
  let b : ℝ := -27
  let c : ℝ := 81
  let equation := fun x : ℝ => a * x^2 + b * x + c
  ∃ r s : ℝ, equation r = 0 ∧ equation s = 0 ∧ r + s = -b / a :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l2834_283431


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2834_283429

/-- Given a > 0 and a ≠ 1, prove that the function f(x) = a^(x+2) - 2 
    always passes through the point (-2, -1) regardless of the value of a -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2) - 2
  f (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2834_283429


namespace NUMINAMATH_CALUDE_third_derivative_x5_minus_7x3_plus_2_l2834_283419

/-- The third derivative of x^5 - 7x^3 + 2 is 60x^2 - 42 -/
theorem third_derivative_x5_minus_7x3_plus_2 (x : ℝ) :
  (deriv^[3] (fun x => x^5 - 7*x^3 + 2)) x = 60*x^2 - 42 := by
  sorry

end NUMINAMATH_CALUDE_third_derivative_x5_minus_7x3_plus_2_l2834_283419


namespace NUMINAMATH_CALUDE_logan_desired_amount_left_l2834_283499

/-- Represents Logan's financial situation and goal --/
structure LoganFinances where
  current_income : ℕ
  rent_expense : ℕ
  groceries_expense : ℕ
  gas_expense : ℕ
  income_increase : ℕ

/-- Calculates the desired amount left each year for Logan --/
def desired_amount_left (f : LoganFinances) : ℕ :=
  (f.current_income + f.income_increase) - (f.rent_expense + f.groceries_expense + f.gas_expense)

/-- Theorem stating the desired amount left each year for Logan --/
theorem logan_desired_amount_left :
  let f : LoganFinances := {
    current_income := 65000,
    rent_expense := 20000,
    groceries_expense := 5000,
    gas_expense := 8000,
    income_increase := 10000
  }
  desired_amount_left f = 42000 := by
  sorry


end NUMINAMATH_CALUDE_logan_desired_amount_left_l2834_283499


namespace NUMINAMATH_CALUDE_football_season_games_l2834_283480

/-- The number of months in the football season -/
def season_months : ℕ := 17

/-- The number of games played each month -/
def games_per_month : ℕ := 19

/-- The total number of games played during the season -/
def total_games : ℕ := season_months * games_per_month

theorem football_season_games :
  total_games = 323 :=
by sorry

end NUMINAMATH_CALUDE_football_season_games_l2834_283480


namespace NUMINAMATH_CALUDE_f_max_value_l2834_283473

/-- The function f(x) = 6x - 2x^2 -/
def f (x : ℝ) := 6 * x - 2 * x^2

/-- The maximum value of f(x) is 9/2 -/
theorem f_max_value : ∃ (M : ℝ), M = 9/2 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l2834_283473


namespace NUMINAMATH_CALUDE_function_composition_l2834_283416

theorem function_composition (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 3 / (3 + x)) :
  ∀ x > 0, 2 * f x = 18 / (9 + x) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l2834_283416


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2834_283410

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : y - x = -1) 
  (h2 : x * y = 2) : 
  -2 * x^3 * y + 4 * x^2 * y^2 - 2 * x * y^3 = -4 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2834_283410


namespace NUMINAMATH_CALUDE_divide_number_with_percentage_condition_l2834_283489

theorem divide_number_with_percentage_condition : 
  ∃ (x : ℝ), 
    x + (80 - x) = 80 ∧ 
    0.3 * x = 0.2 * (80 - x) + 10 ∧ 
    min x (80 - x) = 28 := by
  sorry

end NUMINAMATH_CALUDE_divide_number_with_percentage_condition_l2834_283489


namespace NUMINAMATH_CALUDE_max_distance_is_three_l2834_283402

/-- A figure constructed from an equilateral triangle with semicircles on each side -/
structure TriangleWithSemicircles where
  /-- Side length of the equilateral triangle -/
  triangleSide : ℝ
  /-- Radius of the semicircles -/
  semicircleRadius : ℝ

/-- The maximum distance between any two points on the boundary of the figure -/
def maxBoundaryDistance (figure : TriangleWithSemicircles) : ℝ :=
  figure.triangleSide + 2 * figure.semicircleRadius

/-- Theorem stating the maximum distance for the specific figure described in the problem -/
theorem max_distance_is_three :
  let figure : TriangleWithSemicircles := ⟨2, 1⟩
  maxBoundaryDistance figure = 3 := by
  sorry


end NUMINAMATH_CALUDE_max_distance_is_three_l2834_283402


namespace NUMINAMATH_CALUDE_quadratic_symmetry_axis_l2834_283434

/-- A quadratic function passing through points (-4,m) and (2,m) has its axis of symmetry at x = -1 -/
theorem quadratic_symmetry_axis (f : ℝ → ℝ) (m : ℝ) : 
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →  -- f is a quadratic function
  f (-4) = m →                                    -- f passes through (-4,m)
  f 2 = m →                                       -- f passes through (2,m)
  (∀ x, f (x - 1) = f (-x - 1)) :=                -- axis of symmetry is x = -1
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_axis_l2834_283434

import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_l1966_196685

-- Define the inequality function
def inequality (x : ℝ) : Prop :=
  9.216 * (Real.log x / Real.log 5) + (Real.log x - Real.log 3) / (Real.log x)
  < ((Real.log x / Real.log 5) * (2 - Real.log x / Real.log 3)) / (Real.log x / Real.log 3)

-- State the theorem
theorem inequality_solution :
  ∀ x : ℝ, 
  x > 0 → 
  inequality x ↔ (0 < x ∧ x < 1 / Real.sqrt 5) ∨ (1 < x ∧ x < 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1966_196685


namespace NUMINAMATH_CALUDE_sum_of_four_cubes_equals_three_l1966_196667

theorem sum_of_four_cubes_equals_three (k : ℤ) :
  (1 + 6 * k^3)^3 + (1 - 6 * k^3)^3 + (-6 * k^2)^3 + 1^3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_cubes_equals_three_l1966_196667


namespace NUMINAMATH_CALUDE_workers_wage_before_promotion_l1966_196674

theorem workers_wage_before_promotion (wage_increase_percentage : ℝ) (new_wage : ℝ) : 
  wage_increase_percentage = 0.60 →
  new_wage = 45 →
  (1 + wage_increase_percentage) * (new_wage / (1 + wage_increase_percentage)) = 28.125 := by
sorry

end NUMINAMATH_CALUDE_workers_wage_before_promotion_l1966_196674


namespace NUMINAMATH_CALUDE_sum_of_divisors_360_l1966_196686

/-- The sum of the positive whole number divisors of 360 is 1170. -/
theorem sum_of_divisors_360 : (Finset.filter (· ∣ 360) (Finset.range 361)).sum id = 1170 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_360_l1966_196686


namespace NUMINAMATH_CALUDE_line_slope_45_degrees_l1966_196605

theorem line_slope_45_degrees (m : ℝ) : 
  let P : ℝ × ℝ := (-2, m)
  let Q : ℝ × ℝ := (m, 4)
  (4 - m) / (m - (-2)) = 1 → m = 1 := by
sorry

end NUMINAMATH_CALUDE_line_slope_45_degrees_l1966_196605


namespace NUMINAMATH_CALUDE_minimum_discount_for_profit_margin_l1966_196615

theorem minimum_discount_for_profit_margin 
  (cost_price : ℝ) 
  (marked_price : ℝ) 
  (min_profit_margin : ℝ) 
  (discount : ℝ) :
  cost_price = 800 →
  marked_price = 1200 →
  min_profit_margin = 0.2 →
  discount = 0.08 →
  marked_price * (1 - discount) ≥ cost_price * (1 + min_profit_margin) ∧
  ∀ d : ℝ, d < discount → marked_price * (1 - d) < cost_price * (1 + min_profit_margin) :=
by sorry

end NUMINAMATH_CALUDE_minimum_discount_for_profit_margin_l1966_196615


namespace NUMINAMATH_CALUDE_final_result_l1966_196624

def program_result : ℕ → ℕ → ℕ
| 0, s => s
| (n+1), s => program_result n (s * (11 - n))

theorem final_result : program_result 3 1 = 990 := by
  sorry

#eval program_result 3 1

end NUMINAMATH_CALUDE_final_result_l1966_196624


namespace NUMINAMATH_CALUDE_distance_is_sqrt_6_l1966_196651

def A : ℝ × ℝ × ℝ := (1, -1, -1)
def P : ℝ × ℝ × ℝ := (1, 1, 1)
def direction_vector : ℝ × ℝ × ℝ := (1, 0, -1)

def distance_point_to_line (P : ℝ × ℝ × ℝ) (A : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_sqrt_6 :
  distance_point_to_line P A direction_vector = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_sqrt_6_l1966_196651


namespace NUMINAMATH_CALUDE_larger_number_of_pair_l1966_196610

theorem larger_number_of_pair (x y : ℝ) (h1 : x + y = 29) (h2 : x - y = 5) : 
  max x y = 17 := by
sorry

end NUMINAMATH_CALUDE_larger_number_of_pair_l1966_196610


namespace NUMINAMATH_CALUDE_milk_container_problem_l1966_196642

-- Define the capacity of container A
def A : ℝ := 1184

-- Define the quantity of milk in container B after initial pouring
def B : ℝ := 0.375 * A

-- Define the quantity of milk in container C after initial pouring
def C : ℝ := 0.625 * A

-- Define the amount transferred from C to B
def transfer : ℝ := 148

-- Theorem statement
theorem milk_container_problem :
  -- After transfer, B and C have equal quantities
  B + transfer = C - transfer ∧
  -- The sum of B and C equals A
  B + C = A ∧
  -- A is 1184 liters
  A = 1184 := by
  sorry

end NUMINAMATH_CALUDE_milk_container_problem_l1966_196642


namespace NUMINAMATH_CALUDE_total_amount_paid_l1966_196697

def grape_quantity : ℕ := 3
def grape_rate : ℕ := 70
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 55

theorem total_amount_paid : 
  grape_quantity * grape_rate + mango_quantity * mango_rate = 705 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l1966_196697


namespace NUMINAMATH_CALUDE_complement_of_A_l1966_196676

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {3, 4, 5}

-- Theorem statement
theorem complement_of_A : 
  (U \ A) = {1, 2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1966_196676


namespace NUMINAMATH_CALUDE_log_sum_simplification_l1966_196616

theorem log_sum_simplification : 
  ∀ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 →
  (1 / (Real.log x / Real.log 12 + 1)) + 
  (1 / (Real.log y / Real.log 20 + 1)) + 
  (1 / (Real.log z / Real.log 8 + 1)) = 1.75 :=
by sorry

end NUMINAMATH_CALUDE_log_sum_simplification_l1966_196616


namespace NUMINAMATH_CALUDE_base5_to_base7_conversion_l1966_196660

/-- Converts a number from base 5 to decimal --/
def base5ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from decimal to base 7 --/
def decimalToBase7 (n : ℕ) : ℕ := sorry

theorem base5_to_base7_conversion :
  decimalToBase7 (base5ToDecimal 412) = 212 := by sorry

end NUMINAMATH_CALUDE_base5_to_base7_conversion_l1966_196660


namespace NUMINAMATH_CALUDE_total_registration_methods_l1966_196682

-- Define the number of students and clubs
def num_students : Nat := 5
def num_clubs : Nat := 3

-- Define the students with restrictions
structure RestrictedStudent where
  name : String
  restricted_club : Nat

-- Define the list of restricted students
def restricted_students : List RestrictedStudent := [
  { name := "Xiao Bin", restricted_club := 1 },  -- 1 represents chess club
  { name := "Xiao Cong", restricted_club := 0 }, -- 0 represents basketball club
  { name := "Xiao Hao", restricted_club := 2 }   -- 2 represents environmental club
]

-- Define the theorem
theorem total_registration_methods :
  (restricted_students.length * 2 + (num_students - restricted_students.length) * num_clubs) ^ num_students = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_registration_methods_l1966_196682


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1966_196675

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := sorry
def focus2 : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the line passing through F1, A, and B
def line_passes_through (p q r : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter :
  ellipse A.1 A.2 →
  ellipse B.1 B.2 →
  line_passes_through focus1 A B →
  (A.1 - focus1.1)^2 + (A.2 - focus1.2)^2 + 
  (A.1 - focus2.1)^2 + (A.2 - focus2.2)^2 = 16 →
  (B.1 - focus1.1)^2 + (B.2 - focus1.2)^2 + 
  (B.1 - focus2.1)^2 + (B.2 - focus2.2)^2 = 16 →
  let perimeter := 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
    Real.sqrt ((A.1 - focus2.1)^2 + (A.2 - focus2.2)^2) +
    Real.sqrt ((B.1 - focus2.1)^2 + (B.2 - focus2.2)^2)
  perimeter = 8 := by sorry


end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l1966_196675


namespace NUMINAMATH_CALUDE_range_of_f_l1966_196679

noncomputable def f (m : ℝ) (α β : ℝ) : ℝ := 5 * m^2 + 3 * m * Real.tan (α + β) + 4

theorem range_of_f :
  ∀ m α β : ℝ,
  (2 * m * (Real.tan α)^2 + (4 * m - 2) * Real.tan α + 2 * m - 3 = 0) →
  (2 * m * (Real.tan β)^2 + (4 * m - 2) * Real.tan β + 2 * m - 3 = 0) →
  Real.tan α ≠ Real.tan β →
  ∃ y : ℝ, y ∈ Set.Ioo (13/4) 4 ∪ Set.Ioi 4 ↔ ∃ m, f m α β = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l1966_196679


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l1966_196694

/-- Rectangle represented by its side lengths -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Square represented by its side length -/
structure Square where
  side : ℝ

/-- The overlapping area between a rectangle and a square -/
def overlap_area (r : Rectangle) (s : Square) : ℝ := sorry

theorem rectangle_square_overlap_ratio :
  ∀ (r : Rectangle) (s : Square),
    overlap_area r s = 0.4 * r.length * r.width →
    overlap_area r s = 0.25 * s.side * s.side →
    r.length / r.width = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l1966_196694


namespace NUMINAMATH_CALUDE_ordered_pairs_theorem_l1966_196645

def S : Set (ℕ × ℕ) := {(8, 4), (9, 3), (2, 1)}

def satisfies_conditions (pair : ℕ × ℕ) : Prop :=
  let (x, y) := pair
  x > y ∧ (x - y = 2 * x / y ∨ x - y = 2 * y / x)

theorem ordered_pairs_theorem :
  ∀ (pair : ℕ × ℕ), pair ∈ S ↔ satisfies_conditions pair ∧ pair.1 > 0 ∧ pair.2 > 0 :=
sorry

end NUMINAMATH_CALUDE_ordered_pairs_theorem_l1966_196645


namespace NUMINAMATH_CALUDE_square_root_sum_l1966_196628

theorem square_root_sum (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_l1966_196628


namespace NUMINAMATH_CALUDE_right_triangle_from_medians_l1966_196623

theorem right_triangle_from_medians (m₁ m₂ m₃ : ℝ) 
  (h₁ : m₁ = 5)
  (h₂ : m₂ = Real.sqrt 52)
  (h₃ : m₃ = Real.sqrt 73) :
  ∃ a b c : ℝ, a^2 = b^2 + c^2 ∧ 
    m₁^2 = (2*b^2 + 2*c^2 - a^2) / 4 ∧
    m₂^2 = (2*a^2 + 2*c^2 - b^2) / 4 ∧
    m₃^2 = (2*a^2 + 2*b^2 - c^2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_from_medians_l1966_196623


namespace NUMINAMATH_CALUDE_cylinder_cross_section_angle_l1966_196647

/-- Given a cylinder cut by a plane where the cross-section has an eccentricity of 2√2/3,
    the acute dihedral angle between this cross-section and the cylinder's base is arccos(1/3). -/
theorem cylinder_cross_section_angle (e : ℝ) (θ : ℝ) : 
  e = 2 * Real.sqrt 2 / 3 →
  θ = Real.arccos (1/3) →
  θ = Real.arccos (Real.sqrt (1 - e^2)) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_angle_l1966_196647


namespace NUMINAMATH_CALUDE_number_of_laborers_l1966_196693

/-- Proves that the number of laborers is 24 given the salary information --/
theorem number_of_laborers (total_avg : ℝ) (num_supervisors : ℕ) (supervisor_avg : ℝ) (laborer_avg : ℝ) :
  total_avg = 1250 →
  num_supervisors = 6 →
  supervisor_avg = 2450 →
  laborer_avg = 950 →
  ∃ (num_laborers : ℕ), 
    (num_laborers : ℝ) * laborer_avg + (num_supervisors : ℝ) * supervisor_avg = 
    (num_laborers + num_supervisors : ℝ) * total_avg ∧
    num_laborers = 24 :=
by sorry

end NUMINAMATH_CALUDE_number_of_laborers_l1966_196693


namespace NUMINAMATH_CALUDE_new_person_weight_l1966_196607

theorem new_person_weight 
  (n : ℕ) 
  (initial_weight : ℝ) 
  (weight_increase : ℝ) 
  (replaced_weight : ℝ) :
  n = 9 →
  initial_weight = 65 →
  weight_increase = 2.5 →
  replaced_weight = 65 →
  (n : ℝ) * weight_increase + replaced_weight = 87.5 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1966_196607


namespace NUMINAMATH_CALUDE_baylor_payment_multiple_l1966_196622

theorem baylor_payment_multiple :
  let initial_amount : ℕ := 4000
  let first_client_payment : ℕ := initial_amount / 2
  let second_client_payment : ℕ := first_client_payment + (2 * first_client_payment) / 5
  let combined_payment : ℕ := first_client_payment + second_client_payment
  let final_total : ℕ := 18400
  let third_client_multiple : ℕ := (final_total - initial_amount - combined_payment) / combined_payment
  third_client_multiple = 2 := by sorry

end NUMINAMATH_CALUDE_baylor_payment_multiple_l1966_196622


namespace NUMINAMATH_CALUDE_textbook_packing_probability_l1966_196629

/-- Represents the problem of packing textbooks into boxes -/
structure TextbookPacking where
  total_books : Nat
  math_books : Nat
  box_sizes : Finset Nat

/-- The probability of all math books ending up in the same box -/
def probability_all_math_in_same_box (p : TextbookPacking) : ℚ :=
  sorry

/-- The main theorem stating the probability for the given problem -/
theorem textbook_packing_probability :
  let p := TextbookPacking.mk 15 4 {4, 5, 6}
  probability_all_math_in_same_box p = 27 / 1759 :=
sorry

end NUMINAMATH_CALUDE_textbook_packing_probability_l1966_196629


namespace NUMINAMATH_CALUDE_perimeter_ABCDE_l1966_196695

-- Define the points as 2D vectors
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom AB_eq_BC : dist A B = dist B C
axiom AB_eq_4 : dist A B = 4
axiom AE_eq_5 : dist A E = 5
axiom ED_eq_8 : dist E D = 8
axiom right_angle_AEB : (B.1 - E.1) * (A.2 - E.2) = (A.1 - E.1) * (B.2 - E.2)
axiom right_angle_BAE : (E.1 - A.1) * (B.2 - A.2) = (B.1 - A.1) * (E.2 - A.2)
axiom right_angle_ABC : (C.1 - B.1) * (A.2 - B.2) = (A.1 - B.1) * (C.2 - B.2)

-- Define the perimeter function
def perimeter (A B C D E : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist C D + dist D E + dist E A

-- State the theorem
theorem perimeter_ABCDE :
  perimeter A B C D E = 21 + Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_perimeter_ABCDE_l1966_196695


namespace NUMINAMATH_CALUDE_symmetry_complex_plane_l1966_196662

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal -/
def symmetric_to_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem symmetry_complex_plane (z₁ z₂ : ℂ) :
  symmetric_to_imaginary_axis z₁ z₂ → z₁ = 1 + I → z₂ = -1 + I := by
  sorry

#check symmetry_complex_plane

end NUMINAMATH_CALUDE_symmetry_complex_plane_l1966_196662


namespace NUMINAMATH_CALUDE_consecutive_sum_l1966_196680

theorem consecutive_sum (n : ℤ) : 
  (∃ (x : ℤ), x = n ∧ x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 210) → n = 40 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_sum_l1966_196680


namespace NUMINAMATH_CALUDE_football_players_count_l1966_196689

theorem football_players_count (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 40)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 11) :
  ∃ football : ℕ, football = 26 ∧ 
    football + tennis - both = total - neither :=
by sorry

end NUMINAMATH_CALUDE_football_players_count_l1966_196689


namespace NUMINAMATH_CALUDE_article_pricing_gain_percent_l1966_196668

/-- Proves that if the cost price of 50 articles equals the selling price of 35 articles,
    then the gain percent is 300/7. -/
theorem article_pricing_gain_percent
  (C : ℝ) -- Cost price of one article
  (S : ℝ) -- Selling price of one article
  (h : 50 * C = 35 * S) -- Given condition
  : (S - C) / C * 100 = 300 / 7 := by
  sorry

end NUMINAMATH_CALUDE_article_pricing_gain_percent_l1966_196668


namespace NUMINAMATH_CALUDE_twenty_four_is_eighty_percent_of_thirty_l1966_196626

theorem twenty_four_is_eighty_percent_of_thirty : 
  ∃ x : ℝ, 24 = 0.8 * x ∧ x = 30 := by
sorry

end NUMINAMATH_CALUDE_twenty_four_is_eighty_percent_of_thirty_l1966_196626


namespace NUMINAMATH_CALUDE_danivan_initial_inventory_l1966_196633

/-- Represents the inventory and sales data for Danivan Drugstore --/
structure DrugstoreData where
  monday_sales : ℕ
  tuesday_sales : ℕ
  daily_sales_wed_to_sun : ℕ
  saturday_delivery : ℕ
  end_of_week_inventory : ℕ

/-- Calculates the initial inventory of hand sanitizer gel bottles --/
def initial_inventory (data : DrugstoreData) : ℕ :=
  data.end_of_week_inventory + 
  data.monday_sales + 
  data.tuesday_sales + 
  (5 * data.daily_sales_wed_to_sun) - 
  data.saturday_delivery

/-- Theorem stating that the initial inventory is 4500 bottles --/
theorem danivan_initial_inventory : 
  initial_inventory {
    monday_sales := 2445,
    tuesday_sales := 900,
    daily_sales_wed_to_sun := 50,
    saturday_delivery := 650,
    end_of_week_inventory := 1555
  } = 4500 := by
  sorry


end NUMINAMATH_CALUDE_danivan_initial_inventory_l1966_196633


namespace NUMINAMATH_CALUDE_caviar_cost_calculation_l1966_196644

/-- The cost of caviar per person for Alex's New Year's Eve appetizer -/
def caviar_cost (chips_cost creme_fraiche_cost total_cost : ℚ) : ℚ :=
  total_cost - (chips_cost + creme_fraiche_cost)

/-- Theorem stating the cost of caviar per person -/
theorem caviar_cost_calculation :
  caviar_cost 3 5 27 = 19 := by
  sorry

end NUMINAMATH_CALUDE_caviar_cost_calculation_l1966_196644


namespace NUMINAMATH_CALUDE_cost_of_treat_l1966_196600

/-- The cost of dog treats given daily treats, days, and total cost -/
def treat_cost (treats_per_day : ℕ) (days : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (treats_per_day * days)

/-- Theorem: The cost of each treat is $0.10 given the problem conditions -/
theorem cost_of_treat :
  let treats_per_day : ℕ := 2
  let days : ℕ := 30
  let total_cost : ℚ := 6
  treat_cost treats_per_day days total_cost = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_treat_l1966_196600


namespace NUMINAMATH_CALUDE_ball_distribution_after_four_rounds_l1966_196621

/-- Represents the state of the game at any point -/
structure GameState :=
  (a b c d e : ℕ)

/-- Represents a single round of the game -/
def gameRound (s : GameState) : GameState :=
  let a' := if s.e < s.a then s.a - 2 else s.a
  let b' := if s.a < s.b then s.b - 2 else s.b
  let c' := if s.b < s.c then s.c - 2 else s.c
  let d' := if s.c < s.d then s.d - 2 else s.d
  let e' := if s.d < s.e then s.e - 2 else s.e
  ⟨a', b', c', d', e'⟩

/-- Represents the initial state of the game -/
def initialState : GameState := ⟨2, 4, 6, 8, 10⟩

/-- Represents the state after 4 rounds -/
def finalState : GameState := (gameRound ∘ gameRound ∘ gameRound ∘ gameRound) initialState

/-- The main theorem to be proved -/
theorem ball_distribution_after_four_rounds :
  finalState = ⟨6, 6, 6, 6, 6⟩ := by sorry

end NUMINAMATH_CALUDE_ball_distribution_after_four_rounds_l1966_196621


namespace NUMINAMATH_CALUDE_k_greater_than_one_over_e_l1966_196681

/-- Given that k(e^(kx)+1)-(1+1/x)ln(x) > 0 for all x > 0, prove that k > 1/e -/
theorem k_greater_than_one_over_e (k : ℝ) 
  (h : ∀ x : ℝ, x > 0 → k * (Real.exp (k * x) + 1) - (1 + 1 / x) * Real.log x > 0) : 
  k > 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_k_greater_than_one_over_e_l1966_196681


namespace NUMINAMATH_CALUDE_tina_career_difference_l1966_196659

def boxing_career (initial_wins : ℕ) (second_wins : ℕ) (third_wins : ℕ) (fourth_wins : ℕ) : ℕ := 
  let wins1 := initial_wins + second_wins
  let wins2 := wins1 * 3
  let wins3 := wins2 + third_wins
  let wins4 := wins3 * 2
  let wins5 := wins4 + fourth_wins
  wins5 * wins5

theorem tina_career_difference : 
  boxing_career 10 5 7 11 - 4 = 13221 := by sorry

end NUMINAMATH_CALUDE_tina_career_difference_l1966_196659


namespace NUMINAMATH_CALUDE_unique_positive_integers_sum_l1966_196670

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 37) / 2 + 5 / 2)

theorem unique_positive_integers_sum (d e f : ℕ+) :
  y^50 = 2*y^48 + 6*y^46 + 5*y^44 - y^25 + (d:ℝ)*y^21 + (e:ℝ)*y^19 + (f:ℝ)*y^15 →
  d + e + f = 98 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_integers_sum_l1966_196670


namespace NUMINAMATH_CALUDE_playground_width_l1966_196606

theorem playground_width (area : ℝ) (length : ℝ) (h1 : area = 143.2) (h2 : length = 4) :
  area / length = 35.8 := by
  sorry

end NUMINAMATH_CALUDE_playground_width_l1966_196606


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l1966_196649

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) :
  (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0 → x + y + z = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l1966_196649


namespace NUMINAMATH_CALUDE_expected_socks_theorem_l1966_196663

/-- The expected number of socks taken until a pair is found -/
def expected_socks (n : ℕ) : ℝ := 2 * n

/-- Theorem: For n pairs of distinct socks arranged randomly, 
    the expected number of socks taken until a pair is found is 2n -/
theorem expected_socks_theorem (n : ℕ) : 
  expected_socks n = 2 * n := by sorry

end NUMINAMATH_CALUDE_expected_socks_theorem_l1966_196663


namespace NUMINAMATH_CALUDE_hall_mat_expenditure_l1966_196665

/-- Calculates the total expenditure for covering the interior of a rectangular hall with mat. -/
def total_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  let floor_area := length * width
  let wall_area := 2 * (length * height + width * height)
  let total_area := 2 * floor_area + wall_area
  total_area * cost_per_sqm

/-- Proves that the total expenditure for covering the interior of a specific hall with mat is Rs. 19,000. -/
theorem hall_mat_expenditure :
  total_expenditure 20 15 5 20 = 19000 := by
  sorry

end NUMINAMATH_CALUDE_hall_mat_expenditure_l1966_196665


namespace NUMINAMATH_CALUDE_grandma_salad_ratio_l1966_196655

/-- Proves that the ratio of cherry tomatoes to mushrooms is 2:1 given the conditions of Grandma's salad --/
theorem grandma_salad_ratio : ∀ (cherry_tomatoes pickles bacon_bits : ℕ),
  pickles = 4 * cherry_tomatoes →
  bacon_bits = 4 * pickles →
  bacon_bits / 3 = 32 →
  cherry_tomatoes / 3 = 2 :=
by
  sorry

#check grandma_salad_ratio

end NUMINAMATH_CALUDE_grandma_salad_ratio_l1966_196655


namespace NUMINAMATH_CALUDE_angle_bisector_sum_l1966_196652

/-- A triangle with vertices A(2,3), B(-4,1), and C(5,-6) -/
structure Triangle where
  A : ℝ × ℝ := (2, 3)
  B : ℝ × ℝ := (-4, 1)
  C : ℝ × ℝ := (5, -6)

/-- The equation of an angle bisector in the form 3x + by + c = 0 -/
structure AngleBisectorEquation where
  b : ℝ
  c : ℝ

/-- The angle bisector of ∠A in the given triangle -/
def angleBisectorA (t : Triangle) : AngleBisectorEquation :=
  sorry

theorem angle_bisector_sum (t : Triangle) :
  let bisector := angleBisectorA t
  bisector.b + bisector.c = -2 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_sum_l1966_196652


namespace NUMINAMATH_CALUDE_count_negative_rationals_l1966_196601

theorem count_negative_rationals : 
  let S : Finset ℚ := {-5, -(-3), 3.14, |-2/7|, -(2^3), 0}
  (S.filter (λ x => x < 0)).card = 2 := by sorry

end NUMINAMATH_CALUDE_count_negative_rationals_l1966_196601


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1966_196620

/-- Properties of a hyperbola M with equation x²/4 - y²/2 = 1 -/
theorem hyperbola_properties :
  let M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 - p.2^2 / 2 = 1}
  ∃ (a b c : ℝ) (e : ℝ),
    a = 2 ∧
    b = Real.sqrt 2 ∧
    c = Real.sqrt 6 ∧
    e = Real.sqrt 6 / 2 ∧
    (2 * a = 4) ∧  -- Length of real axis
    (2 * b = 2 * Real.sqrt 2) ∧  -- Length of imaginary axis
    (2 * c = 2 * Real.sqrt 6) ∧  -- Focal distance
    (e = Real.sqrt 6 / 2)  -- Eccentricity
  := by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1966_196620


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1966_196625

theorem diophantine_equation_solution (x y z : ℕ) (h : x^2 + 3*y^2 = 2^z) :
  ∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1966_196625


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l1966_196664

/-- The polynomial Q(x) -/
def Q (d : ℝ) (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + d * x^2 - 4 * x + 15

/-- Theorem stating that if x-2 is a factor of Q(x), then d = -15.75 -/
theorem factor_implies_d_value :
  ∀ d : ℝ, (∀ x : ℝ, Q d x = 0 → x = 2) → d = -15.75 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l1966_196664


namespace NUMINAMATH_CALUDE_principal_amount_calculation_l1966_196657

theorem principal_amount_calculation (rate : ℝ) (interest : ℝ) (time : ℝ) :
  rate = 0.08333333333333334 →
  interest = 400 →
  time = 4 →
  interest = (interest / (rate * time)) * rate * time :=
by
  sorry

end NUMINAMATH_CALUDE_principal_amount_calculation_l1966_196657


namespace NUMINAMATH_CALUDE_max_elevation_l1966_196637

/-- The elevation function of a particle thrown vertically upwards -/
def s (t : ℝ) : ℝ := 240 * t - 24 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ (t : ℝ), ∀ (t' : ℝ), s t ≥ s t' ∧ s t = 600 := by
  sorry

end NUMINAMATH_CALUDE_max_elevation_l1966_196637


namespace NUMINAMATH_CALUDE_machines_for_hundred_books_l1966_196603

/-- The number of printing machines required to print a given number of books in a given number of days. -/
def machines_required (initial_machines : ℕ) (initial_books : ℕ) (initial_days : ℕ) 
                      (target_books : ℕ) (target_days : ℕ) : ℕ :=
  (target_books * initial_machines * initial_days) / (initial_books * target_days)

/-- Theorem stating that 5 machines are required to print 100 books in 100 days,
    given that 5 machines can print 5 books in 5 days. -/
theorem machines_for_hundred_books : 
  machines_required 5 5 5 100 100 = 5 := by
  sorry

#eval machines_required 5 5 5 100 100

end NUMINAMATH_CALUDE_machines_for_hundred_books_l1966_196603


namespace NUMINAMATH_CALUDE_magazine_cost_l1966_196612

theorem magazine_cost (total_books : ℕ) (book_cost : ℕ) (total_magazines : ℕ) (total_spent : ℕ) :
  total_books = 10 →
  book_cost = 15 →
  total_magazines = 10 →
  total_spent = 170 →
  ∃ (magazine_cost : ℕ), magazine_cost = 2 ∧ total_spent = total_books * book_cost + total_magazines * magazine_cost :=
by
  sorry

end NUMINAMATH_CALUDE_magazine_cost_l1966_196612


namespace NUMINAMATH_CALUDE_gas_refill_proof_l1966_196691

def gas_problem (initial_gas tank_capacity gas_to_store gas_to_doctor : ℕ) : Prop :=
  let remaining_gas := initial_gas - gas_to_store - gas_to_doctor
  tank_capacity - remaining_gas = tank_capacity - (initial_gas - gas_to_store - gas_to_doctor)

theorem gas_refill_proof (initial_gas tank_capacity gas_to_store gas_to_doctor : ℕ) 
  (h1 : initial_gas ≥ gas_to_store + gas_to_doctor)
  (h2 : tank_capacity ≥ initial_gas) :
  gas_problem initial_gas tank_capacity gas_to_store gas_to_doctor :=
by
  sorry

#check gas_refill_proof 10 12 6 2

end NUMINAMATH_CALUDE_gas_refill_proof_l1966_196691


namespace NUMINAMATH_CALUDE_equation_equivalence_l1966_196632

theorem equation_equivalence (x : ℝ) (Q : ℝ) (h : 5 * (3 * x - 4 * Real.pi) = Q) :
  10 * (6 * x - 8 * Real.pi) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1966_196632


namespace NUMINAMATH_CALUDE_divisibility_999_from_50_l1966_196677

/-- A function that extracts 50 consecutive digits from a 999-digit number starting at a given index -/
def extract_50_digits (n : ℕ) (start_index : ℕ) : ℕ := sorry

/-- Predicate to check if a number is a valid 999-digit number -/
def is_999_digit_number (n : ℕ) : Prop := sorry

theorem divisibility_999_from_50 (n : ℕ) (h1 : is_999_digit_number n)
  (h2 : ∀ i, i ≤ 950 → extract_50_digits n i % 2^50 = 0) :
  n % 2^999 = 0 := by sorry

end NUMINAMATH_CALUDE_divisibility_999_from_50_l1966_196677


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l1966_196631

/-- Given two parallel lines y = (a - a^2)x - 2 and y = (3a + 1)x + 1, prove that a = -1 -/
theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, y = (a - a^2) * x - 2 ↔ y = (3*a + 1) * x + 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l1966_196631


namespace NUMINAMATH_CALUDE_quiche_cost_is_15_l1966_196669

/-- Represents the cost of a single quiche -/
def quiche_cost : ℝ := sorry

/-- Represents the number of quiches ordered -/
def num_quiches : ℕ := 2

/-- Represents the cost of a single croissant -/
def croissant_cost : ℝ := 3

/-- Represents the number of croissants ordered -/
def num_croissants : ℕ := 6

/-- Represents the cost of a single biscuit -/
def biscuit_cost : ℝ := 2

/-- Represents the number of biscuits ordered -/
def num_biscuits : ℕ := 6

/-- Represents the discount rate -/
def discount_rate : ℝ := 0.1

/-- Represents the discounted total cost -/
def discounted_total : ℝ := 54

/-- Theorem stating that the cost of each quiche is $15 -/
theorem quiche_cost_is_15 : quiche_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_quiche_cost_is_15_l1966_196669


namespace NUMINAMATH_CALUDE_coefficient_x3_equals_neg16_l1966_196650

/-- The coefficient of x^3 in the expansion of (1-ax)^2(1+x)^6 -/
def coefficient_x3 (a : ℝ) : ℝ :=
  20 - 30*a + 6*a^2

theorem coefficient_x3_equals_neg16 (a : ℝ) :
  coefficient_x3 a = -16 → a = 2 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3_equals_neg16_l1966_196650


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1966_196646

/-- The focal length of a hyperbola with equation x²/9 - y²/4 = 1 is 2√13 -/
theorem hyperbola_focal_length : 
  ∀ (x y : ℝ), x^2/9 - y^2/4 = 1 → 
  ∃ (f : ℝ), f = 2 * Real.sqrt 13 ∧ f = 2 * Real.sqrt ((9 : ℝ) + (4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1966_196646


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1966_196688

theorem rectangular_box_volume : ∃ (x : ℕ), 
  x > 0 ∧ 20 * x^3 = 160 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1966_196688


namespace NUMINAMATH_CALUDE_baseball_team_selection_l1966_196696

theorem baseball_team_selection (total_players : Nat) (selected_players : Nat) (twins : Nat) :
  total_players = 16 →
  selected_players = 9 →
  twins = 2 →
  Nat.choose (total_players - twins) (selected_players - twins) = 3432 := by
sorry

end NUMINAMATH_CALUDE_baseball_team_selection_l1966_196696


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1966_196617

/-- Given a rectangle divided into three congruent smaller rectangles,
    where each smaller rectangle is similar to the large rectangle,
    the ratio of the longer side to the shorter side is √3 : 1 for all rectangles. -/
theorem rectangle_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_similar : x / y = (3 * y) / x) : x / y = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1966_196617


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l1966_196690

theorem quadratic_root_sum (p q : ℝ) : 
  (∃ (x : ℂ), x^2 + p*x + q = 0 ∧ x = 1 + I) → p + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l1966_196690


namespace NUMINAMATH_CALUDE_smallest_coloring_number_l1966_196672

/-- A coloring of positive integers -/
def Coloring (k : ℕ) := ℕ+ → Fin k

/-- A function from positive integers to positive integers -/
def IntegerFunction := ℕ+ → ℕ+

/-- Condition 1: For all n, m of the same color, f(n+m) = f(n) + f(m) -/
def SameColorAdditive (c : Coloring k) (f : IntegerFunction) : Prop :=
  ∀ n m : ℕ+, c n = c m → f (n + m) = f n + f m

/-- Condition 2: There exist n, m such that f(n+m) ≠ f(n) + f(m) -/
def ExistsNonAdditive (f : IntegerFunction) : Prop :=
  ∃ n m : ℕ+, f (n + m) ≠ f n + f m

/-- The main theorem statement -/
theorem smallest_coloring_number :
  (∃ k : ℕ+, ∃ c : Coloring k, ∃ f : IntegerFunction,
    SameColorAdditive c f ∧ ExistsNonAdditive f) ∧
  (∀ k : ℕ+, k < 3 →
    ¬∃ c : Coloring k, ∃ f : IntegerFunction,
      SameColorAdditive c f ∧ ExistsNonAdditive f) :=
sorry

end NUMINAMATH_CALUDE_smallest_coloring_number_l1966_196672


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l1966_196658

def U : Set ℕ := {x | 0 < x ∧ x ≤ 6}
def M : Set ℕ := {1, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement_equals : M ∩ (U \ N) = {1, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l1966_196658


namespace NUMINAMATH_CALUDE_divisor_problem_l1966_196687

theorem divisor_problem : ∃ (d : ℕ), d > 0 ∧ (1019 + 6) % d = 0 ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1966_196687


namespace NUMINAMATH_CALUDE_present_age_of_b_l1966_196618

theorem present_age_of_b (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- In 10 years, A will be twice as old as B was 10 years ago
  (a = b + 8) →              -- A is currently 8 years older than B
  b = 38                     -- B's present age is 38
  := by sorry

end NUMINAMATH_CALUDE_present_age_of_b_l1966_196618


namespace NUMINAMATH_CALUDE_steves_return_speed_l1966_196648

def one_way_distance : ℝ := 40
def total_travel_time : ℝ := 6

theorem steves_return_speed (v : ℝ) (h1 : v > 0) :
  (one_way_distance / v + one_way_distance / (2 * v) = total_travel_time) →
  2 * v = 20 := by
  sorry

end NUMINAMATH_CALUDE_steves_return_speed_l1966_196648


namespace NUMINAMATH_CALUDE_variable_value_proof_l1966_196698

theorem variable_value_proof : ∃ x : ℝ, 3 * x + 36 = 48 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_variable_value_proof_l1966_196698


namespace NUMINAMATH_CALUDE_compound_ratio_example_l1966_196640

-- Define the compound ratio function
def compound_ratio (a b c d e f g h : ℚ) : ℚ := (a * c * e * g) / (b * d * f * h)

-- State the theorem
theorem compound_ratio_example : compound_ratio 2 3 6 7 1 3 3 8 = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_compound_ratio_example_l1966_196640


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l1966_196619

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l1966_196619


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l1966_196692

theorem two_digit_number_puzzle :
  ∃! n : ℕ,
    n ≥ 10 ∧ n < 100 ∧
    (n / 10 + n % 10 = 8) ∧
    (n - 36 = (n % 10) * 10 + (n / 10)) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l1966_196692


namespace NUMINAMATH_CALUDE_distance_to_origin_l1966_196683

theorem distance_to_origin : 
  let P : ℝ × ℝ := (2, -3)
  Real.sqrt ((P.1 - 0)^2 + (P.2 - 0)^2) = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1966_196683


namespace NUMINAMATH_CALUDE_women_in_third_group_l1966_196611

/-- Represents the work rate of a person -/
structure WorkRate where
  rate : ℝ

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  women : ℕ

/-- Calculates the total work done by a group -/
def totalWork (g : WorkGroup) (m : WorkRate) (w : WorkRate) : ℝ :=
  (g.men : ℝ) * m.rate + (g.women : ℝ) * w.rate

theorem women_in_third_group 
  (m : WorkRate) (w : WorkRate)
  (h1 : totalWork ⟨3, 8⟩ m w = totalWork ⟨6, 2⟩ m w)
  (h2 : ∃ x : ℕ, totalWork ⟨4, x⟩ m w = 0.9285714285714286 * totalWork ⟨3, 8⟩ m w) :
  ∃ x : ℕ, x = 5 ∧ totalWork ⟨4, x⟩ m w = 0.9285714285714286 * totalWork ⟨3, 8⟩ m w :=
sorry

end NUMINAMATH_CALUDE_women_in_third_group_l1966_196611


namespace NUMINAMATH_CALUDE_frequency_distribution_purpose_l1966_196673

/-- A frequency distribution table showing sample data sizes in groups -/
structure FrequencyDistributionTable where
  groups : Set (ℕ → ℕ)  -- Each function represents a group mapping sample size to frequency

/-- The proportion of data in each group -/
def proportion (t : FrequencyDistributionTable) : Set (ℕ → ℝ) :=
  sorry

/-- The overall corresponding situation being estimated -/
def overallSituation (t : FrequencyDistributionTable) : Type :=
  sorry

/-- Theorem stating the equivalence between the frequency distribution table
    and understanding proportions and estimating the overall situation -/
theorem frequency_distribution_purpose (t : FrequencyDistributionTable) :
  (∃ p : Set (ℕ → ℝ), p = proportion t) ∧
  (∃ s : Type, s = overallSituation t) :=
sorry

end NUMINAMATH_CALUDE_frequency_distribution_purpose_l1966_196673


namespace NUMINAMATH_CALUDE_prob_spade_then_king_value_l1966_196653

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of kings in a standard deck -/
def NumKings : ℕ := 4

/-- Probability of drawing a spade as the first card and a king as the second card -/
def prob_spade_then_king : ℚ :=
  (NumSpades / StandardDeck) * (NumKings / (StandardDeck - 1))

theorem prob_spade_then_king_value :
  prob_spade_then_king = 17 / 884 := by
  sorry

end NUMINAMATH_CALUDE_prob_spade_then_king_value_l1966_196653


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1966_196602

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (2, x)
  parallel a b → x = -6 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1966_196602


namespace NUMINAMATH_CALUDE_set_operations_l1966_196638

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 + 2*x - 3 > 0}

-- Define the theorem
theorem set_operations :
  (Set.compl (A ∪ B) = {x | -3 ≤ x ∧ x ≤ 0}) ∧
  ((Set.compl A) ∩ B = {x | x > 1 ∨ x < -3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1966_196638


namespace NUMINAMATH_CALUDE_parabola_coefficient_l1966_196666

/-- Given a parabola y = ax^2 + bx + c with vertex at (q, q) and y-intercept at (0, -2q),
    where q ≠ 0, the value of b is 6/q. -/
theorem parabola_coefficient (a b c q : ℝ) : q ≠ 0 →
  (∀ x y, y = a * x^2 + b * x + c ↔ 
    ((x - q)^2 = 0 → y = q) ∧ 
    (x = 0 → y = -2 * q)) →
  b = 6 / q := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l1966_196666


namespace NUMINAMATH_CALUDE_ones_digit_of_3_to_52_l1966_196634

theorem ones_digit_of_3_to_52 : (3^52 : ℕ) % 10 = 1 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_3_to_52_l1966_196634


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l1966_196643

/-- Given a curve C with polar equation ρ = 2cos(θ), 
    its Cartesian coordinate equation is x² + y² - 2x = 0 -/
theorem polar_to_cartesian_circle (x y : ℝ) :
  (∃ θ : ℝ, x = 2 * Real.cos θ * Real.cos θ ∧ y = 2 * Real.cos θ * Real.sin θ) ↔ 
  x^2 + y^2 - 2*x = 0 := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l1966_196643


namespace NUMINAMATH_CALUDE_lake_circumference_diameter_ratio_l1966_196608

/-- For a circular lake with given diameter and circumference, 
    prove that the ratio of circumference to diameter is 3.14 -/
theorem lake_circumference_diameter_ratio :
  ∀ (diameter circumference : ℝ),
    diameter = 100 →
    circumference = 314 →
    circumference / diameter = 3.14 := by
  sorry

end NUMINAMATH_CALUDE_lake_circumference_diameter_ratio_l1966_196608


namespace NUMINAMATH_CALUDE_all_yarns_are_xants_and_wooks_l1966_196671

-- Define the sets
variable (Zelm Xant Yarn Wook : Type)

-- Define the conditions
variable (zelm_xant : Zelm → Xant)
variable (yarn_zelm : Yarn → Zelm)
variable (xant_wook : Xant → Wook)

-- Theorem to prove
theorem all_yarns_are_xants_and_wooks :
  (∀ y : Yarn, ∃ x : Xant, zelm_xant (yarn_zelm y) = x) ∧
  (∀ y : Yarn, ∃ w : Wook, xant_wook (zelm_xant (yarn_zelm y)) = w) :=
sorry

end NUMINAMATH_CALUDE_all_yarns_are_xants_and_wooks_l1966_196671


namespace NUMINAMATH_CALUDE_palindrome_product_sum_l1966_196604

/-- A positive three-digit palindrome is a number between 100 and 999 (inclusive) that reads the same forwards and backwards. -/
def IsPositiveThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10) ∧ ((n / 10) % 10 = (n % 100) / 10)

/-- The theorem stating that if there exist two positive three-digit palindromes whose product is 445,545, then their sum is 1436. -/
theorem palindrome_product_sum : 
  ∃ (a b : ℕ), IsPositiveThreeDigitPalindrome a ∧ 
                IsPositiveThreeDigitPalindrome b ∧ 
                a * b = 445545 → 
                a + b = 1436 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_product_sum_l1966_196604


namespace NUMINAMATH_CALUDE_area_of_rectangle_in_18_gon_l1966_196641

/-- Given a regular 18-sided polygon with area 2016 square centimeters,
    the area of a rectangle formed by connecting the midpoints of four adjacent sides
    is 448 square centimeters. -/
theorem area_of_rectangle_in_18_gon (A : ℝ) (h : A = 2016) :
  let rectangle_area := A / 18 * 4
  rectangle_area = 448 :=
by sorry

end NUMINAMATH_CALUDE_area_of_rectangle_in_18_gon_l1966_196641


namespace NUMINAMATH_CALUDE_chongqing_population_scientific_notation_l1966_196630

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Conversion from a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The population of Chongqing at the end of 2022 -/
def chongqingPopulation : ℕ := 32000000

theorem chongqing_population_scientific_notation :
  toScientificNotation (chongqingPopulation : ℝ) =
    ScientificNotation.mk 3.2 7 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_chongqing_population_scientific_notation_l1966_196630


namespace NUMINAMATH_CALUDE_solve_equation_l1966_196635

theorem solve_equation (y : ℝ) : (7 - y = 4) → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1966_196635


namespace NUMINAMATH_CALUDE_cubic_equation_root_l1966_196656

theorem cubic_equation_root (a b : ℚ) : 
  (3 + Real.sqrt 5) ^ 3 + a * (3 + Real.sqrt 5) ^ 2 + b * (3 + Real.sqrt 5) + 20 = 0 → 
  b = -26 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l1966_196656


namespace NUMINAMATH_CALUDE_coin_array_problem_l1966_196639

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem coin_array_problem :
  ∃ (n : ℕ), triangular_sum n = 3003 ∧ n = 77 ∧ sum_of_digits n = 14 :=
sorry

end NUMINAMATH_CALUDE_coin_array_problem_l1966_196639


namespace NUMINAMATH_CALUDE_paint_tins_needed_half_tin_leftover_l1966_196613

-- Define the wall area range
def wall_area_min : ℝ := 1915
def wall_area_max : ℝ := 1925

-- Define the paint coverage range per tin
def coverage_min : ℝ := 17.5
def coverage_max : ℝ := 18.5

-- Define the minimum number of tins needed
def min_tins : ℕ := 111

-- Theorem statement
theorem paint_tins_needed :
  ∀ (wall_area paint_coverage : ℝ),
    wall_area_min ≤ wall_area ∧ wall_area < wall_area_max →
    coverage_min ≤ paint_coverage ∧ paint_coverage < coverage_max →
    (↑min_tins : ℝ) * coverage_min > wall_area_max ∧
    (↑(min_tins - 1) : ℝ) * coverage_min ≤ wall_area_max :=
by sorry

-- Additional theorem to ensure at least half a tin is left over
theorem half_tin_leftover :
  (↑min_tins : ℝ) * coverage_min - wall_area_max ≥ 0.5 * coverage_min :=
by sorry

end NUMINAMATH_CALUDE_paint_tins_needed_half_tin_leftover_l1966_196613


namespace NUMINAMATH_CALUDE_paper_cranes_problem_l1966_196636

/-- The number of paper cranes folded by student A -/
def cranes_A (x : ℤ) : ℤ := 3 * x - 100

/-- The number of paper cranes folded by student C -/
def cranes_C (x : ℤ) : ℤ := cranes_A x - 67

theorem paper_cranes_problem (x : ℤ) 
  (h1 : cranes_A x + x + cranes_C x = 1000) : 
  cranes_A x = 443 := by
  sorry

end NUMINAMATH_CALUDE_paper_cranes_problem_l1966_196636


namespace NUMINAMATH_CALUDE_system_solutions_l1966_196661

def system_of_equations (x y z : ℝ) : Prop :=
  3 * x * y - 5 * y * z - x * z = 3 * y ∧
  x * y + y * z = -y ∧
  -5 * x * y + 4 * y * z + x * z = -4 * y

theorem system_solutions :
  (∀ x : ℝ, system_of_equations x 0 0) ∧
  (∀ z : ℝ, system_of_equations 0 0 z) ∧
  system_of_equations 2 (-1/3) (-3) :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l1966_196661


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_condition_l1966_196627

/-- Theorem: For an infinite geometric sequence with first term a₁ and common ratio q,
    if the sum of the sequence is 1/2, then 2a₁ + q = 1. -/
theorem geometric_sequence_sum_condition (a₁ q : ℝ) (h : |q| < 1) :
  (∑' n, a₁ * q ^ (n - 1) = 1/2) → 2 * a₁ + q = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_condition_l1966_196627


namespace NUMINAMATH_CALUDE_integral_shift_reciprocal_l1966_196684

/-- For a continuous function f: ℝ → ℝ, if the integral of f over the real line exists,
    then the integral of f(x - 1/x) over the real line equals the integral of f. -/
theorem integral_shift_reciprocal (f : ℝ → ℝ) (hf : Continuous f) 
  (L : ℝ) (hL : ∫ (x : ℝ), f x = L) :
  ∫ (x : ℝ), f (x - 1/x) = L := by
  sorry

end NUMINAMATH_CALUDE_integral_shift_reciprocal_l1966_196684


namespace NUMINAMATH_CALUDE_assignment_ways_theorem_l1966_196609

/-- The number of ways to assign 7 friends to 7 rooms with at most 3 friends per room -/
def assignment_ways : ℕ := 17640

/-- The number of rooms in the inn -/
def num_rooms : ℕ := 7

/-- The number of friends arriving -/
def num_friends : ℕ := 7

/-- The maximum number of friends allowed per room -/
def max_per_room : ℕ := 3

/-- Theorem stating that the number of ways to assign 7 friends to 7 rooms,
    with at most 3 friends per room, is equal to 17640 -/
theorem assignment_ways_theorem :
  ∃ (ways : ℕ → ℕ → ℕ → ℕ),
    ways num_rooms num_friends max_per_room = assignment_ways :=
by sorry

end NUMINAMATH_CALUDE_assignment_ways_theorem_l1966_196609


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1966_196678

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x + 3 = 7) : 
  3*x^2 + 3*x + 7 = 19 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1966_196678


namespace NUMINAMATH_CALUDE_pelican_fish_count_l1966_196654

theorem pelican_fish_count (P : ℕ) : 
  (P + 7 = P + 7) →  -- Kingfisher caught 7 more fish than the pelican
  (3 * (P + (P + 7)) = P + 86) →  -- Fisherman caught 3 times the total and 86 more than the pelican
  P = 13 := by
sorry

end NUMINAMATH_CALUDE_pelican_fish_count_l1966_196654


namespace NUMINAMATH_CALUDE_double_infinite_sum_equals_two_l1966_196614

theorem double_infinite_sum_equals_two :
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / (m * n * (m + n + 3))) = 2 := by sorry

end NUMINAMATH_CALUDE_double_infinite_sum_equals_two_l1966_196614


namespace NUMINAMATH_CALUDE_room_length_proof_l1966_196699

/-- Proves that a room with given width, paving cost per area, and total paving cost has a specific length -/
theorem room_length_proof (width : ℝ) (cost_per_area : ℝ) (total_cost : ℝ) :
  width = 3.75 →
  cost_per_area = 800 →
  total_cost = 16500 →
  (total_cost / cost_per_area) / width = 5.5 := by
  sorry

#check room_length_proof

end NUMINAMATH_CALUDE_room_length_proof_l1966_196699

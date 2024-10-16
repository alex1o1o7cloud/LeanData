import Mathlib

namespace NUMINAMATH_CALUDE_bean_feast_spending_l2832_283224

/-- The bean-feast spending problem -/
theorem bean_feast_spending
  (cobblers tailors hatters glovers : ℕ)
  (total_spent : ℕ)
  (h_cobblers : cobblers = 25)
  (h_tailors : tailors = 20)
  (h_hatters : hatters = 18)
  (h_glovers : glovers = 12)
  (h_total : total_spent = 133)  -- 133 shillings = £6 13s
  (h_cobbler_tailor : 5 * (cobblers : ℚ) = 4 * (tailors : ℚ))
  (h_tailor_hatter : 12 * (tailors : ℚ) = 9 * (hatters : ℚ))
  (h_hatter_glover : 6 * (hatters : ℚ) = 8 * (glovers : ℚ)) :
  ∃ (g h t c : ℚ),
    g = 21 ∧ h = 42 ∧ t = 35 ∧ c = 35 ∧
    g * glovers + h * hatters + t * tailors + c * cobblers = total_spent :=
by sorry


end NUMINAMATH_CALUDE_bean_feast_spending_l2832_283224


namespace NUMINAMATH_CALUDE_dhoni_rent_percentage_l2832_283298

theorem dhoni_rent_percentage (rent_percentage : ℝ) 
  (h1 : rent_percentage > 0)
  (h2 : rent_percentage < 100)
  (h3 : rent_percentage + (rent_percentage - 10) + 52.5 = 100) :
  rent_percentage = 28.75 := by
sorry

end NUMINAMATH_CALUDE_dhoni_rent_percentage_l2832_283298


namespace NUMINAMATH_CALUDE_abs_4y_minus_6_not_positive_l2832_283244

theorem abs_4y_minus_6_not_positive (y : ℚ) : ¬(|4*y - 6| > 0) ↔ y = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_4y_minus_6_not_positive_l2832_283244


namespace NUMINAMATH_CALUDE_mean_equals_n_l2832_283255

theorem mean_equals_n (n : ℝ) : 
  (17 + 98 + 39 + 54 + n) / 5 = n → n = 52 := by
  sorry

end NUMINAMATH_CALUDE_mean_equals_n_l2832_283255


namespace NUMINAMATH_CALUDE_dissimilar_terms_eq_choose_l2832_283294

/-- The number of dissimilar terms in the expansion of (x + y + z)^8 -/
def dissimilar_terms : ℕ :=
  Nat.choose 10 2

/-- Theorem stating that the number of dissimilar terms in (x + y + z)^8 is equal to (10 choose 2) -/
theorem dissimilar_terms_eq_choose : dissimilar_terms = 45 := by
  sorry

end NUMINAMATH_CALUDE_dissimilar_terms_eq_choose_l2832_283294


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2832_283233

/-- Given a > 0, prove that the solution set of |((2x - 3 - 2a) / (x - a))| ≤ 1 is {x | a + 1 ≤ x ≤ a + 3} -/
theorem solution_set_inequality (a : ℝ) (ha : a > 0) :
  {x : ℝ | |((2 * x - 3 - 2 * a) / (x - a))| ≤ 1} = {x : ℝ | a + 1 ≤ x ∧ x ≤ a + 3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2832_283233


namespace NUMINAMATH_CALUDE_quadratic_roots_real_distinct_l2832_283222

theorem quadratic_roots_real_distinct :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + 6*x₁ + 8 = 0) ∧ (x₂^2 + 6*x₂ + 8 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_distinct_l2832_283222


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2832_283271

/-- The range of k for an ellipse with equation x^2/4 + y^2/k = 1 and eccentricity e ∈ (1/2, 1) -/
theorem ellipse_k_range (e : ℝ) (h1 : 1/2 < e) (h2 : e < 1) :
  ∀ k : ℝ, (∃ x y : ℝ, x^2/4 + y^2/k = 1 ∧ e^2 = 1 - (min 4 k)/(max 4 k)) ↔
  (k ∈ Set.Ioo 0 3 ∪ Set.Ioi (16/3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2832_283271


namespace NUMINAMATH_CALUDE_b_n_equals_c_1_l2832_283220

theorem b_n_equals_c_1 (n : ℕ) (a : ℕ → ℝ) (b c : ℕ → ℝ)
  (h_positive : ∀ i, 1 ≤ i → i ≤ n → 0 < a i)
  (h_b_1 : b 1 = a 1)
  (h_b_2 : b 2 = max (a 1) (a 2))
  (h_b_i : ∀ i, 3 ≤ i → i ≤ n → b i = max (b (i - 1)) (b (i - 2) + a i))
  (h_c_n : c n = a n)
  (h_c_n_1 : c (n - 1) = max (a n) (a (n - 1)))
  (h_c_i : ∀ i, 1 ≤ i → i ≤ n - 2 → c i = max (c (i + 1)) (c (i + 2) + a i)) :
  b n = c 1 := by
  sorry


end NUMINAMATH_CALUDE_b_n_equals_c_1_l2832_283220


namespace NUMINAMATH_CALUDE_sin_plus_cos_value_l2832_283299

theorem sin_plus_cos_value (x : Real) 
  (h1 : 0 < x ∧ x < Real.pi / 2) 
  (h2 : Real.tan (x - Real.pi / 4) = -1 / 7) : 
  Real.sin x + Real.cos x = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_value_l2832_283299


namespace NUMINAMATH_CALUDE_finite_perfect_squares_l2832_283200

theorem finite_perfect_squares (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (S : Finset ℤ), ∀ (n : ℤ),
    (∃ (x : ℤ), a * n^2 + b = x^2) ∧ (∃ (y : ℤ), a * (n + 1)^2 + b = y^2) →
    n ∈ S :=
sorry

end NUMINAMATH_CALUDE_finite_perfect_squares_l2832_283200


namespace NUMINAMATH_CALUDE_rectangle_tiling_l2832_283263

theorem rectangle_tiling (m n a b : ℕ) (hm : m > 0) (hn : n > 0) 
  (ha : a > 0) (hb : b > 0) 
  (h_tiling : ∃ (h v : ℕ), a * b = h * m + v * n) :
  a % m = 0 ∨ b % n = 0 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_tiling_l2832_283263


namespace NUMINAMATH_CALUDE_female_officers_on_duty_percentage_l2832_283258

def total_on_duty : ℕ := 240
def female_ratio_on_duty : ℚ := 1/2
def total_female_officers : ℕ := 300

theorem female_officers_on_duty_percentage :
  (female_ratio_on_duty * total_on_duty) / total_female_officers * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_on_duty_percentage_l2832_283258


namespace NUMINAMATH_CALUDE_circle_configuration_l2832_283241

-- Define the types of people
inductive PersonType
| Knight
| Liar
| Visitor

-- Define a person
structure Person where
  id : Fin 7
  type : PersonType

-- Define the circle of people
def Circle := Fin 7 → Person

-- Define a statement made by a pair of people
structure Statement where
  speaker1 : Fin 7
  speaker2 : Fin 7
  content : Nat
  category : PersonType

-- Define the function to check if a statement is true
def isStatementTrue (c : Circle) (s : Statement) : Prop :=
  (c s.speaker1).type = PersonType.Knight ∨
  (c s.speaker2).type = PersonType.Knight ∨
  ((c s.speaker1).type = PersonType.Visitor ∧ (c s.speaker2).type = PersonType.Visitor)

-- Define the list of statements
def statements : List Statement := [
  ⟨0, 1, 1, PersonType.Liar⟩,
  ⟨1, 2, 2, PersonType.Knight⟩,
  ⟨2, 3, 3, PersonType.Liar⟩,
  ⟨3, 4, 4, PersonType.Knight⟩,
  ⟨4, 5, 5, PersonType.Liar⟩,
  ⟨5, 6, 6, PersonType.Knight⟩,
  ⟨6, 0, 7, PersonType.Liar⟩
]

-- Define the theorem
theorem circle_configuration (c : Circle) :
  (∀ s ∈ statements, isStatementTrue c s ∨ ¬isStatementTrue c s) →
  (∃! (i j : Fin 7), i ≠ j ∧ 
    (c i).type = PersonType.Visitor ∧ 
    (c j).type = PersonType.Visitor ∧
    (∀ k : Fin 7, k ≠ i ∧ k ≠ j → (c k).type = PersonType.Liar)) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_configuration_l2832_283241


namespace NUMINAMATH_CALUDE_omm_moo_not_synonyms_l2832_283259

/-- Represents a word in the Ancient Tribe language --/
inductive Word
| empty : Word
| cons : Char → Word → Word

/-- Counts the number of occurrences of a given character in a word --/
def count_char (c : Char) : Word → Nat
| Word.empty => 0
| Word.cons x rest => (if x = c then 1 else 0) + count_char c rest

/-- Calculates the difference between the count of 'M's and 'O's in a word --/
def m_o_difference (w : Word) : Int :=
  (count_char 'M' w : Int) - (count_char 'O' w : Int)

/-- Defines when two words are synonyms --/
def are_synonyms (w1 w2 : Word) : Prop :=
  m_o_difference w1 = m_o_difference w2

/-- Represents the word OMM --/
def omm : Word := Word.cons 'O' (Word.cons 'M' (Word.cons 'M' Word.empty))

/-- Represents the word MOO --/
def moo : Word := Word.cons 'M' (Word.cons 'O' (Word.cons 'O' Word.empty))

/-- Theorem stating that OMM and MOO are not synonyms --/
theorem omm_moo_not_synonyms : ¬(are_synonyms omm moo) := by
  sorry

end NUMINAMATH_CALUDE_omm_moo_not_synonyms_l2832_283259


namespace NUMINAMATH_CALUDE_exists_divisor_in_range_l2832_283204

theorem exists_divisor_in_range : ∃ n : ℕ, 
  100 ≤ n ∧ n ≤ 1997 ∧ (n ∣ 2 * n + 2) ∧ n = 946 := by
  sorry

end NUMINAMATH_CALUDE_exists_divisor_in_range_l2832_283204


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2832_283273

theorem absolute_value_inequality_solution_set (x : ℝ) :
  (1 < |2*x - 1| ∧ |2*x - 1| < 3) ↔ ((-1 < x ∧ x < 0) ∨ (1 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2832_283273


namespace NUMINAMATH_CALUDE_min_perimeter_is_16_l2832_283281

/-- A regular polygon -/
structure RegularPolygon :=
  (sides : ℕ)
  (sideLength : ℝ)

/-- The perimeter of a regular polygon -/
def perimeter (p : RegularPolygon) : ℝ :=
  p.sides * p.sideLength

/-- The configuration of polygons surrounding the triangle -/
structure PolygonConfiguration :=
  (p1 : RegularPolygon)
  (p2 : RegularPolygon)
  (p3 : RegularPolygon)

/-- The total perimeter of the configuration, excluding shared edges -/
def totalPerimeter (c : PolygonConfiguration) : ℝ :=
  perimeter c.p1 + perimeter c.p2 + perimeter c.p3 - 3 * 2

/-- The theorem stating the minimum perimeter -/
theorem min_perimeter_is_16 :
  ∃ (c : PolygonConfiguration),
    (c.p1.sideLength = 2 ∧ c.p2.sideLength = 2 ∧ c.p3.sideLength = 2) ∧
    (c.p1 = c.p2 ∨ c.p1 = c.p3 ∨ c.p2 = c.p3) ∧
    (∀ (d : PolygonConfiguration),
      (d.p1.sideLength = 2 ∧ d.p2.sideLength = 2 ∧ d.p3.sideLength = 2) →
      (d.p1 = d.p2 ∨ d.p1 = d.p3 ∨ d.p2 = d.p3) →
      totalPerimeter c ≤ totalPerimeter d) ∧
    totalPerimeter c = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_is_16_l2832_283281


namespace NUMINAMATH_CALUDE_circular_arrangement_students_l2832_283226

/-- 
Given a circular arrangement of students, if the 10th and 40th positions 
are opposite each other, then the total number of students is 62.
-/
theorem circular_arrangement_students (n : ℕ) : 
  (∃ (a b : ℕ), a = 10 ∧ b = 40 ∧ a < b ∧ b - a = n - (b - a)) → n = 62 :=
by sorry

end NUMINAMATH_CALUDE_circular_arrangement_students_l2832_283226


namespace NUMINAMATH_CALUDE_equation_transformation_l2832_283269

theorem equation_transformation (x : ℝ) (h : x ≠ 1) :
  1 / (x - 1) + 3 = 3 * x / (1 - x) → 1 + 3 * (x - 1) = -3 * x := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l2832_283269


namespace NUMINAMATH_CALUDE_river_depth_l2832_283272

theorem river_depth (depth_may : ℝ) (depth_june : ℝ) (depth_july : ℝ) 
  (h1 : depth_june = depth_may + 10)
  (h2 : depth_july = 3 * depth_june)
  (h3 : depth_july = 45) : 
  depth_may = 5 := by
sorry

end NUMINAMATH_CALUDE_river_depth_l2832_283272


namespace NUMINAMATH_CALUDE_friendly_function_properties_l2832_283227

/-- Definition of a Friendly Function on [0, 1] -/
def FriendlyFunction (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧
  (f 1 = 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

theorem friendly_function_properties
  (f : ℝ → ℝ) (hf : FriendlyFunction f) :
  (f 0 = 0) ∧
  (∀ x₀ ∈ Set.Icc 0 1, f x₀ ∈ Set.Icc 0 1 → f (f x₀) = x₀ → f x₀ = x₀) :=
by sorry

end NUMINAMATH_CALUDE_friendly_function_properties_l2832_283227


namespace NUMINAMATH_CALUDE_raft_sticks_total_l2832_283276

/-- The number of sticks needed for Simon's raft -/
def simon_sticks : ℕ := 36

/-- The number of sticks needed for Gerry's raft -/
def gerry_sticks : ℕ := (2 * simon_sticks) / 3

/-- The number of sticks needed for Micky's raft -/
def micky_sticks : ℕ := simon_sticks + gerry_sticks + 9

/-- The total number of sticks needed for all three rafts -/
def total_sticks : ℕ := simon_sticks + gerry_sticks + micky_sticks

theorem raft_sticks_total : total_sticks = 129 := by
  sorry

end NUMINAMATH_CALUDE_raft_sticks_total_l2832_283276


namespace NUMINAMATH_CALUDE_door_height_proof_l2832_283205

theorem door_height_proof (pole_length width height diagonal : ℝ) :
  width = pole_length - 4 →
  height = pole_length - 2 →
  diagonal = pole_length →
  diagonal^2 = width^2 + height^2 →
  height = 8 :=
by sorry

end NUMINAMATH_CALUDE_door_height_proof_l2832_283205


namespace NUMINAMATH_CALUDE_prob_select_all_leaders_in_district_l2832_283242

/-- Represents a math club with a given number of students and leaders -/
structure MathClub where
  students : Nat
  leaders : Nat

/-- Calculates the probability of selecting all leaders in a given club -/
def prob_select_all_leaders (club : MathClub) : Rat :=
  (club.students - club.leaders).choose 1 / club.students.choose 4

/-- The list of math clubs in the school district -/
def math_clubs : List MathClub := [
  ⟨6, 3⟩,
  ⟨8, 3⟩,
  ⟨9, 3⟩,
  ⟨10, 3⟩
]

/-- The main theorem stating the probability of selecting all leaders -/
theorem prob_select_all_leaders_in_district : 
  (1 / 4 : Rat) * (math_clubs.map prob_select_all_leaders).sum = 37 / 420 := by
  sorry

end NUMINAMATH_CALUDE_prob_select_all_leaders_in_district_l2832_283242


namespace NUMINAMATH_CALUDE_chess_competition_games_l2832_283254

theorem chess_competition_games (W M : ℕ) (h1 : W = 12) (h2 : M = 24) : W * M = 288 := by
  sorry

end NUMINAMATH_CALUDE_chess_competition_games_l2832_283254


namespace NUMINAMATH_CALUDE_cubic_factorization_l2832_283291

theorem cubic_factorization (x : ℝ) : x^3 - 5*x^2 + 4*x = x*(x-1)*(x-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2832_283291


namespace NUMINAMATH_CALUDE_rectangle_length_l2832_283229

/-- Proves that the length of a rectangle is 16 centimeters, given specific conditions. -/
theorem rectangle_length (square_side : ℝ) (rect_width : ℝ) : 
  square_side = 8 →
  rect_width = 4 →
  square_side * square_side = rect_width * (16 : ℝ) :=
by
  sorry

#check rectangle_length

end NUMINAMATH_CALUDE_rectangle_length_l2832_283229


namespace NUMINAMATH_CALUDE_bunny_birth_rate_l2832_283234

theorem bunny_birth_rate (initial_bunnies : ℕ) (fraction_given : ℚ) (total_after_birth : ℕ) : 
  initial_bunnies = 30 →
  fraction_given = 2 / 5 →
  total_after_birth = 54 →
  (initial_bunnies - (fraction_given * initial_bunnies).num) * 2 = total_after_birth - (initial_bunnies - (fraction_given * initial_bunnies).num) :=
by
  sorry

end NUMINAMATH_CALUDE_bunny_birth_rate_l2832_283234


namespace NUMINAMATH_CALUDE_mat_cost_per_square_meter_l2832_283228

/-- Given a rectangular hall with specified dimensions and total expenditure for floor covering,
    calculate the cost per square meter of the mat. -/
theorem mat_cost_per_square_meter
  (length width height : ℝ)
  (total_expenditure : ℝ)
  (h_length : length = 20)
  (h_width : width = 15)
  (h_height : height = 5)
  (h_expenditure : total_expenditure = 57000) :
  total_expenditure / (length * width) = 190 := by
  sorry

end NUMINAMATH_CALUDE_mat_cost_per_square_meter_l2832_283228


namespace NUMINAMATH_CALUDE_statement_to_equation_l2832_283284

theorem statement_to_equation (a : ℝ) : 
  (3 * a + 5 = 4 * a) ↔ 
  (∃ x : ℝ, x = 3 * a + 5 ∧ x = 4 * a) :=
by sorry

end NUMINAMATH_CALUDE_statement_to_equation_l2832_283284


namespace NUMINAMATH_CALUDE_max_saturdays_is_five_l2832_283213

/-- Represents the possible number of days in a month -/
inductive MonthLength
  | Days28
  | Days29
  | Days30
  | Days31

/-- Represents the day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the number of Saturdays in a month -/
def saturdays_in_month (length : MonthLength) (start : DayOfWeek) : Nat :=
  sorry

/-- The maximum number of Saturdays in any month -/
def max_saturdays : Nat := 5

/-- Theorem: The maximum number of Saturdays in any month is 5 -/
theorem max_saturdays_is_five :
  ∀ (length : MonthLength) (start : DayOfWeek),
    saturdays_in_month length start ≤ max_saturdays :=
  sorry

end NUMINAMATH_CALUDE_max_saturdays_is_five_l2832_283213


namespace NUMINAMATH_CALUDE_arithmetic_sequence_average_l2832_283288

/-- 
Given an arithmetic sequence with:
- First term a₁ = 10
- Last term aₙ = 160
- Common difference d = 10

Prove that the average (arithmetic mean) of this sequence is 85.
-/
theorem arithmetic_sequence_average : 
  let a₁ : ℕ := 10
  let aₙ : ℕ := 160
  let d : ℕ := 10
  let n : ℕ := (aₙ - a₁) / d + 1
  (a₁ + aₙ) / 2 = 85 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_average_l2832_283288


namespace NUMINAMATH_CALUDE_line_through_points_slope_one_l2832_283251

/-- Given a line passing through points M(-2, m) and N(m, 4) with a slope of 1, prove that m = 1 -/
theorem line_through_points_slope_one (m : ℝ) : 
  (4 - m) / (m - (-2)) = 1 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_slope_one_l2832_283251


namespace NUMINAMATH_CALUDE_real_part_of_z_l2832_283231

theorem real_part_of_z (z : ℂ) (h : Complex.I * z = 1 + 2 * Complex.I) : 
  (z.re : ℝ) = 2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2832_283231


namespace NUMINAMATH_CALUDE_relative_speed_object_image_l2832_283289

/-- Relative speed between an object and its image for a converging lens -/
theorem relative_speed_object_image 
  (f : ℝ) (t : ℝ) (v_object : ℝ) :
  f = 10 →
  t = 30 →
  v_object = 200 →
  let k := f * t / (t - f)
  let v_image := f^2 / (t - f)^2 * v_object
  |v_object + v_image| = 150 := by
  sorry

end NUMINAMATH_CALUDE_relative_speed_object_image_l2832_283289


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_75_7350_l2832_283257

theorem gcd_lcm_sum_75_7350 : Nat.gcd 75 7350 + Nat.lcm 75 7350 = 3225 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_75_7350_l2832_283257


namespace NUMINAMATH_CALUDE_division_negative_ten_by_five_l2832_283261

theorem division_negative_ten_by_five : -10 / 5 = -2 := by sorry

end NUMINAMATH_CALUDE_division_negative_ten_by_five_l2832_283261


namespace NUMINAMATH_CALUDE_age_ratio_six_years_ago_l2832_283248

/-- Given Henry and Jill's ages, prove their age ratio 6 years ago -/
theorem age_ratio_six_years_ago 
  (henry_age : ℕ) 
  (jill_age : ℕ) 
  (henry_age_eq : henry_age = 20)
  (jill_age_eq : jill_age = 13)
  (sum_ages : henry_age + jill_age = 33)
  (past_multiple : ∃ k : ℕ, henry_age - 6 = k * (jill_age - 6)) :
  (henry_age - 6) / (jill_age - 6) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_six_years_ago_l2832_283248


namespace NUMINAMATH_CALUDE_computer_table_markup_l2832_283219

/-- The percentage markup on a product's cost price, given the selling price and cost price. -/
def percentageMarkup (sellingPrice costPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

/-- Proof that the percentage markup on a computer table is 30% -/
theorem computer_table_markup :
  percentageMarkup 8450 6500 = 30 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_markup_l2832_283219


namespace NUMINAMATH_CALUDE_wall_construction_boys_l2832_283211

/-- The number of boys who can construct the wall in 6 days -/
def num_boys : ℕ := 24

/-- The number of days it takes B boys or 24 girls to construct the wall -/
def days_boys_or_girls : ℕ := 6

/-- The number of days it takes B boys and 12 girls to construct the wall -/
def days_boys_and_girls : ℕ := 4

/-- The number of girls that can construct the wall in the same time as B boys -/
def equivalent_girls : ℕ := 24

theorem wall_construction_boys (B : ℕ) :
  (B * days_boys_or_girls = equivalent_girls * days_boys_or_girls) →
  ((B + 12 * equivalent_girls) * days_boys_and_girls = equivalent_girls * days_boys_or_girls) →
  B = num_boys :=
by sorry

end NUMINAMATH_CALUDE_wall_construction_boys_l2832_283211


namespace NUMINAMATH_CALUDE_quadratic_roots_characterization_l2832_283243

/-- The quadratic equation a² - 18a + 72 = 0 has solutions a = 6 and a = 12 -/
def quad_eq (a : ℝ) : Prop := a^2 - 18*a + 72 = 0

/-- The general form of the roots -/
def root_form (a x : ℝ) : Prop := x = a + Real.sqrt (18*(a-4)) ∨ x = a - Real.sqrt (18*(a-4))

/-- Condition for distinct positive roots -/
def distinct_positive_roots (a : ℝ) : Prop :=
  (4 < a ∧ a < 6) ∨ a > 12

/-- Condition for equal roots -/
def equal_roots (a : ℝ) : Prop :=
  (6 ≤ a ∧ a ≤ 12) ∨ a = 22

theorem quadratic_roots_characterization :
  ∀ a : ℝ, quad_eq a →
    (∃ x y : ℝ, x ≠ y ∧ root_form a x ∧ root_form a y ∧ x > 0 ∧ y > 0 ↔ distinct_positive_roots a) ∧
    (∃ x : ℝ, root_form a x ∧ x > 0 ↔ equal_roots a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_characterization_l2832_283243


namespace NUMINAMATH_CALUDE_combined_eel_length_l2832_283275

/-- The combined length of Jenna's and Bill's eels given their relative lengths -/
theorem combined_eel_length (jenna_length : ℝ) (h1 : jenna_length = 16) 
  (h2 : jenna_length = (1/3) * bill_length) : jenna_length + bill_length = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_eel_length_l2832_283275


namespace NUMINAMATH_CALUDE_marks_animals_legs_l2832_283285

def total_legs (num_kangaroos : ℕ) (num_goats : ℕ) : ℕ :=
  2 * num_kangaroos + 4 * num_goats

theorem marks_animals_legs : 
  let num_kangaroos : ℕ := 23
  let num_goats : ℕ := 3 * num_kangaroos
  total_legs num_kangaroos num_goats = 322 := by
sorry

end NUMINAMATH_CALUDE_marks_animals_legs_l2832_283285


namespace NUMINAMATH_CALUDE_equivalent_discount_l2832_283214

theorem equivalent_discount (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price > 0 →
  0 ≤ discount1 → discount1 < 1 →
  0 ≤ discount2 → discount2 < 1 →
  original_price * (1 - discount1) * (1 - discount2) = original_price * (1 - 0.4) :=
by
  sorry

#check equivalent_discount 50 0.25 0.2

end NUMINAMATH_CALUDE_equivalent_discount_l2832_283214


namespace NUMINAMATH_CALUDE_equilateral_triangle_division_exists_l2832_283230

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents a division of an equilateral triangle into smaller equilateral triangles -/
structure TriangleDivision where
  original : EquilateralTriangle
  num_divisions : ℕ
  side_lengths : Finset ℝ
  all_positive : ∀ l ∈ side_lengths, l > 0

/-- Theorem stating that there exists a division of an equilateral triangle into 2011 smaller equilateral triangles with only two different side lengths -/
theorem equilateral_triangle_division_exists : 
  ∃ (div : TriangleDivision), div.num_divisions = 2011 ∧ div.side_lengths.card = 2 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_division_exists_l2832_283230


namespace NUMINAMATH_CALUDE_lemonade_revenue_is_110_l2832_283279

/-- Calculates the total revenue from selling lemonade over three weeks -/
def lemonade_revenue : ℝ :=
  let first_week_cups : ℕ := 20
  let first_week_price : ℝ := 1
  let second_week_increase : ℝ := 0.5
  let second_week_price : ℝ := 1.25
  let third_week_increase : ℝ := 0.75
  let third_week_price : ℝ := 1.5

  let first_week_revenue := first_week_cups * first_week_price
  let second_week_revenue := (first_week_cups * (1 + second_week_increase)) * second_week_price
  let third_week_revenue := (first_week_cups * (1 + third_week_increase)) * third_week_price

  first_week_revenue + second_week_revenue + third_week_revenue

theorem lemonade_revenue_is_110 : lemonade_revenue = 110 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_revenue_is_110_l2832_283279


namespace NUMINAMATH_CALUDE_standard_spherical_coords_example_l2832_283274

/-- Given a point in spherical coordinates (ρ, θ, φ), this function returns the equivalent
    standard spherical coordinate representation that satisfies the conditions
    ρ > 0, 0 ≤ θ < 2π, and 0 ≤ φ ≤ π. -/
def standardSphericalCoords (ρ θ φ : Real) : Real × Real × Real :=
  sorry

/-- Theorem stating that the standard spherical coordinate representation
    of (4, 3π/4, 9π/4) is (4, 3π/4, π/4). -/
theorem standard_spherical_coords_example :
  standardSphericalCoords 4 (3 * Real.pi / 4) (9 * Real.pi / 4) = (4, 3 * Real.pi / 4, Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_standard_spherical_coords_example_l2832_283274


namespace NUMINAMATH_CALUDE_jellybean_probability_l2832_283262

theorem jellybean_probability (p_red p_orange p_yellow p_green : ℝ) :
  p_red = 0.25 →
  p_orange = 0.35 →
  p_yellow = 0.1 →
  p_red + p_orange + p_yellow + p_green = 1 →
  p_green = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l2832_283262


namespace NUMINAMATH_CALUDE_natural_number_representation_l2832_283202

theorem natural_number_representation (A : ℕ) :
  ∃ n : ℕ, A = 3 * n ∨ A = 3 * n + 1 ∨ A = 3 * n + 2 := by
  sorry

end NUMINAMATH_CALUDE_natural_number_representation_l2832_283202


namespace NUMINAMATH_CALUDE_intersection_solutions_l2832_283217

theorem intersection_solutions (α β : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ 2*x + 2*y - 1 - Real.sqrt 3 = 0 ∧
   (x = Real.sin α ∧ y = Real.sin (2*β) ∨ x = Real.sin β ∧ y = Real.cos (2*α))) →
  (∃ (n k : ℤ), α = (-1)^n * π/6 + π * (n : ℝ) ∧ β = π/3 + 2*π * (k : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_solutions_l2832_283217


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l2832_283221

/-- Given vectors a and b in R², if (a - b) is perpendicular to b, then the x-coordinate of b is either -1 or 3. -/
theorem vector_perpendicular_condition (x : ℝ) :
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, 3)
  (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 = 0 → x = -1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l2832_283221


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficients_l2832_283238

theorem binomial_expansion_coefficients :
  ∃ (b₁ b₂ b₃ b₄ : ℝ),
    (∀ x : ℝ, x^4 = (x+1)^4 + b₁*(x+1)^3 + b₂*(x+1)^2 + b₃*(x+1) + b₄) ∧
    b₁ = -4 ∧ b₂ = 6 ∧ b₃ = -4 ∧ b₄ = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficients_l2832_283238


namespace NUMINAMATH_CALUDE_fishing_contest_result_l2832_283283

/-- The number of salmons Hazel caught -/
def hazel_catch : ℕ := 24

/-- The number of salmons Hazel's father caught -/
def father_catch : ℕ := 27

/-- The total number of salmons caught by Hazel and her father -/
def total_catch : ℕ := hazel_catch + father_catch

theorem fishing_contest_result : total_catch = 51 := by
  sorry

end NUMINAMATH_CALUDE_fishing_contest_result_l2832_283283


namespace NUMINAMATH_CALUDE_perimeter_is_24_l2832_283210

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/36 + y^2/25 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- State that A and B are on the ellipse
axiom A_on_ellipse : ellipse A.1 A.2
axiom B_on_ellipse : ellipse B.1 B.2

-- State that A, B, and F₁ are collinear
axiom A_B_F₁_collinear : ∃ (t : ℝ), A = F₁ + t • (B - F₁) ∨ B = F₁ + t • (A - F₁)

-- Define the perimeter of triangle ABF₂
def perimeter_ABF₂ : ℝ := sorry

-- Theorem to prove
theorem perimeter_is_24 : perimeter_ABF₂ = 24 := by sorry

end NUMINAMATH_CALUDE_perimeter_is_24_l2832_283210


namespace NUMINAMATH_CALUDE_car_wheels_count_l2832_283225

theorem car_wheels_count (num_cars : ℕ) (wheels_per_car : ℕ) (h1 : num_cars = 12) (h2 : wheels_per_car = 4) :
  num_cars * wheels_per_car = 48 := by
  sorry

end NUMINAMATH_CALUDE_car_wheels_count_l2832_283225


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2832_283218

theorem square_sum_given_sum_and_product (m n : ℝ) 
  (h1 : m + n = 10) (h2 : m * n = 24) : m^2 + n^2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l2832_283218


namespace NUMINAMATH_CALUDE_audiobook_purchase_l2832_283223

theorem audiobook_purchase (audiobook_length : ℕ) (daily_listening : ℕ) (total_days : ℕ) : 
  audiobook_length = 30 → 
  daily_listening = 2 → 
  total_days = 90 → 
  (total_days * daily_listening) / audiobook_length = 6 := by
sorry

end NUMINAMATH_CALUDE_audiobook_purchase_l2832_283223


namespace NUMINAMATH_CALUDE_circle_properties_l2832_283208

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Theorem statement
theorem circle_properties :
  -- The center is on the y-axis
  ∃ b : ℝ, circle_equation 0 b ∧
  -- The radius is 1
  (∀ x y : ℝ, circle_equation x y → (x^2 + (y - 2)^2 = 1)) ∧
  -- The circle passes through (1, 2)
  circle_equation 1 2 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l2832_283208


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2832_283252

/-- Calculates the length of a bridge given train and crossing parameters. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) :
  train_length = 280 →
  train_speed = 18 →
  crossing_time = 20 →
  train_speed * crossing_time - train_length = 80 := by
  sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l2832_283252


namespace NUMINAMATH_CALUDE_proportion_problem_l2832_283216

theorem proportion_problem (y : ℝ) : 0.75 / 0.9 = 5 / y → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l2832_283216


namespace NUMINAMATH_CALUDE_ratio_closest_to_five_l2832_283201

theorem ratio_closest_to_five : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |(10^2000 + 10^2002) / (10^2001 + 10^2001) - 5| < ε :=
sorry

end NUMINAMATH_CALUDE_ratio_closest_to_five_l2832_283201


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l2832_283267

theorem quadratic_solution_difference_squared (α β : ℝ) : 
  α ≠ β ∧ 
  α^2 - 3*α + 2 = 0 ∧ 
  β^2 - 3*β + 2 = 0 → 
  (α - β)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l2832_283267


namespace NUMINAMATH_CALUDE_jessy_book_pages_l2832_283206

/-- The number of pages in Jessy's book -/
def book_pages : ℕ := 140

/-- The number of days to finish the book -/
def days_to_finish : ℕ := 7

/-- The number of reading sessions per day in the initial plan -/
def initial_sessions_per_day : ℕ := 3

/-- The number of pages per session in the initial plan -/
def initial_pages_per_session : ℕ := 6

/-- The additional pages required per day -/
def additional_pages_per_day : ℕ := 2

theorem jessy_book_pages :
  book_pages = 
    days_to_finish * 
    (initial_sessions_per_day * initial_pages_per_session + additional_pages_per_day) :=
by sorry

end NUMINAMATH_CALUDE_jessy_book_pages_l2832_283206


namespace NUMINAMATH_CALUDE_price_reduction_proof_l2832_283282

theorem price_reduction_proof (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.3
  let second_discount := 0.4
  let price_after_first := original_price * (1 - first_discount)
  let price_after_second := price_after_first * (1 - second_discount)
  let total_reduction := original_price - price_after_second
  let reduction_percentage := total_reduction / original_price * 100
  reduction_percentage = 58 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_proof_l2832_283282


namespace NUMINAMATH_CALUDE_equation_real_root_l2832_283253

theorem equation_real_root (k : ℝ) : ∃ x : ℝ, x = k^2 * (x - 1) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_real_root_l2832_283253


namespace NUMINAMATH_CALUDE_profit_maximized_at_six_l2832_283268

/-- Sales revenue function -/
def sales_revenue (x : ℝ) : ℝ := 17 * x^2

/-- Total production cost function -/
def total_cost (x : ℝ) : ℝ := 2 * x^3 - x^2

/-- Profit function -/
def profit (x : ℝ) : ℝ := sales_revenue x - total_cost x

/-- The production quantity that maximizes profit -/
def optimal_quantity : ℝ := 6

theorem profit_maximized_at_six :
  ∀ x > 0, profit x ≤ profit optimal_quantity :=
by sorry

end NUMINAMATH_CALUDE_profit_maximized_at_six_l2832_283268


namespace NUMINAMATH_CALUDE_lcm_16_24_45_l2832_283256

theorem lcm_16_24_45 : Nat.lcm (Nat.lcm 16 24) 45 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_16_24_45_l2832_283256


namespace NUMINAMATH_CALUDE_total_coins_is_188_l2832_283260

/-- The number of US pennies turned in -/
def us_pennies : ℕ := 38

/-- The number of US nickels turned in -/
def us_nickels : ℕ := 27

/-- The number of US dimes turned in -/
def us_dimes : ℕ := 19

/-- The number of US quarters turned in -/
def us_quarters : ℕ := 24

/-- The number of US half-dollars turned in -/
def us_half_dollars : ℕ := 13

/-- The number of US one-dollar coins turned in -/
def us_one_dollar_coins : ℕ := 17

/-- The number of US two-dollar coins turned in -/
def us_two_dollar_coins : ℕ := 5

/-- The number of Australian fifty-cent coins turned in -/
def australian_fifty_cent_coins : ℕ := 4

/-- The number of Mexican one-Peso coins turned in -/
def mexican_one_peso_coins : ℕ := 12

/-- The number of Canadian loonies turned in -/
def canadian_loonies : ℕ := 3

/-- The number of British 20 pence coins turned in -/
def british_20_pence_coins : ℕ := 7

/-- The number of pre-1965 US dimes turned in -/
def pre_1965_us_dimes : ℕ := 6

/-- The number of post-2005 Euro two-euro coins turned in -/
def euro_two_euro_coins : ℕ := 5

/-- The number of Swiss 5 franc coins turned in -/
def swiss_5_franc_coins : ℕ := 8

/-- Theorem: The total number of coins turned in is 188 -/
theorem total_coins_is_188 :
  us_pennies + us_nickels + us_dimes + us_quarters + us_half_dollars +
  us_one_dollar_coins + us_two_dollar_coins + australian_fifty_cent_coins +
  mexican_one_peso_coins + canadian_loonies + british_20_pence_coins +
  pre_1965_us_dimes + euro_two_euro_coins + swiss_5_franc_coins = 188 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_is_188_l2832_283260


namespace NUMINAMATH_CALUDE_ball_count_theorem_l2832_283207

theorem ball_count_theorem (total : ℕ) (red_freq black_freq : ℚ) :
  total = 120 →
  red_freq = 15 / 100 →
  black_freq = 45 / 100 →
  ∃ (red black white : ℕ),
    red = (total : ℚ) * red_freq ∧
    black = (total : ℚ) * black_freq ∧
    white = total - red - black ∧
    white = 48 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_theorem_l2832_283207


namespace NUMINAMATH_CALUDE_odd_sum_floor_condition_l2832_283203

theorem odd_sum_floor_condition (p a b : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) 
  (ha : 0 < a ∧ a < p) (hb : 0 < b ∧ b < p) :
  (a + b = p) ↔ 
  (∀ n : ℕ, 0 < n → n < p → 
    ∃ k : ℕ, k % 2 = 1 ∧ 
      (⌊(2 * a * n : ℚ) / p⌋ + ⌊(2 * b * n : ℚ) / p⌋ : ℤ) = k) :=
by sorry

end NUMINAMATH_CALUDE_odd_sum_floor_condition_l2832_283203


namespace NUMINAMATH_CALUDE_units_digit_of_2137_pow_753_l2832_283295

theorem units_digit_of_2137_pow_753 : ∃ n : ℕ, 2137^753 ≡ 7 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_2137_pow_753_l2832_283295


namespace NUMINAMATH_CALUDE_ellipse_equation_l2832_283246

/-- Given an ellipse with standard form equation, prove its specific equation -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1) → -- standard form of ellipse
  (3^2 = a^2 - b^2) →                    -- condition for right focus at (3,0)
  (9/b^2 = 1) →                          -- condition for point (0,-3) on ellipse
  (∀ (x y : ℝ), x^2/18 + y^2/9 = 1 ↔ x^2/a^2 + y^2/b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2832_283246


namespace NUMINAMATH_CALUDE_principal_amount_proof_l2832_283292

/-- Proves that given the conditions of the problem, the principal amount is 1500 --/
theorem principal_amount_proof (P : ℝ) : 
  (P * 0.04 * 4 = P - 1260) → P = 1500 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l2832_283292


namespace NUMINAMATH_CALUDE_playground_snow_volume_l2832_283280

/-- Represents a rectangular playground covered in snow -/
structure SnowCoveredPlayground where
  length : ℝ
  width : ℝ
  snowDepth : ℝ

/-- Calculates the volume of snow on a rectangular playground -/
def snowVolume (p : SnowCoveredPlayground) : ℝ :=
  p.length * p.width * p.snowDepth

/-- Theorem stating that the volume of snow on the given playground is 50 cubic feet -/
theorem playground_snow_volume :
  let p : SnowCoveredPlayground := {
    length := 40,
    width := 5,
    snowDepth := 0.25
  }
  snowVolume p = 50 := by sorry

end NUMINAMATH_CALUDE_playground_snow_volume_l2832_283280


namespace NUMINAMATH_CALUDE_inequality_solution_l2832_283277

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := (a * x) / (x - 1) < 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | x < 1 ∨ x > 3}

-- Theorem statement
theorem inequality_solution (a : ℝ) :
  (∀ x, x ∈ solution_set a ↔ inequality a x) → a = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_inequality_solution_l2832_283277


namespace NUMINAMATH_CALUDE_tom_share_calculation_l2832_283232

def total_amount : ℝ := 18500

def natalie_percentage : ℝ := 0.35
def rick_percentage : ℝ := 0.30
def lucy_percentage : ℝ := 0.40

def minimum_share : ℝ := 1000

def natalie_share : ℝ := natalie_percentage * total_amount
def remaining_after_natalie : ℝ := total_amount - natalie_share

def rick_share : ℝ := rick_percentage * remaining_after_natalie
def remaining_after_rick : ℝ := remaining_after_natalie - rick_share

def lucy_share : ℝ := lucy_percentage * remaining_after_rick
def tom_share : ℝ := remaining_after_rick - lucy_share

theorem tom_share_calculation :
  tom_share = 5050.50 ∧
  natalie_share ≥ minimum_share ∧
  rick_share ≥ minimum_share ∧
  lucy_share ≥ minimum_share ∧
  tom_share ≥ minimum_share :=
by sorry

end NUMINAMATH_CALUDE_tom_share_calculation_l2832_283232


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2832_283297

theorem trigonometric_simplification (α : Real) 
  (h1 : π/2 < α ∧ α < π) : 
  (Real.sqrt (1 + 2 * Real.sin (5 * π - α) * Real.cos (α - π))) / 
  (Real.sin (α - 3 * π / 2) - Real.sqrt (1 - Real.sin (3 * π / 2 + α) ^ 2)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2832_283297


namespace NUMINAMATH_CALUDE_P_intersect_Q_eq_P_l2832_283245

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x, y = Real.cos x}

theorem P_intersect_Q_eq_P : P ∩ Q = P := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_eq_P_l2832_283245


namespace NUMINAMATH_CALUDE_complex_modulus_one_l2832_283270

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l2832_283270


namespace NUMINAMATH_CALUDE_missing_edge_length_l2832_283236

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.edge1 * d.edge2 * d.edge3

/-- Theorem: Given a cuboid with two known edges 5 cm and 8 cm, and a volume of 80 cm³,
    the length of the third edge is 2 cm -/
theorem missing_edge_length :
  ∀ (x : ℝ),
    let d := CuboidDimensions.mk x 5 8
    cuboidVolume d = 80 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_missing_edge_length_l2832_283236


namespace NUMINAMATH_CALUDE_degree_of_minus_five_x_four_y_l2832_283239

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (coeff : ℤ) (x_exp y_exp : ℕ) : ℕ :=
  x_exp + y_exp

/-- The monomial -5x^4y has degree 5 -/
theorem degree_of_minus_five_x_four_y :
  degree_of_monomial (-5) 4 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_minus_five_x_four_y_l2832_283239


namespace NUMINAMATH_CALUDE_max_value_f_times_g_l2832_283212

noncomputable def f (x : ℝ) : ℝ := 3 - x

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (2 * x + 5)

def is_non_negative (x : ℝ) : Prop := x ≥ 0

theorem max_value_f_times_g :
  ∃ (M : ℝ), M = 2 * Real.sqrt 3 - 1 ∧
  (∀ (x : ℝ), is_non_negative x →
    (f x * g x = min (f x) (g x)) →
    f x * g x ≤ M) ∧
  (∃ (x : ℝ), is_non_negative x ∧
    (f x * g x = min (f x) (g x)) ∧
    f x * g x = M) :=
sorry

end NUMINAMATH_CALUDE_max_value_f_times_g_l2832_283212


namespace NUMINAMATH_CALUDE_janet_complaint_time_l2832_283278

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The time Janet spends looking for her keys daily (in minutes) -/
def daily_key_search_time : ℕ := 8

/-- The total time Janet saves weekly by not losing her keys (in minutes) -/
def weekly_time_saved : ℕ := 77

/-- The time Janet spends complaining after finding her keys daily (in minutes) -/
def daily_complaint_time : ℕ := (weekly_time_saved - days_in_week * daily_key_search_time) / days_in_week

theorem janet_complaint_time :
  daily_complaint_time = 3 :=
sorry

end NUMINAMATH_CALUDE_janet_complaint_time_l2832_283278


namespace NUMINAMATH_CALUDE_binomial_coeff_divisible_by_two_primes_l2832_283293

theorem binomial_coeff_divisible_by_two_primes (n k : ℕ) 
  (h1 : k > 1) (h2 : k < n - 1) : 
  ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ p ∣ Nat.choose n k ∧ q ∣ Nat.choose n k :=
by sorry

end NUMINAMATH_CALUDE_binomial_coeff_divisible_by_two_primes_l2832_283293


namespace NUMINAMATH_CALUDE_a_plus_b_values_l2832_283247

theorem a_plus_b_values (a b : ℝ) : 
  (abs a = 1) → (b = -2) → ((a + b = -1) ∨ (a + b = -3)) :=
by sorry

end NUMINAMATH_CALUDE_a_plus_b_values_l2832_283247


namespace NUMINAMATH_CALUDE_intersection_when_a_is_neg_two_intersection_equals_A_iff_l2832_283235

-- Define sets A and B
def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x + a < 0}

-- Theorem 1: When a = -2, A ∩ B = {x | 1/2 ≤ x < 2}
theorem intersection_when_a_is_neg_two :
  A ∩ B (-2) = {x : ℝ | 1/2 ≤ x ∧ x < 2} := by sorry

-- Theorem 2: A ∩ B = A if and only if a < -3
theorem intersection_equals_A_iff (a : ℝ) :
  A ∩ B a = A ↔ a < -3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_neg_two_intersection_equals_A_iff_l2832_283235


namespace NUMINAMATH_CALUDE_square_of_negative_two_m_squared_l2832_283286

theorem square_of_negative_two_m_squared (m : ℝ) : (-2 * m^2)^2 = 4 * m^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_m_squared_l2832_283286


namespace NUMINAMATH_CALUDE_dave_initial_tickets_l2832_283209

/-- The number of tickets Dave spent on a stuffed tiger -/
def spent_tickets : ℕ := 43

/-- The number of tickets Dave had left after the purchase -/
def remaining_tickets : ℕ := 55

/-- The initial number of tickets Dave had -/
def initial_tickets : ℕ := spent_tickets + remaining_tickets

theorem dave_initial_tickets : initial_tickets = 98 := by
  sorry

end NUMINAMATH_CALUDE_dave_initial_tickets_l2832_283209


namespace NUMINAMATH_CALUDE_min_value_expression_l2832_283265

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  b / (3 * a) + 3 / b ≥ 5 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ b₀ / (3 * a₀) + 3 / b₀ = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2832_283265


namespace NUMINAMATH_CALUDE_marys_brother_height_l2832_283290

/-- The height of Mary's brother given the conditions of the roller coaster problem -/
theorem marys_brother_height (min_height : ℝ) (mary_ratio : ℝ) (mary_growth : ℝ) :
  min_height = 140 ∧ mary_ratio = 2/3 ∧ mary_growth = 20 →
  ∃ (mary_height : ℝ) (brother_height : ℝ),
    mary_height + mary_growth = min_height ∧
    mary_height = mary_ratio * brother_height ∧
    brother_height = 180 := by
  sorry

end NUMINAMATH_CALUDE_marys_brother_height_l2832_283290


namespace NUMINAMATH_CALUDE_square_side_length_l2832_283215

theorem square_side_length : ∃ (X : ℝ), X = 2.6 ∧ 
  (∃ (A B C D : ℝ × ℝ),
    -- Four points inside the square
    (0 < A.1 ∧ A.1 < X) ∧ (0 < A.2 ∧ A.2 < X) ∧
    (0 < B.1 ∧ B.1 < X) ∧ (0 < B.2 ∧ B.2 < X) ∧
    (0 < C.1 ∧ C.1 < X) ∧ (0 < C.2 ∧ C.2 < X) ∧
    (0 < D.1 ∧ D.1 < X) ∧ (0 < D.2 ∧ D.2 < X) ∧
    -- Nine segments of length 1
    (A.1 - 0)^2 + (A.2 - 0)^2 = 1 ∧
    (B.1 - X)^2 + (B.2 - X)^2 = 1 ∧
    (C.1 - 0)^2 + (C.2 - X)^2 = 1 ∧
    (D.1 - X)^2 + (D.2 - 0)^2 = 1 ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = 1 ∧
    (D.1 - A.1)^2 + (D.2 - A.2)^2 = 1 ∧
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = 1 ∧
    (D.1 - C.1)^2 + (D.2 - C.2)^2 = 1 ∧
    -- Perpendicular segments
    A.1 = 0 ∧ B.1 = X ∧ C.2 = X ∧ D.2 = 0 ∧
    -- Distance conditions
    A.1 = (X - 1) / 2 ∧
    X - B.1 = 1) :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l2832_283215


namespace NUMINAMATH_CALUDE_second_box_clay_capacity_l2832_283264

/-- Represents the dimensions and clay capacity of a box -/
structure Box where
  height : ℝ
  width : ℝ
  length : ℝ
  clayCapacity : ℝ

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℝ := b.height * b.width * b.length

/-- Theorem stating the clay capacity of the second box -/
theorem second_box_clay_capacity 
  (box1 : Box)
  (box2 : Box)
  (h1 : box1.height = 4)
  (h2 : box1.width = 3)
  (h3 : box1.length = 7)
  (h4 : box1.clayCapacity = 84)
  (h5 : box2.height = box1.height / 2)
  (h6 : box2.width = box1.width * 4)
  (h7 : box2.length = box1.length)
  (h8 : boxVolume box1 * box1.clayCapacity = boxVolume box2 * box2.clayCapacity) :
  box2.clayCapacity = 168 := by
  sorry


end NUMINAMATH_CALUDE_second_box_clay_capacity_l2832_283264


namespace NUMINAMATH_CALUDE_median_in_70_74_interval_l2832_283250

/-- Represents a score interval with its lower bound and number of students -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (num_students : ℕ)

/-- Finds the interval containing the median score -/
def median_interval (intervals : List ScoreInterval) : Option ScoreInterval :=
  sorry

theorem median_in_70_74_interval :
  let intervals : List ScoreInterval := [
    ⟨55, 4⟩,
    ⟨60, 8⟩,
    ⟨65, 15⟩,
    ⟨70, 20⟩,
    ⟨75, 18⟩,
    ⟨80, 10⟩
  ]
  let total_students : ℕ := 75
  median_interval intervals = some ⟨70, 20⟩ := by
    sorry

end NUMINAMATH_CALUDE_median_in_70_74_interval_l2832_283250


namespace NUMINAMATH_CALUDE_balloon_ascent_rate_l2832_283296

/-- The rate of descent of the balloon in feet per minute -/
def descent_rate : ℝ := 10

/-- The duration of the first ascent in minutes -/
def first_ascent_duration : ℝ := 15

/-- The duration of the descent in minutes -/
def descent_duration : ℝ := 10

/-- The duration of the second ascent in minutes -/
def second_ascent_duration : ℝ := 15

/-- The maximum height reached by the balloon in feet -/
def max_height : ℝ := 1400

/-- The theorem stating the rate of ascent of the balloon -/
theorem balloon_ascent_rate :
  ∃ (ascent_rate : ℝ),
    ascent_rate * first_ascent_duration
    - descent_rate * descent_duration
    + ascent_rate * second_ascent_duration
    = max_height
    ∧ ascent_rate = 50 := by sorry

end NUMINAMATH_CALUDE_balloon_ascent_rate_l2832_283296


namespace NUMINAMATH_CALUDE_two_digit_product_equals_concatenation_l2832_283287

def has_same_digits (a b : ℕ) : Prop :=
  (Nat.log 10 a).succ = (Nat.log 10 b).succ

def concatenate (a b : ℕ) : ℕ :=
  a * (10 ^ ((Nat.log 10 b).succ)) + b

theorem two_digit_product_equals_concatenation :
  ∀ A B : ℕ,
    A > 0 ∧ B > 0 →
    has_same_digits A B →
    2 * A * B = concatenate A B →
    (A = 3 ∧ B = 6) ∨ (A = 13 ∧ B = 52) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_product_equals_concatenation_l2832_283287


namespace NUMINAMATH_CALUDE_hawks_score_l2832_283266

def total_points : ℕ := 82
def margin : ℕ := 22

theorem hawks_score (eagles_score hawks_score : ℕ) 
  (h1 : eagles_score + hawks_score = total_points)
  (h2 : eagles_score = hawks_score + margin) : 
  hawks_score = 30 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l2832_283266


namespace NUMINAMATH_CALUDE_notebook_cost_is_three_l2832_283240

/-- The cost of each notebook given the total spent, costs of other items, and number of notebooks. -/
def notebook_cost (total_spent backpack_cost pen_cost pencil_cost num_notebooks : ℚ) : ℚ :=
  (total_spent - (backpack_cost + pen_cost + pencil_cost)) / num_notebooks

/-- Theorem stating that the cost of each notebook is $3 given the problem conditions. -/
theorem notebook_cost_is_three :
  notebook_cost 32 15 1 1 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_is_three_l2832_283240


namespace NUMINAMATH_CALUDE_max_value_of_complex_difference_l2832_283237

theorem max_value_of_complex_difference (Z : ℂ) (h : Complex.abs Z = 1) :
  ∃ (max_val : ℝ), max_val = 6 ∧ ∀ (W : ℂ), Complex.abs W = 1 → Complex.abs (W - (3 - 4*I)) ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_value_of_complex_difference_l2832_283237


namespace NUMINAMATH_CALUDE_average_height_combined_groups_l2832_283249

theorem average_height_combined_groups (n₁ n₂ : ℕ) (h₁ h₂ : ℝ) :
  n₁ = 35 →
  n₂ = 25 →
  h₁ = 22 →
  h₂ = 18 →
  (n₁ * h₁ + n₂ * h₂) / (n₁ + n₂ : ℝ) = 20.33 :=
by sorry

end NUMINAMATH_CALUDE_average_height_combined_groups_l2832_283249

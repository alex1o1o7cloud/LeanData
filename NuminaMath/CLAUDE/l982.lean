import Mathlib

namespace total_balloons_l982_98286

def tom_balloons : ℕ := 9
def sara_balloons : ℕ := 8

theorem total_balloons : tom_balloons + sara_balloons = 17 := by
  sorry

end total_balloons_l982_98286


namespace trigonometric_problem_l982_98210

theorem trigonometric_problem (α : Real) 
  (h1 : 3 * Real.pi / 4 < α ∧ α < Real.pi) 
  (h2 : Real.tan α + 1 / Real.tan α = -10/3) : 
  Real.tan α = -1/3 ∧ 
  (Real.sin (Real.pi + α))^2 + 2 * Real.sin α * Real.sin (Real.pi/2 + α) + 1 / 
  (3 * Real.sin α * Real.cos (Real.pi/2 - α) - 2 * Real.cos α * Real.cos (Real.pi - α)) = 5/21 := by
  sorry

end trigonometric_problem_l982_98210


namespace circle_on_parabola_tangent_to_axes_l982_98262

/-- A circle whose center lies on a parabola and is tangent to the parabola's axis and y-axis -/
theorem circle_on_parabola_tangent_to_axes :
  ∃ (x₀ y₀ r : ℝ),
    (x₀ < 0) ∧                             -- Center is on the left side of y-axis
    (y₀ = (1/2) * x₀^2) ∧                  -- Center lies on the parabola
    (∀ x y : ℝ,
      (x + 1)^2 + (y - 1/2)^2 = 1 ↔        -- Equation of the circle
      (x - x₀)^2 + (y - y₀)^2 = r^2) ∧     -- Standard form of circle equation
    (r = |x₀|) ∧                           -- Circle is tangent to y-axis
    (r = |y₀ - 1/2|)                       -- Circle is tangent to parabola's axis
  := by sorry

end circle_on_parabola_tangent_to_axes_l982_98262


namespace percentage_markup_l982_98263

theorem percentage_markup (cost_price selling_price : ℝ) : 
  cost_price = 7000 →
  selling_price = 8400 →
  (selling_price - cost_price) / cost_price * 100 = 20 :=
by
  sorry

end percentage_markup_l982_98263


namespace consecutive_integers_perfect_square_product_specific_consecutive_integers_l982_98207

theorem consecutive_integers_perfect_square_product :
  ∃ (n : ℤ), (n - 1) * n * (n + 1) * (n + 2) = (n^2 + n - 1)^2 - 1 ∧
  ∃ (k : ℤ), (n^2 + n - 1)^2 = k^2 + 1 ∧
  (n = 0 ∨ n = -1 ∨ n = 1 ∨ n = -2) :=
by sorry

theorem specific_consecutive_integers :
  (-1 : ℤ) * 0 * 1 * 2 = 0^2 :=
by sorry

end consecutive_integers_perfect_square_product_specific_consecutive_integers_l982_98207


namespace cosine_identity_l982_98220

theorem cosine_identity (x : ℝ) (h1 : x ∈ Set.Ioo 0 π) (h2 : Real.cos (x - π/6) = -Real.sqrt 3 / 3) :
  Real.cos (x - π/3) = (-3 + Real.sqrt 6) / 6 := by
  sorry

end cosine_identity_l982_98220


namespace min_people_liking_both_l982_98271

theorem min_people_liking_both (total : ℕ) (mozart : ℕ) (beethoven : ℕ)
  (h1 : total = 150)
  (h2 : mozart = 130)
  (h3 : beethoven = 110)
  (h4 : mozart ≤ total)
  (h5 : beethoven ≤ total) :
  mozart + beethoven - total ≤ (min mozart beethoven) ∧
  (min mozart beethoven) = 90 :=
by sorry

end min_people_liking_both_l982_98271


namespace class_composition_l982_98257

/-- Represents a child's response about the number of classmates -/
structure Response :=
  (boys : ℕ)
  (girls : ℕ)

/-- Checks if a response is valid according to the problem conditions -/
def is_valid_response (actual_boys actual_girls : ℕ) (r : Response) : Prop :=
  (r.boys = actual_boys ∧ (r.girls = actual_girls + 2 ∨ r.girls = actual_girls - 2)) ∨
  (r.girls = actual_girls ∧ (r.boys = actual_boys + 2 ∨ r.boys = actual_boys - 2))

/-- The main theorem stating the correct number of boys and girls in the class -/
theorem class_composition :
  ∃ (actual_boys actual_girls : ℕ),
    actual_boys = 15 ∧
    actual_girls = 12 ∧
    is_valid_response actual_boys actual_girls ⟨13, 11⟩ ∧
    is_valid_response actual_boys actual_girls ⟨17, 11⟩ ∧
    is_valid_response actual_boys actual_girls ⟨14, 14⟩ :=
  sorry

end class_composition_l982_98257


namespace exists_term_with_100_nines_l982_98285

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticProgression (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A number contains 100 consecutive nines if it can be written in the form
    k * 10^(100 + m) + (10^100 - 1) for some natural numbers k and m. -/
def Contains100ConsecutiveNines (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = k * (10^(100 + m)) + (10^100 - 1)

/-- In any infinite arithmetic progression of natural numbers,
    there exists a term that contains 100 consecutive nines. -/
theorem exists_term_with_100_nines (a : ℕ → ℕ) (h : ArithmeticProgression a) :
  ∃ n : ℕ, Contains100ConsecutiveNines (a n) := by
  sorry


end exists_term_with_100_nines_l982_98285


namespace aunt_gift_amount_l982_98266

theorem aunt_gift_amount (jade_initial julia_initial jack_initial total_after_gift : ℕ) : 
  jade_initial = 38 →
  julia_initial = jade_initial / 2 →
  jack_initial = 12 →
  total_after_gift = 132 →
  ∃ gift : ℕ, 
    jade_initial + julia_initial + jack_initial + 3 * gift = total_after_gift ∧
    gift = 21 := by
  sorry

end aunt_gift_amount_l982_98266


namespace complex_equation_solution_l982_98291

theorem complex_equation_solution (z : ℂ) : (z * Complex.I = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end complex_equation_solution_l982_98291


namespace factory_temporary_workers_percentage_l982_98224

theorem factory_temporary_workers_percentage 
  (total_workers : ℕ) 
  (technicians : ℕ) 
  (non_technicians : ℕ) 
  (permanent_technicians : ℕ) 
  (permanent_non_technicians : ℕ) 
  (h1 : technicians + non_technicians = total_workers)
  (h2 : technicians = non_technicians)
  (h3 : permanent_technicians = technicians / 2)
  (h4 : permanent_non_technicians = non_technicians / 2)
  : (total_workers - (permanent_technicians + permanent_non_technicians)) / total_workers = 1/2 := by
  sorry

end factory_temporary_workers_percentage_l982_98224


namespace z_value_theorem_l982_98243

theorem z_value_theorem (z w : ℝ) (hz : z ≠ 0) (hw : w ≠ 0)
  (h1 : z + 1 / w = 15) (h2 : w^2 + 1 / z = 3) : z = 44 / 3 := by
  sorry

end z_value_theorem_l982_98243


namespace sqrt_expression_equals_six_l982_98284

theorem sqrt_expression_equals_six :
  (Real.sqrt 3 - 1)^2 + Real.sqrt 12 + (1/2)⁻¹ = 6 := by
  sorry

end sqrt_expression_equals_six_l982_98284


namespace exterior_angle_regular_pentagon_exterior_angle_regular_pentagon_proof_l982_98252

/-- The size of an exterior angle of a regular pentagon is 72 degrees. -/
theorem exterior_angle_regular_pentagon : ℝ :=
  72

/-- The number of sides in a pentagon. -/
def pentagon_sides : ℕ := 5

/-- The sum of exterior angles of any polygon in degrees. -/
def sum_exterior_angles : ℝ := 360

/-- Theorem: The size of an exterior angle of a regular pentagon is 72 degrees. -/
theorem exterior_angle_regular_pentagon_proof :
  exterior_angle_regular_pentagon = sum_exterior_angles / pentagon_sides :=
by sorry

end exterior_angle_regular_pentagon_exterior_angle_regular_pentagon_proof_l982_98252


namespace perimeter_stones_count_l982_98234

/-- Given a square arrangement of stones with 5 stones on each side,
    the number of stones on the perimeter is 16. -/
theorem perimeter_stones_count (side_length : ℕ) (h : side_length = 5) :
  4 * side_length - 4 = 16 := by
  sorry

end perimeter_stones_count_l982_98234


namespace angle_supplement_complement_difference_l982_98227

theorem angle_supplement_complement_difference (α : ℝ) : (180 - α) - (90 - α) = 90 := by
  sorry

end angle_supplement_complement_difference_l982_98227


namespace total_spider_legs_is_33_l982_98206

/-- The total number of spider legs in a room with 5 spiders -/
def total_spider_legs : ℕ :=
  let spider1 := 6
  let spider2 := 7
  let spider3 := 8
  let spider4 := 5
  let spider5 := 7
  spider1 + spider2 + spider3 + spider4 + spider5

/-- Theorem stating that the total number of spider legs is 33 -/
theorem total_spider_legs_is_33 : total_spider_legs = 33 := by
  sorry

end total_spider_legs_is_33_l982_98206


namespace exam_exemption_logic_l982_98256

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (score_above_90 : Student → Prop)
variable (exempted : Student → Prop)

-- State the theorem
theorem exam_exemption_logic (s : Student) 
  (h : ∀ x, score_above_90 x → exempted x) :
  ¬(exempted s) → ¬(score_above_90 s) := by
  sorry

end exam_exemption_logic_l982_98256


namespace jessica_bank_account_l982_98283

theorem jessica_bank_account (initial_balance : ℝ) 
  (withdrawal : ℝ) (final_balance : ℝ) (deposit_fraction : ℝ) :
  withdrawal = 200 ∧
  initial_balance - withdrawal = (3/5) * initial_balance ∧
  final_balance = 360 ∧
  final_balance = (initial_balance - withdrawal) + deposit_fraction * (initial_balance - withdrawal) →
  deposit_fraction = 1/5 := by
  sorry

end jessica_bank_account_l982_98283


namespace f_symmetry_and_increase_l982_98287

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 - 1/2

def is_center_of_symmetry (c : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (c.1 + x) = f (c.1 - x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_symmetry_and_increase :
  (∀ k : ℤ, is_center_of_symmetry (k * Real.pi / 2 + Real.pi / 12, -1) f) ∧
  is_increasing_on f 0 (Real.pi / 3) ∧
  is_increasing_on f (5 * Real.pi / 6) Real.pi :=
sorry

end f_symmetry_and_increase_l982_98287


namespace trapezoid_midline_length_l982_98245

/-- Given a trapezoid with parallel sides of length a and b, 
    the length of the line segment joining the midpoints of these parallel sides is (a + b) / 2 -/
theorem trapezoid_midline_length (a b c d : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  let midline_length := (a + b) / 2
  midline_length = (a + b) / 2 := by
  sorry


end trapezoid_midline_length_l982_98245


namespace keychain_manufacturing_cost_l982_98265

theorem keychain_manufacturing_cost 
  (P : ℝ) -- Selling price
  (initial_cost : ℝ) -- Initial manufacturing cost
  (initial_profit_percentage : ℝ) -- Initial profit percentage
  (new_profit_percentage : ℝ) -- New profit percentage
  (h1 : initial_cost = 65) -- Initial cost is $65
  (h2 : P - initial_cost = initial_profit_percentage * P) -- Initial profit equation
  (h3 : initial_profit_percentage = 0.35) -- Initial profit is 35%
  (h4 : new_profit_percentage = 0.50) -- New profit is 50%
  : ∃ C, P - C = new_profit_percentage * P ∧ C = 50 := by
sorry

end keychain_manufacturing_cost_l982_98265


namespace outfits_count_l982_98209

/-- The number of different outfits that can be formed by choosing one top and one pair of pants -/
def number_of_outfits (num_tops : ℕ) (num_pants : ℕ) : ℕ :=
  num_tops * num_pants

/-- Theorem stating that with 4 tops and 3 pants, the number of outfits is 12 -/
theorem outfits_count : number_of_outfits 4 3 = 12 := by
  sorry

end outfits_count_l982_98209


namespace pirate_treasure_l982_98277

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 := by
  sorry

end pirate_treasure_l982_98277


namespace clothing_size_puzzle_l982_98274

theorem clothing_size_puzzle (anna_size becky_size ginger_size subtracted_number : ℕ) : 
  anna_size = 2 →
  becky_size = 3 * anna_size →
  ginger_size = 2 * becky_size - subtracted_number →
  ginger_size = 8 →
  subtracted_number = 4 := by
sorry

end clothing_size_puzzle_l982_98274


namespace janette_breakfast_jerky_l982_98202

/-- The number of days Janette went camping -/
def camping_days : ℕ := 5

/-- The initial number of beef jerky pieces Janette brought -/
def initial_jerky : ℕ := 40

/-- The number of beef jerky pieces Janette eats for lunch each day -/
def lunch_jerky : ℕ := 1

/-- The number of beef jerky pieces Janette eats for dinner each day -/
def dinner_jerky : ℕ := 2

/-- The number of beef jerky pieces Janette has left after giving half to her brother -/
def final_jerky : ℕ := 10

/-- The number of beef jerky pieces Janette eats for breakfast each day -/
def breakfast_jerky : ℕ := 1

theorem janette_breakfast_jerky :
  breakfast_jerky = 1 ∧
  camping_days * (breakfast_jerky + lunch_jerky + dinner_jerky) = initial_jerky - 2 * final_jerky :=
by sorry

end janette_breakfast_jerky_l982_98202


namespace sum_PV_squared_l982_98288

-- Define the triangle PQR
def PQR : Set (ℝ × ℝ) := sorry

-- Define the property of PQR being equilateral with side length 10
def is_equilateral_10 (triangle : Set (ℝ × ℝ)) : Prop := sorry

-- Define the four triangles PU₁V₁, PU₁V₂, PU₂V₃, and PU₂V₄
def PU1V1 : Set (ℝ × ℝ) := sorry
def PU1V2 : Set (ℝ × ℝ) := sorry
def PU2V3 : Set (ℝ × ℝ) := sorry
def PU2V4 : Set (ℝ × ℝ) := sorry

-- Define the property of a triangle being congruent to PQR
def is_congruent_to_PQR (triangle : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property QU₁ = QU₂ = 3
def QU1_QU2_eq_3 : Prop := sorry

-- Define the function to calculate PVₖ
def PV (k : ℕ) : ℝ := sorry

-- Theorem statement
theorem sum_PV_squared :
  is_equilateral_10 PQR ∧
  is_congruent_to_PQR PU1V1 ∧
  is_congruent_to_PQR PU1V2 ∧
  is_congruent_to_PQR PU2V3 ∧
  is_congruent_to_PQR PU2V4 ∧
  QU1_QU2_eq_3 →
  (PV 1)^2 + (PV 2)^2 + (PV 3)^2 + (PV 4)^2 = 800 := by sorry

end sum_PV_squared_l982_98288


namespace last_digit_product_l982_98294

theorem last_digit_product : (3^65 * 6^59 * 7^71) % 10 = 4 := by sorry

end last_digit_product_l982_98294


namespace vector_properties_l982_98249

/-- Given vectors in R², prove perpendicularity implies tan(α + β) = 2
    and tan(α)tan(β) = 16 implies vectors are parallel -/
theorem vector_properties (α β : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ)
  (ha : a = (4 * Real.cos α, Real.sin α))
  (hb : b = (Real.sin β, 4 * Real.cos β))
  (hc : c = (Real.cos β, -4 * Real.sin β)) :
  (a.1 * (b.1 - 2*c.1) + a.2 * (b.2 - 2*c.2) = 0 → Real.tan (α + β) = 2) ∧
  (Real.tan α * Real.tan β = 16 → ∃ (k : ℝ), a = k • b) :=
by sorry

end vector_properties_l982_98249


namespace longest_collection_pages_l982_98299

/-- Represents the number of pages per inch for a book collection -/
structure PagesPerInch where
  value : ℕ

/-- Represents the height of a book collection in inches -/
structure CollectionHeight where
  value : ℕ

/-- Calculates the total number of pages in a collection -/
def total_pages (ppi : PagesPerInch) (height : CollectionHeight) : ℕ :=
  ppi.value * height.value

/-- Represents Miles's book collection -/
def miles_collection : PagesPerInch × CollectionHeight :=
  ({ value := 5 }, { value := 240 })

/-- Represents Daphne's book collection -/
def daphne_collection : PagesPerInch × CollectionHeight :=
  ({ value := 50 }, { value := 25 })

theorem longest_collection_pages : 
  max (total_pages miles_collection.1 miles_collection.2)
      (total_pages daphne_collection.1 daphne_collection.2) = 1250 := by
  sorry

end longest_collection_pages_l982_98299


namespace solution_set_a_2_range_of_a_l982_98281

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end solution_set_a_2_range_of_a_l982_98281


namespace fraction_multiplication_simplification_l982_98297

theorem fraction_multiplication_simplification :
  (270 : ℚ) / 18 * (7 : ℚ) / 210 * (12 : ℚ) / 4 = (3 : ℚ) / 2 := by
  sorry

end fraction_multiplication_simplification_l982_98297


namespace number_composition_l982_98216

def number_from_parts (ten_thousands : ℕ) (ones : ℕ) : ℕ :=
  ten_thousands * 10000 + ones

theorem number_composition :
  number_from_parts 45 64 = 450064 := by
  sorry

end number_composition_l982_98216


namespace persons_count_l982_98282

/-- The total number of persons in the group --/
def n : ℕ := sorry

/-- The total amount spent by the group in rupees --/
def total_spent : ℚ := 292.5

/-- The amount spent by each of the first 8 persons in rupees --/
def regular_spend : ℚ := 30

/-- The number of persons who spent the regular amount --/
def regular_count : ℕ := 8

/-- The extra amount spent by the last person compared to the average --/
def extra_spend : ℚ := 20

theorem persons_count :
  n = 9 ∧
  total_spent = regular_count * regular_spend + (total_spent / n + extra_spend) :=
sorry

end persons_count_l982_98282


namespace birdhouse_revenue_theorem_l982_98293

/-- Calculates the total revenue from selling birdhouses with discount and tax --/
def birdhouse_revenue (
  extra_large_price : ℚ)
  (large_price : ℚ)
  (medium_price : ℚ)
  (small_price : ℚ)
  (extra_small_price : ℚ)
  (extra_large_qty : ℕ)
  (large_qty : ℕ)
  (medium_qty : ℕ)
  (small_qty : ℕ)
  (extra_small_qty : ℕ)
  (discount_rate : ℚ)
  (tax_rate : ℚ) : ℚ :=
  let total_before_discount :=
    extra_large_price * extra_large_qty +
    large_price * large_qty +
    medium_price * medium_qty +
    small_price * small_qty +
    extra_small_price * extra_small_qty
  let discounted_amount := total_before_discount * (1 - discount_rate)
  let final_amount := discounted_amount * (1 + tax_rate)
  final_amount

/-- Theorem stating the total revenue from selling birdhouses --/
theorem birdhouse_revenue_theorem :
  birdhouse_revenue 45 22 16 10 5 3 5 7 8 10 (1/10) (6/100) = 464.60 := by
  sorry

end birdhouse_revenue_theorem_l982_98293


namespace handshake_count_l982_98205

/-- The number of handshakes in a conference of 25 people -/
def conference_handshakes : ℕ := 300

/-- The number of attendees at the conference -/
def num_attendees : ℕ := 25

theorem handshake_count :
  (num_attendees.choose 2 : ℕ) = conference_handshakes :=
sorry

end handshake_count_l982_98205


namespace log_expression_equals_two_l982_98259

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  2 * (log10 2) + log10 25 = 2 := by sorry

end log_expression_equals_two_l982_98259


namespace zhe_same_meaning_and_usage_l982_98241

/-- Represents a function word in classical Chinese --/
structure FunctionWord where
  word : String
  meaning : String
  usage : String

/-- Represents a sentence in classical Chinese --/
structure Sentence where
  text : String
  functionWords : List FunctionWord

/-- The function word "者" as it appears in the first sentence --/
def zhe1 : FunctionWord := {
  word := "者",
  meaning := "the person",
  usage := "nominalizer"
}

/-- The function word "者" as it appears in the second sentence --/
def zhe2 : FunctionWord := {
  word := "者",
  meaning := "the person",
  usage := "nominalizer"
}

/-- The first sentence containing "者" --/
def sentence1 : Sentence := {
  text := "智者能勿丧",
  functionWords := [zhe1]
}

/-- The second sentence containing "者" --/
def sentence2 : Sentence := {
  text := "所知贫穷者，将从我乎？",
  functionWords := [zhe2]
}

/-- Theorem stating that the function word "者" has the same meaning and usage in both sentences --/
theorem zhe_same_meaning_and_usage : 
  zhe1.meaning = zhe2.meaning ∧ zhe1.usage = zhe2.usage :=
sorry

end zhe_same_meaning_and_usage_l982_98241


namespace square_semicircle_diagonal_l982_98248

-- Define the square and semicircle
def Square (A B C D : ℝ × ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ 
    B.1 - A.1 = s ∧ B.2 - A.2 = 0 ∧
    C.1 - B.1 = 0 ∧ C.2 - B.2 = s ∧
    D.1 - C.1 = -s ∧ D.2 - C.2 = 0 ∧
    A.1 - D.1 = 0 ∧ A.2 - D.2 = -s

def Semicircle (O : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2 ∧
    (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2 ∧
    B.1 - A.1 = 2 * r ∧ B.2 = A.2

-- Define the theorem
theorem square_semicircle_diagonal (A B C D M : ℝ × ℝ) :
  Square A B C D →
  Semicircle ((A.1 + B.1) / 2, A.2) A B →
  B.1 - A.1 = 8 →
  M.1 = (A.1 + B.1) / 2 ∧ M.2 - A.2 = 4 →
  (M.1 - D.1)^2 + (M.2 - D.2)^2 = 160 :=
sorry

end square_semicircle_diagonal_l982_98248


namespace weight_of_BaF2_l982_98214

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of moles of BaF2 -/
def moles_BaF2 : ℝ := 6

/-- The molecular weight of BaF2 in g/mol -/
def molecular_weight_BaF2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_F

/-- The weight of BaF2 in grams -/
def weight_BaF2 : ℝ := moles_BaF2 * molecular_weight_BaF2

theorem weight_of_BaF2 : weight_BaF2 = 1051.98 := by
  sorry

end weight_of_BaF2_l982_98214


namespace voronovich_inequality_l982_98296

theorem voronovich_inequality (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 1) :
  (a^2 + b^2 + c^2)^2 + 6*a*b*c ≥ a*b + b*c + c*a := by
  sorry

end voronovich_inequality_l982_98296


namespace faye_candy_problem_l982_98246

theorem faye_candy_problem (initial : ℕ) (received : ℕ) (final : ℕ) (eaten : ℕ) : 
  initial = 47 → received = 40 → final = 62 → 
  initial - eaten + received = final → 
  eaten = 25 := by sorry

end faye_candy_problem_l982_98246


namespace stanley_lemonade_sales_l982_98218

/-- The number of cups of lemonade Carl sells per hour -/
def carl_cups_per_hour : ℕ := 7

/-- The number of hours considered -/
def hours : ℕ := 3

/-- The difference in cups sold between Carl and Stanley over 3 hours -/
def difference_in_cups : ℕ := 9

/-- The number of cups of lemonade Stanley sells per hour -/
def stanley_cups_per_hour : ℕ := 4

theorem stanley_lemonade_sales :
  stanley_cups_per_hour * hours + difference_in_cups = carl_cups_per_hour * hours := by
  sorry

end stanley_lemonade_sales_l982_98218


namespace mathville_running_difference_l982_98250

/-- The side length of a square block in Mathville -/
def block_side_length : ℝ := 500

/-- The width of streets in Mathville -/
def street_width : ℝ := 30

/-- The length of Matt's path around the block -/
def matt_path_length : ℝ := 4 * block_side_length

/-- The length of Mike's path around the block -/
def mike_path_length : ℝ := 4 * (block_side_length + 2 * street_width)

/-- The difference between Mike's and Matt's path lengths -/
def path_length_difference : ℝ := mike_path_length - matt_path_length

theorem mathville_running_difference : path_length_difference = 240 := by
  sorry

end mathville_running_difference_l982_98250


namespace hyperbola_m_range_l982_98240

-- Define the hyperbola equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (4 - m) - y^2 / (2 + m) = 1

-- Theorem statement
theorem hyperbola_m_range (m : ℝ) :
  is_hyperbola m → m > -2 ∧ m < 4 :=
by
  sorry

end hyperbola_m_range_l982_98240


namespace stool_height_is_53_l982_98236

/-- The height of the stool Alice needs to reach the light bulb -/
def stool_height (ceiling_height floor_dip alice_height hat_height reach_above_head light_bulb_distance : ℝ) : ℝ :=
  ceiling_height * 100 - light_bulb_distance - (alice_height * 100 + hat_height + reach_above_head - floor_dip)

/-- Theorem stating that the stool height is 53 cm given the problem conditions -/
theorem stool_height_is_53 :
  stool_height 2.8 3 1.6 5 50 15 = 53 := by
  sorry

end stool_height_is_53_l982_98236


namespace equivalent_rotation_l982_98292

/-- Given a full rotation of 450 degrees, if a point is rotated 650 degrees clockwise
    to reach a destination, then the equivalent counterclockwise rotation to reach
    the same destination is 250 degrees. -/
theorem equivalent_rotation (full_rotation : ℕ) (clockwise_rotation : ℕ) (counterclockwise_rotation : ℕ) : 
  full_rotation = 450 → 
  clockwise_rotation = 650 → 
  counterclockwise_rotation < full_rotation →
  (clockwise_rotation % full_rotation + counterclockwise_rotation) % full_rotation = 0 →
  counterclockwise_rotation = 250 := by
  sorry

#check equivalent_rotation

end equivalent_rotation_l982_98292


namespace min_sum_squares_l982_98279

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) 
  (pos₁ : 0 < y₁) (pos₂ : 0 < y₂) (pos₃ : 0 < y₃)
  (sum_eq : y₁ + 3 * y₂ + 5 * y₃ = 120) :
  y₁^2 + y₂^2 + y₃^2 ≥ 43200 / 361 ∧
  ∃ y₁' y₂' y₃' : ℝ, 
    0 < y₁' ∧ 0 < y₂' ∧ 0 < y₃' ∧
    y₁' + 3 * y₂' + 5 * y₃' = 120 ∧
    y₁'^2 + y₂'^2 + y₃'^2 = 43200 / 361 :=
by sorry

end min_sum_squares_l982_98279


namespace f_increasing_condition_f_extremum_at_3_f_max_value_f_min_value_l982_98200

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - 3*x

-- Part 1: f(x) is increasing on [1, +∞) iff a ≤ 0
theorem f_increasing_condition (a : ℝ) :
  (∀ x ≥ 1, Monotone (f a)) ↔ a ≤ 0 := by sorry

-- Part 2: When x = 3 is an extremum point
theorem f_extremum_at_3 (a : ℝ) :
  (∃ x, HasDerivAt (f a) 0 x) → a = 6 := by sorry

-- Maximum value of f(x) on [1, 6] is -6
theorem f_max_value :
  ∃ x ∈ Set.Icc 1 6, ∀ y ∈ Set.Icc 1 6, f 6 y ≤ f 6 x ∧ f 6 x = -6 := by sorry

-- Minimum value of f(x) on [1, 6] is -18
theorem f_min_value :
  ∃ x ∈ Set.Icc 1 6, ∀ y ∈ Set.Icc 1 6, f 6 x ≤ f 6 y ∧ f 6 x = -18 := by sorry

end f_increasing_condition_f_extremum_at_3_f_max_value_f_min_value_l982_98200


namespace sqrt_300_simplified_l982_98239

theorem sqrt_300_simplified : Real.sqrt 300 = 10 * Real.sqrt 3 := by
  sorry

end sqrt_300_simplified_l982_98239


namespace dog_food_calculation_l982_98204

/-- Calculates the total amount of dog food needed per day for a given list of dog weights -/
def totalDogFood (weights : List ℕ) : ℕ :=
  (weights.map (· / 10)).sum

/-- Theorem: Given five dogs with specific weights, the total dog food needed is 15 pounds -/
theorem dog_food_calculation :
  totalDogFood [20, 40, 10, 30, 50] = 15 := by
  sorry

end dog_food_calculation_l982_98204


namespace least_positive_integer_with_remainder_one_l982_98298

theorem least_positive_integer_with_remainder_one (n : ℕ) : 
  (n > 1) →
  (n % 3 = 1) →
  (n % 4 = 1) →
  (n % 5 = 1) →
  (n % 6 = 1) →
  (n % 7 = 1) →
  (n % 10 = 1) →
  (n % 11 = 1) →
  (∀ m : ℕ, m > 1 → 
    (m % 3 = 1) →
    (m % 4 = 1) →
    (m % 5 = 1) →
    (m % 6 = 1) →
    (m % 7 = 1) →
    (m % 10 = 1) →
    (m % 11 = 1) →
    (n ≤ m)) →
  n = 4621 :=
by
  sorry

end least_positive_integer_with_remainder_one_l982_98298


namespace polynomial_evaluation_l982_98276

theorem polynomial_evaluation :
  ∀ y : ℝ, y > 0 → y^2 - 3*y - 9 = 0 → y^3 - 3*y^2 - 9*y + 7 = 7 := by
  sorry

end polynomial_evaluation_l982_98276


namespace daily_profit_properties_l982_98217

/-- Represents the daily sales profit function for a company -/
def daily_profit (x : ℝ) : ℝ := 10 * x^2 - 80 * x

/-- Theorem stating the properties of the daily sales profit function -/
theorem daily_profit_properties :
  -- The daily profit function is correct
  (∀ x, daily_profit x = 10 * x^2 - 80 * x) ∧
  -- When the selling price increases by 3 yuan, the daily profit is 350 yuan
  (daily_profit 3 = 350) ∧
  -- When the daily profit is 360 yuan, the selling price has increased by 4 yuan
  (daily_profit 4 = 360) := by
  sorry


end daily_profit_properties_l982_98217


namespace inscribed_circle_radius_l982_98251

/-- Given three mutually externally tangent circles with radii a, b, and c,
    the radius r of the inscribed circle satisfies the equation:
    1/r = 1/a + 1/b + 1/c + 2 * sqrt(1/(a*b) + 1/(a*c) + 1/(b*c)) -/
theorem inscribed_circle_radius (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  a = 5 → b = 10 → c = 20 → r = 20 / (3.5 + 2 * Real.sqrt 14) :=
by sorry

end inscribed_circle_radius_l982_98251


namespace binomial_coefficient_equality_l982_98233

theorem binomial_coefficient_equality (n : ℕ) (r : ℕ) : 
  (Nat.choose n (4*r - 1) = Nat.choose n (r + 1)) → 
  (n = 20 ∧ r = 4) := by
  sorry

end binomial_coefficient_equality_l982_98233


namespace rational_expression_evaluation_l982_98232

theorem rational_expression_evaluation :
  let x : ℝ := 7
  (x^8 + 18*x^4 + 81) / (x^4 + 9) = 2410 := by
  sorry

end rational_expression_evaluation_l982_98232


namespace ellipse_right_angle_triangle_area_l982_98290

/-- The ellipse with equation x²/2 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}

/-- The foci of the ellipse -/
def Foci : Set (ℝ × ℝ) :=
  {f : ℝ × ℝ | ∃ (x y : ℝ), f = (x, y) ∧ x^2 + y^2 = 1}

/-- Angle between two vectors -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- Area of a triangle given by three points -/
def triangleArea (a b c : ℝ × ℝ) : ℝ := sorry

theorem ellipse_right_angle_triangle_area 
  (p : ℝ × ℝ) (f₁ f₂ : ℝ × ℝ) 
  (h_p : p ∈ Ellipse) 
  (h_f : f₁ ∈ Foci ∧ f₂ ∈ Foci ∧ f₁ ≠ f₂) 
  (h_angle : angle (f₁.1 - p.1, f₁.2 - p.2) (f₂.1 - p.1, f₂.2 - p.2) = π / 2) :
  triangleArea f₁ p f₂ = 1 := by
  sorry

end ellipse_right_angle_triangle_area_l982_98290


namespace determinant_of_specific_matrix_l982_98226

theorem determinant_of_specific_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![8, 4; -1, 5]
  Matrix.det A = 44 := by
sorry

end determinant_of_specific_matrix_l982_98226


namespace zoo_meat_supply_duration_l982_98242

/-- The number of full days a meat supply lasts for a group of animals -/
def meat_supply_duration (lion_consumption tiger_consumption leopard_consumption hyena_consumption total_meat : ℕ) : ℕ :=
  (total_meat / (lion_consumption + tiger_consumption + leopard_consumption + hyena_consumption))

/-- Theorem: Given the specified daily meat consumption for four animals and a total meat supply of 500 kg, the meat supply will last for 7 full days -/
theorem zoo_meat_supply_duration :
  meat_supply_duration 25 20 15 10 500 = 7 := by
  sorry

end zoo_meat_supply_duration_l982_98242


namespace xanadu_license_plates_l982_98267

/-- The number of possible letters in each letter position of a Xanadu license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in each digit position of a Xanadu license plate. -/
def num_digits : ℕ := 10

/-- The total number of valid license plates in Xanadu. -/
def total_license_plates : ℕ := num_letters^4 * num_digits^2

/-- Theorem stating the total number of valid license plates in Xanadu. -/
theorem xanadu_license_plates : total_license_plates = 45697600 := by
  sorry

end xanadu_license_plates_l982_98267


namespace expression_simplification_and_evaluation_l982_98231

theorem expression_simplification_and_evaluation :
  let x : ℚ := 1/3
  let y : ℚ := -6
  let original_expression := 3 * x^2 * y - (6 * x * y^2 - 2 * (x * y + 3/2 * x^2 * y)) + 2 * (3 * x * y^2 - x * y)
  let simplified_expression := 6 * x^2 * y
  original_expression = simplified_expression ∧ simplified_expression = -4 :=
by sorry

end expression_simplification_and_evaluation_l982_98231


namespace pages_copied_l982_98268

/-- Given the cost of 7 cents for 5 pages, prove that $35 allows copying 2500 pages. -/
theorem pages_copied (cost_per_5_pages : ℚ) (total_dollars : ℚ) : 
  cost_per_5_pages = 7 / 100 → 
  total_dollars = 35 → 
  (total_dollars * 100 * 5) / cost_per_5_pages = 2500 := by
  sorry

end pages_copied_l982_98268


namespace mistaken_divisor_problem_l982_98201

theorem mistaken_divisor_problem (dividend : ℕ) (mistaken_divisor : ℕ) :
  dividend % 21 = 0 →
  dividend / 21 = 28 →
  dividend / mistaken_divisor = 49 →
  mistaken_divisor = 12 := by
  sorry

end mistaken_divisor_problem_l982_98201


namespace value_of_expression_constant_difference_implies_b_value_l982_98215

/-- Definition of A in terms of a and b -/
def A (a b : ℝ) : ℝ := 2*a^2 + 3*a*b - 2*a - 1

/-- Definition of B in terms of a and b -/
def B (a b : ℝ) : ℝ := a^2 + a*b - 1

/-- Theorem 1: The value of 4A - (3A - 2B) -/
theorem value_of_expression (a b : ℝ) :
  4 * A a b - (3 * A a b - 2 * B a b) = 4*a^2 + 5*a*b - 2*a - 3 := by sorry

/-- Theorem 2: When A - 2B is constant for all a, b must equal 2 -/
theorem constant_difference_implies_b_value (b : ℝ) :
  (∀ a : ℝ, ∃ k : ℝ, A a b - 2 * B a b = k) → b = 2 := by sorry

end value_of_expression_constant_difference_implies_b_value_l982_98215


namespace max_min_x_plus_reciprocal_l982_98213

theorem max_min_x_plus_reciprocal (x : ℝ) (h : 12 = x^2 + 1/x^2) :
  (∀ y : ℝ, y ≠ 0 → 12 = y^2 + 1/y^2 → x + 1/x ≤ Real.sqrt 14) ∧
  (∀ y : ℝ, y ≠ 0 → 12 = y^2 + 1/y^2 → -Real.sqrt 14 ≤ x + 1/x) :=
by sorry

end max_min_x_plus_reciprocal_l982_98213


namespace three_std_dev_below_mean_l982_98260

/-- Represents a normal distribution --/
structure NormalDistribution where
  mean : ℝ
  stdDev : ℝ

/-- Calculates the value that is n standard deviations below the mean --/
def valueBelow (nd : NormalDistribution) (n : ℝ) : ℝ :=
  nd.mean - n * nd.stdDev

/-- Theorem: For a normal distribution with standard deviation 2 and mean 51,
    the value 3 standard deviations below the mean is 45 --/
theorem three_std_dev_below_mean (nd : NormalDistribution) 
    (h1 : nd.stdDev = 2) 
    (h2 : nd.mean = 51) : 
    valueBelow nd 3 = 45 := by
  sorry

end three_std_dev_below_mean_l982_98260


namespace min_value_sum_reciprocals_l982_98237

theorem min_value_sum_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 5) : 
  (9 / x + 4 / y + 25 / z) ≥ 20 := by
  sorry

end min_value_sum_reciprocals_l982_98237


namespace distribute_tickets_count_l982_98258

/-- The number of ways to distribute 4 consecutive numbered tickets among 3 people -/
def distribute_tickets : ℕ :=
  -- Number of ways to split 4 tickets into 3 portions
  let split_ways := Nat.choose 3 2
  -- Number of ways to distribute 3 portions to 3 people
  let distribute_ways := Nat.factorial 3
  -- Total number of distribution methods
  split_ways * distribute_ways

/-- Theorem stating that the number of distribution methods is 18 -/
theorem distribute_tickets_count : distribute_tickets = 18 := by
  sorry

end distribute_tickets_count_l982_98258


namespace sin_cos_value_l982_98278

theorem sin_cos_value (x : ℝ) : 
  let a : ℝ × ℝ := (4 * Real.sin x, 1 - Real.cos x)
  let b : ℝ × ℝ := (1, -2)
  (a.1 * b.1 + a.2 * b.2 = -2) → (Real.sin x * Real.cos x = -2/5) := by
  sorry

end sin_cos_value_l982_98278


namespace greatest_number_jo_thinking_l982_98280

theorem greatest_number_jo_thinking : ∃ n : ℕ,
  n < 100 ∧
  (∃ k : ℕ, n = 5 * k - 2) ∧
  (∃ m : ℕ, n = 9 * m - 4) ∧
  (∀ x : ℕ, x < 100 ∧ (∃ k : ℕ, x = 5 * k - 2) ∧ (∃ m : ℕ, x = 9 * m - 4) → x ≤ n) ∧
  n = 68 :=
by sorry

end greatest_number_jo_thinking_l982_98280


namespace parking_lot_wheels_l982_98235

/-- The number of wheels on a car -/
def car_wheels : ℕ := 4

/-- The number of wheels on a bike -/
def bike_wheels : ℕ := 2

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 14

/-- The number of bikes in the parking lot -/
def num_bikes : ℕ := 5

/-- The total number of wheels in the parking lot -/
def total_wheels : ℕ := num_cars * car_wheels + num_bikes * bike_wheels

theorem parking_lot_wheels : total_wheels = 66 := by
  sorry

end parking_lot_wheels_l982_98235


namespace sum_of_divisors_is_96_l982_98212

-- Define the property of n having exactly 8 divisors, including 1, n, 14, and 21
def has_eight_divisors_with_14_and_21 (n : ℕ) : Prop :=
  (∃ d : Finset ℕ, d.card = 8 ∧ 
    (∀ x, x ∈ d ↔ x ∣ n) ∧
    1 ∈ d ∧ n ∈ d ∧ 14 ∈ d ∧ 21 ∈ d)

-- Theorem stating that if n satisfies the above property, 
-- then the sum of its divisors is 96
theorem sum_of_divisors_is_96 (n : ℕ) 
  (h : has_eight_divisors_with_14_and_21 n) : 
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id = 96 := by
  sorry

end sum_of_divisors_is_96_l982_98212


namespace ada_original_seat_l982_98244

-- Define the number of seats
def num_seats : ℕ := 6

-- Define the movements of friends
def bea_move : ℤ := 3
def ceci_move : ℤ := 1
def dee_move : ℤ := -2
def edie_move : ℤ := -1

-- Define Ada's final position
def ada_final_seat : ℕ := 2

-- Theorem statement
theorem ada_original_seat :
  let net_displacement := bea_move + ceci_move + dee_move + edie_move
  net_displacement = 1 →
  ∃ (ada_original : ℕ), 
    ada_original > 0 ∧ 
    ada_original ≤ num_seats ∧
    ada_original - ada_final_seat = 1 := by
  sorry

end ada_original_seat_l982_98244


namespace set_intersection_and_union_l982_98254

def A (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def B := {x : ℝ | x > 1 ∨ x < -6}

theorem set_intersection_and_union (a : ℝ) :
  (A a ∩ B = ∅ → a ∈ Set.Icc (-6) (-2)) ∧
  (A a ∪ B = B → a ∈ Set.Ioi 1 ∪ Set.Iio (-9)) := by
  sorry

end set_intersection_and_union_l982_98254


namespace carpenters_completion_time_l982_98253

def carpenter1_rate : ℚ := 1 / 5
def carpenter2_rate : ℚ := 1 / 5
def combined_rate : ℚ := carpenter1_rate + carpenter2_rate
def job_completion : ℚ := 1

theorem carpenters_completion_time :
  ∃ (time : ℚ), time * combined_rate = job_completion ∧ time = 5 / 2 := by
  sorry

end carpenters_completion_time_l982_98253


namespace sine_area_theorem_l982_98255

open Set
open MeasureTheory
open Interval

-- Define the sine function
noncomputable def f (x : ℝ) := Real.sin x

-- Define the interval
def I : Set ℝ := Icc (-Real.pi) (2 * Real.pi)

-- State the theorem
theorem sine_area_theorem :
  (∫ x in I, |f x| ∂volume) = 6 := by sorry

end sine_area_theorem_l982_98255


namespace f_def_f_5_eq_0_l982_98228

def f (x : ℝ) : ℝ := sorry

theorem f_def (x : ℝ) : f (2 * x + 1) = x^2 - 2*x := sorry

theorem f_5_eq_0 : f 5 = 0 := by sorry

end f_def_f_5_eq_0_l982_98228


namespace white_square_area_l982_98229

-- Define the cube's properties
def cube_edge : ℝ := 8
def total_green_paint : ℝ := 192

-- Define the theorem
theorem white_square_area :
  let face_area := cube_edge ^ 2
  let total_surface_area := 6 * face_area
  let green_area_per_face := total_green_paint / 6
  let white_area_per_face := face_area - green_area_per_face
  white_area_per_face = 32 := by sorry

end white_square_area_l982_98229


namespace base_h_equation_solution_l982_98272

/-- Represents a number in base h --/
def BaseH (digits : List Nat) (h : Nat) : Nat :=
  digits.foldr (fun d acc => d + h * acc) 0

/-- The theorem statement --/
theorem base_h_equation_solution :
  ∃ (h : Nat), h > 1 ∧ 
    BaseH [8, 3, 7, 4] h + BaseH [6, 9, 2, 5] h = BaseH [1, 5, 3, 0, 9] h ∧
    h = 9 := by
  sorry

end base_h_equation_solution_l982_98272


namespace fraction_difference_l982_98295

theorem fraction_difference : (7 : ℚ) / 12 - (3 : ℚ) / 8 = (5 : ℚ) / 24 := by
  sorry

end fraction_difference_l982_98295


namespace f_even_and_decreasing_l982_98219

def f (x : ℝ) := -x^2 + 1

theorem f_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end f_even_and_decreasing_l982_98219


namespace shaded_area_ratio_is_five_ninths_l982_98289

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a shaded region in the grid -/
structure ShadedRegion :=
  (start_row : ℕ)
  (start_col : ℕ)
  (end_row : ℕ)
  (end_col : ℕ)

/-- Calculates the ratio of shaded area to total area -/
def shaded_area_ratio (g : Grid) (sr : ShadedRegion) : ℚ :=
  sorry

/-- Theorem stating the ratio of shaded area to total area for the given problem -/
theorem shaded_area_ratio_is_five_ninths :
  let g : Grid := ⟨9⟩
  let sr : ShadedRegion := ⟨2, 1, 5, 9⟩
  shaded_area_ratio g sr = 5 / 9 := by
  sorry

end shaded_area_ratio_is_five_ninths_l982_98289


namespace waiter_tip_problem_l982_98211

theorem waiter_tip_problem (total_customers : ℕ) (tip_amount : ℕ) (total_tips : ℕ) 
  (h1 : total_customers = 10)
  (h2 : tip_amount = 3)
  (h3 : total_tips = 15) :
  total_customers - (total_tips / tip_amount) = 5 := by
  sorry

end waiter_tip_problem_l982_98211


namespace current_speed_l982_98264

/-- The speed of the current in a river, given the rowing speed in still water and the time taken to cover a certain distance downstream. -/
theorem current_speed (still_water_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  still_water_speed = 22 →
  downstream_distance = 80 →
  downstream_time = 11.519078473722104 →
  ∃ current_speed : ℝ, 
    (current_speed * 1000 / 3600 + still_water_speed * 1000 / 3600) * downstream_time = downstream_distance ∧ 
    abs (current_speed - 2.9988) < 0.0001 := by
  sorry

end current_speed_l982_98264


namespace highest_probability_prime_l982_98223

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_factor_of_12 (n : ℕ) : Prop := 12 % n = 0

def total_outcomes : ℕ := 36

def prime_outcomes : ℕ := 15
def multiple_of_4_outcomes : ℕ := 9
def perfect_square_outcomes : ℕ := 7
def score_7_outcomes : ℕ := 6
def factor_of_12_outcomes : ℕ := 12

theorem highest_probability_prime :
  prime_outcomes > multiple_of_4_outcomes ∧
  prime_outcomes > perfect_square_outcomes ∧
  prime_outcomes > score_7_outcomes ∧
  prime_outcomes > factor_of_12_outcomes :=
sorry

end highest_probability_prime_l982_98223


namespace blue_chip_percentage_l982_98230

theorem blue_chip_percentage
  (total : ℕ)
  (blue : ℕ)
  (white : ℕ)
  (green : ℕ)
  (h1 : blue = 3)
  (h2 : white = total / 2)
  (h3 : green = 12)
  (h4 : total = blue + white + green) :
  (blue : ℚ) / total * 100 = 10 := by
sorry

end blue_chip_percentage_l982_98230


namespace intersection_of_P_and_Q_l982_98261

def P : Set ℝ := {x | x ≤ 1}
def Q : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

theorem intersection_of_P_and_Q : P ∩ Q = {x | -1 ≤ x ∧ x ≤ 1} := by sorry

end intersection_of_P_and_Q_l982_98261


namespace cafeteria_problem_l982_98203

/-- The cafeteria problem -/
theorem cafeteria_problem 
  (initial_apples : ℕ)
  (apple_cost orange_cost : ℚ)
  (total_earnings : ℚ)
  (apples_left oranges_left : ℕ)
  (h1 : initial_apples = 50)
  (h2 : apple_cost = 8/10)
  (h3 : orange_cost = 1/2)
  (h4 : total_earnings = 49)
  (h5 : apples_left = 10)
  (h6 : oranges_left = 6) :
  ∃ initial_oranges : ℕ, 
    initial_oranges = 40 ∧
    (initial_apples - apples_left) * apple_cost + 
    (initial_oranges - oranges_left) * orange_cost = total_earnings :=
by sorry

end cafeteria_problem_l982_98203


namespace y_value_proof_l982_98270

theorem y_value_proof (x y z a b c : ℝ) 
  (ha : x * y / (x + y) = a)
  (hb : x * z / (x + z) = b)
  (hc : y * z / (y + z) = c)
  (ha_nonzero : a ≠ 0)
  (hb_nonzero : b ≠ 0)
  (hc_nonzero : c ≠ 0) :
  y = 2 * a * b * c / (b * c + a * c - a * b) :=
sorry

end y_value_proof_l982_98270


namespace linear_function_properties_l982_98225

/-- A linear function y = kx + b where k < 0 and b > 0 -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  h₁ : k < 0
  h₂ : b > 0

/-- Properties of the linear function -/
theorem linear_function_properties (f : LinearFunction) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f.k * x₁ + f.b > f.k * x₂ + f.b) ∧ 
  (f.k * (-1) + f.b ≠ -2) ∧
  (f.k * 0 + f.b = f.b) ∧
  (∀ x : ℝ, x > -f.b / f.k → f.k * x + f.b < 0) := by
  sorry

end linear_function_properties_l982_98225


namespace root_implies_sum_l982_98222

theorem root_implies_sum (c d : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2)^3 + c * (Complex.I * Real.sqrt 2 + 2) + d = 0 → 
  c + d = 14 := by
sorry

end root_implies_sum_l982_98222


namespace simplify_polynomial_l982_98221

theorem simplify_polynomial (x : ℝ) : (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*(x - 1) = x^4 - 1 := by
  sorry

end simplify_polynomial_l982_98221


namespace dance_result_l982_98238

/-- Represents a sequence of dance steps, where positive numbers are forward steps
    and negative numbers are backward steps. -/
def dance_sequence : List Int := [-5, 10, -2, 2 * 2]

/-- Calculates the final position after performing a sequence of dance steps. -/
def final_position (steps : List Int) : Int :=
  steps.sum

/-- Proves that the given dance sequence results in a final position 7 steps forward. -/
theorem dance_result :
  final_position dance_sequence = 7 := by
  sorry

end dance_result_l982_98238


namespace unique_solution_exponential_equation_l982_98208

theorem unique_solution_exponential_equation :
  ∃! (n : ℕ+), Real.exp (1 / n.val) + Real.exp (-1 / n.val) = Real.sqrt n.val :=
by
  sorry

end unique_solution_exponential_equation_l982_98208


namespace julia_miles_driven_l982_98273

theorem julia_miles_driven (darius_miles julia_miles total_miles : ℕ) 
  (h1 : darius_miles = 679)
  (h2 : total_miles = 1677)
  (h3 : total_miles = darius_miles + julia_miles) : 
  julia_miles = 998 := by
  sorry

end julia_miles_driven_l982_98273


namespace hot_drink_sales_at_2_degrees_l982_98269

/-- Represents the linear regression equation for hot drink sales -/
def hot_drink_sales (x : ℝ) : ℝ := -2.35 * x + 147.77

/-- Theorem stating that when the temperature is 2℃, approximately 143 hot drinks are sold -/
theorem hot_drink_sales_at_2_degrees :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |hot_drink_sales 2 - 143| < ε :=
sorry

end hot_drink_sales_at_2_degrees_l982_98269


namespace magazine_budget_cut_percentage_l982_98247

def original_budget : ℚ := 940
def new_budget : ℚ := 752

theorem magazine_budget_cut_percentage : 
  (original_budget - new_budget) / original_budget * 100 = 20 := by
  sorry

end magazine_budget_cut_percentage_l982_98247


namespace complex_exp_conversion_l982_98275

theorem complex_exp_conversion (z : ℂ) :
  z = Real.sqrt 2 * Complex.exp (13 * Real.pi * Complex.I / 4) →
  z = 1 + Complex.I := by
  sorry

end complex_exp_conversion_l982_98275

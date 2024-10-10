import Mathlib

namespace min_value_of_a_l1813_181313

theorem min_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x / (x^2 + 3*x + 1) ≤ a) → a ≥ 1/5 := by
  sorry

end min_value_of_a_l1813_181313


namespace unique_intersection_l1813_181352

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the property of having domain [-1, 5]
def HasDomain (f : RealFunction) : Prop :=
  ∀ x, x ∈ Set.Icc (-1) 5 → ∃ y, f x = y

-- Define the intersection of f with the line x=1
def Intersection (f : RealFunction) : Set ℝ :=
  {y : ℝ | f 1 = y}

-- Theorem statement
theorem unique_intersection
  (f : RealFunction) (h : HasDomain f) :
  ∃! y, y ∈ Intersection f :=
sorry

end unique_intersection_l1813_181352


namespace xy_equals_five_l1813_181337

theorem xy_equals_five (x y : ℝ) (h : x * (x + 2*y) = x^2 + 10) : x * y = 5 := by
  sorry

end xy_equals_five_l1813_181337


namespace negation_of_square_positive_l1813_181354

theorem negation_of_square_positive :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, ¬(x^2 > 0)) :=
by sorry

end negation_of_square_positive_l1813_181354


namespace roots_sum_of_squares_l1813_181303

theorem roots_sum_of_squares (m n a b : ℝ) : 
  (∀ x, x^2 - m*x + n = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = m^2 - 2*n := by
  sorry

end roots_sum_of_squares_l1813_181303


namespace absolute_value_probability_l1813_181345

theorem absolute_value_probability (x : ℝ) : ℝ := by
  have h : ∀ x : ℝ, |x| ≥ 0 := by sorry
  have event : Set ℝ := {x | |x| < 0}
  have prob_event : ℝ := 0
  sorry

end absolute_value_probability_l1813_181345


namespace vector_equality_exists_l1813_181367

theorem vector_equality_exists (a b : ℝ × ℝ) :
  let a : ℝ × ℝ := (1, Real.sqrt 3)
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧
    let b : ℝ × ℝ := (Real.cos θ, Real.sin θ)
    ‖a + b‖ = ‖a - b‖ :=
by sorry

end vector_equality_exists_l1813_181367


namespace stockholm_uppsala_distance_l1813_181305

/-- Represents the distance between two cities on a map and in reality. -/
structure MapDistance where
  /-- Distance on the map in centimeters -/
  map_distance : ℝ
  /-- Map scale: 1 cm on map represents this many km in reality -/
  scale : ℝ

/-- Calculates the real-world distance in meters given a MapDistance -/
def real_distance (d : MapDistance) : ℝ :=
  d.map_distance * d.scale * 1000

/-- The distance between Stockholm and Uppsala -/
def stockholm_uppsala : MapDistance :=
  { map_distance := 55
  , scale := 30 }

/-- Theorem stating that the distance between Stockholm and Uppsala is 1650000 meters -/
theorem stockholm_uppsala_distance :
  real_distance stockholm_uppsala = 1650000 := by
  sorry

end stockholm_uppsala_distance_l1813_181305


namespace special_sequence_has_large_number_l1813_181370

/-- A sequence of natural numbers with the given properties -/
def SpecialSequence (seq : Fin 20 → ℕ) : Prop :=
  (∀ i, seq i ≠ seq (i + 1)) ∧  -- distinct numbers
  (∀ i < 19, ∃ k : ℕ, seq i * seq (i + 1) = k * k) ∧  -- product is perfect square
  seq 0 = 42  -- first number is 42

theorem special_sequence_has_large_number (seq : Fin 20 → ℕ) 
  (h : SpecialSequence seq) : 
  ∃ i, seq i > 16000 := by
sorry

end special_sequence_has_large_number_l1813_181370


namespace quadratic_equation_root_l1813_181374

theorem quadratic_equation_root (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ x = 1) → k = 1 := by
  sorry

end quadratic_equation_root_l1813_181374


namespace group_abelian_l1813_181351

variable {G : Type*} [Group G]

theorem group_abelian (h : ∀ x : G, x * x = 1) : ∀ a b : G, a * b = b * a := by
  sorry

end group_abelian_l1813_181351


namespace park_animals_ratio_l1813_181348

theorem park_animals_ratio (lions leopards elephants : ℕ) : 
  lions = 200 →
  elephants = (lions + leopards) / 2 →
  lions + leopards + elephants = 450 →
  lions = 2 * leopards :=
by
  sorry

end park_animals_ratio_l1813_181348


namespace original_number_l1813_181378

theorem original_number (t : ℝ) : 
  t * (1 + 0.125) - t * (1 - 0.25) = 30 → t = 80 := by
  sorry

end original_number_l1813_181378


namespace remainder_theorem_l1813_181353

-- Define the polynomial p(x)
def p (A B C : ℝ) (x : ℝ) : ℝ := A * x^5 + B * x^3 + C * x + 4

-- Theorem statement
theorem remainder_theorem (A B C : ℝ) :
  (p A B C 3 = 11) → (p A B C (-3) = -3) := by
  sorry

end remainder_theorem_l1813_181353


namespace largest_α_is_173_l1813_181397

/-- A triangle with angles satisfying specific conditions -/
structure SpecialTriangle where
  α : ℕ
  β : ℕ
  γ : ℕ
  angle_sum : α + β + γ = 180
  angle_order : α > β ∧ β > γ
  α_obtuse : α > 90
  α_prime : Nat.Prime α
  β_prime : Nat.Prime β

/-- The largest possible value of α in a SpecialTriangle is 173 -/
theorem largest_α_is_173 : ∀ t : SpecialTriangle, t.α ≤ 173 ∧ ∃ t' : SpecialTriangle, t'.α = 173 :=
  sorry

end largest_α_is_173_l1813_181397


namespace ellipse_chord_properties_l1813_181304

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse mx² + ny² = 1 -/
structure Ellipse where
  m : ℝ
  n : ℝ
  h_positive : m > 0 ∧ n > 0
  h_distinct : m ≠ n

/-- Theorem about properties of chords in an ellipse -/
theorem ellipse_chord_properties (e : Ellipse) (a b c d : Point) (e_mid f_mid : Point) : 
  -- AB is a chord with slope 1
  (b.y - a.y) / (b.x - a.x) = 1 →
  -- CD is perpendicular to AB
  (d.y - c.y) / (d.x - c.x) = -1 →
  -- E is midpoint of AB
  e_mid.x = (a.x + b.x) / 2 ∧ e_mid.y = (a.y + b.y) / 2 →
  -- F is midpoint of CD
  f_mid.x = (c.x + d.x) / 2 ∧ f_mid.y = (c.y + d.y) / 2 →
  -- A, B, C, D are on the ellipse
  e.m * a.x^2 + e.n * a.y^2 = 1 ∧
  e.m * b.x^2 + e.n * b.y^2 = 1 ∧
  e.m * c.x^2 + e.n * c.y^2 = 1 ∧
  e.m * d.x^2 + e.n * d.y^2 = 1 →
  -- Conclusion 1: |CD|² - |AB|² = 4|EF|²
  ((c.x - d.x)^2 + (c.y - d.y)^2) - ((a.x - b.x)^2 + (a.y - b.y)^2) = 
    4 * ((e_mid.x - f_mid.x)^2 + (e_mid.y - f_mid.y)^2) ∧
  -- Conclusion 2: A, B, C, D are concyclic
  ∃ (center : Point) (r : ℝ),
    (a.x - center.x)^2 + (a.y - center.y)^2 = r^2 ∧
    (b.x - center.x)^2 + (b.y - center.y)^2 = r^2 ∧
    (c.x - center.x)^2 + (c.y - center.y)^2 = r^2 ∧
    (d.x - center.x)^2 + (d.y - center.y)^2 = r^2 :=
by
  sorry

end ellipse_chord_properties_l1813_181304


namespace fraction_of_married_women_l1813_181390

/-- Given a company with employees, prove that 3/4 of women are married under specific conditions -/
theorem fraction_of_married_women (total : ℕ) (h_total_pos : total > 0) : 
  let women := (64 : ℚ) / 100 * total
  let married := (60 : ℚ) / 100 * total
  let men := total - women
  let single_men := (2 : ℚ) / 3 * men
  let married_men := men - single_men
  let married_women := married - married_men
  married_women / women = (3 : ℚ) / 4 := by
  sorry


end fraction_of_married_women_l1813_181390


namespace rectangle_area_l1813_181389

theorem rectangle_area (length width : ℚ) (h1 : length = 1/3) (h2 : width = 1/5) :
  length * width = 1/15 := by
  sorry

end rectangle_area_l1813_181389


namespace second_number_proof_l1813_181327

theorem second_number_proof (x : ℕ) : 
  (1255 % 29 = 8) → (x % 29 = 11) → x = 1287 := by
  sorry

end second_number_proof_l1813_181327


namespace inverse_of_ln_l1813_181355

theorem inverse_of_ln (x : ℝ) : 
  (fun y ↦ Real.exp y) ∘ (fun x ↦ Real.log x) = id ∧ 
  (fun x ↦ Real.log x) ∘ (fun y ↦ Real.exp y) = id :=
sorry

end inverse_of_ln_l1813_181355


namespace collaborative_work_theorem_work_result_l1813_181311

/-- Represents the time taken to complete a task -/
structure TaskTime where
  days : ℝ
  hours_per_day : ℝ := 24
  total_hours : ℝ := days * hours_per_day

/-- Represents a worker's productivity -/
structure Worker where
  name : String
  task_time : TaskTime

/-- Represents a collaborative work scenario -/
structure CollaborativeWork where
  john : Worker
  jane : Worker
  total_time : TaskTime
  jane_indisposed_time : ℝ

/-- The main theorem to prove -/
theorem collaborative_work_theorem (work : CollaborativeWork) : 
  work.john.task_time.days = 18 →
  work.jane.task_time.days = 12 →
  work.total_time.days = 10.8 →
  work.jane_indisposed_time = 6 := by
  sorry

/-- An instance of the collaborative work scenario -/
def work_scenario : CollaborativeWork := {
  john := { name := "John", task_time := { days := 18 } }
  jane := { name := "Jane", task_time := { days := 12 } }
  total_time := { days := 10.8 }
  jane_indisposed_time := 6
}

/-- The main result -/
theorem work_result : work_scenario.jane_indisposed_time = 6 := by
  apply collaborative_work_theorem work_scenario
  · rfl
  · rfl
  · rfl

end collaborative_work_theorem_work_result_l1813_181311


namespace triangle_inequality_ortho_segments_inequality_not_always_true_l1813_181346

/-- A triangle with sides a ≥ b ≥ c and corresponding altitudes m_a ≤ m_b ≤ m_c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  h_sides : a ≥ b ∧ b ≥ c
  h_altitudes : m_a ≤ m_b ∧ m_b ≤ m_c

/-- Lengths of segments from vertex to orthocenter along corresponding altitudes -/
structure OrthoSegments where
  m_a_star : ℝ
  m_b_star : ℝ
  m_c_star : ℝ

/-- Theorem stating the inequality for sides and altitudes -/
theorem triangle_inequality (t : Triangle) : t.a + t.m_a ≥ t.b + t.m_b ∧ t.b + t.m_b ≥ t.c + t.m_c :=
  sorry

/-- Statement that the inequality for orthocenter segments is not always true -/
theorem ortho_segments_inequality_not_always_true : 
  ¬ ∀ (t : Triangle) (o : OrthoSegments), t.a + o.m_a_star ≥ t.b + o.m_b_star ∧ t.b + o.m_b_star ≥ t.c + o.m_c_star :=
  sorry

end triangle_inequality_ortho_segments_inequality_not_always_true_l1813_181346


namespace max_value_product_ratios_l1813_181363

/-- Line l in Cartesian coordinates -/
def line_l (y : ℝ) : Prop := y = 8

/-- Circle C in parametric form -/
def circle_C (x y φ : ℝ) : Prop := x = 2 + 2 * Real.cos φ ∧ y = 2 * Real.sin φ

/-- Ray OM in polar coordinates -/
def ray_OM (θ α : ℝ) : Prop := θ = α ∧ 0 < α ∧ α < Real.pi / 2

/-- Ray ON in polar coordinates -/
def ray_ON (θ α : ℝ) : Prop := θ = α - Real.pi / 2

/-- Theorem stating the maximum value of the product of ratios -/
theorem max_value_product_ratios (α : ℝ) 
  (h_ray_OM : ray_OM α α) 
  (h_ray_ON : ray_ON (α - Real.pi / 2) α) : 
  ∃ (OP OM OQ ON : ℝ), 
    (OP / OM) * (OQ / ON) ≤ 1 / 16 ∧ 
    ∃ (α_max : ℝ), (OP / OM) * (OQ / ON) = 1 / 16 := by
  sorry

end max_value_product_ratios_l1813_181363


namespace cos_minus_sin_identity_l1813_181300

theorem cos_minus_sin_identity (θ : Real) (a b : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (h_sin : Real.sin (2 * θ) = a)
  (h_cos : Real.cos (2 * θ) = b) :
  Real.cos θ - Real.sin θ = Real.sqrt (1 - a) :=
sorry

end cos_minus_sin_identity_l1813_181300


namespace t_range_l1813_181386

/-- The function f(x) = |xe^x| -/
noncomputable def f (x : ℝ) : ℝ := |x * Real.exp x|

/-- The function g(x) = [f(x)]^2 - tf(x) -/
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := (f x)^2 - t * (f x)

/-- The theorem stating the range of t -/
theorem t_range (t : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    g t x₁ = -2 ∧ g t x₂ = -2 ∧ g t x₃ = -2 ∧ g t x₄ = -2) →
  t > Real.exp (-1) + 2 * Real.exp 1 :=
by sorry

end t_range_l1813_181386


namespace water_added_to_container_l1813_181356

theorem water_added_to_container (capacity : ℝ) (initial_fullness : ℝ) (final_fullness : ℝ) : 
  capacity = 120 →
  initial_fullness = 0.3 →
  final_fullness = 0.75 →
  (final_fullness - initial_fullness) * capacity = 54 := by
sorry

end water_added_to_container_l1813_181356


namespace pants_gross_profit_l1813_181328

/-- Calculates the gross profit for a store selling pants -/
theorem pants_gross_profit (purchase_price : ℝ) (markup_percent : ℝ) (price_decrease : ℝ) :
  purchase_price = 210 ∧ 
  markup_percent = 0.25 ∧ 
  price_decrease = 0.20 →
  let original_price := purchase_price / (1 - markup_percent)
  let new_price := original_price * (1 - price_decrease)
  new_price - purchase_price = 14 := by
  sorry

end pants_gross_profit_l1813_181328


namespace recurrence_initial_values_l1813_181331

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (x : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, x (n + 1) = (x n ^ 2 + 10) / 7

/-- The property of being bounded above -/
def BoundedAbove (x : ℤ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ n : ℤ, x n ≤ M

/-- The set of possible initial values for bounded sequences satisfying the recurrence -/
def PossibleInitialValues : Set ℝ :=
  {x₀ : ℝ | ∃ x : ℤ → ℝ, RecurrenceSequence x ∧ BoundedAbove x ∧ x 0 = x₀}

theorem recurrence_initial_values :
    PossibleInitialValues = Set.Icc 2 5 := by
  sorry

end recurrence_initial_values_l1813_181331


namespace f_monotonically_decreasing_l1813_181387

-- Define the function f(x) = e^x / x
noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

-- Define the property of being monotonically decreasing on an interval
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f y < f x

-- State the theorem
theorem f_monotonically_decreasing :
  (∀ x y, x < y → y < 0 → f y < f x) ∧
  (∀ x y, 0 < x → x < y → y ≤ 1 → f y < f x) :=
sorry

end f_monotonically_decreasing_l1813_181387


namespace function_properties_l1813_181308

noncomputable def f (x : ℝ) := Real.cos (2 * x + 2 * Real.pi / 3) + 2 * (Real.cos x) ^ 2

theorem function_properties :
  (∀ x, f x ≤ 2) ∧
  (∀ k : ℤ, f (k * Real.pi - Real.pi / 6) = 2) ∧
  (∀ A B C a b c : ℝ,
    0 < A ∧ A < Real.pi →
    0 < B ∧ B < Real.pi →
    0 < C ∧ C < Real.pi →
    A + B + C = Real.pi →
    a > 0 ∧ b > 0 ∧ c > 0 →
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
    f A = 3 / 2 →
    b + c = 2 →
    a ≥ Real.sqrt 3) := by
  sorry

#check function_properties

end function_properties_l1813_181308


namespace playlist_composition_l1813_181320

theorem playlist_composition (initial_hip_hop_ratio : Real) 
  (country_percentage : Real) (hip_hop_percentage : Real) : 
  initial_hip_hop_ratio = 0.65 →
  country_percentage = 0.4 →
  hip_hop_percentage = (1 - country_percentage) * initial_hip_hop_ratio →
  hip_hop_percentage = 0.39 := by
  sorry

end playlist_composition_l1813_181320


namespace jensen_family_trip_l1813_181344

/-- Calculates the miles driven on city streets given the total distance on highways,
    car efficiency on highways and city streets, and total gas used. -/
theorem jensen_family_trip (highway_miles : ℝ) (highway_efficiency : ℝ) 
  (city_efficiency : ℝ) (total_gas : ℝ) (city_miles : ℝ) : 
  highway_miles = 210 →
  highway_efficiency = 35 →
  city_efficiency = 18 →
  total_gas = 9 →
  city_miles = (total_gas - highway_miles / highway_efficiency) * city_efficiency →
  city_miles = 54 := by sorry

end jensen_family_trip_l1813_181344


namespace complex_coordinate_l1813_181333

/-- Given zi = 2-i, prove that z = -1 - 2i -/
theorem complex_coordinate (z : ℂ) : z * Complex.I = 2 - Complex.I → z = -1 - 2 * Complex.I := by
  sorry

end complex_coordinate_l1813_181333


namespace geometric_sequence_sum_l1813_181383

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16) →
  a 3 + a 6 = 4 := by
sorry

end geometric_sequence_sum_l1813_181383


namespace necklace_cuts_l1813_181339

theorem necklace_cuts (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 8) :
  Nat.choose n k = 145422675 :=
by sorry

end necklace_cuts_l1813_181339


namespace zero_multiple_of_all_integers_l1813_181358

theorem zero_multiple_of_all_integers : ∀ (n : ℤ), ∃ (k : ℤ), 0 = k * n := by
  sorry

end zero_multiple_of_all_integers_l1813_181358


namespace binomial_22_12_l1813_181310

theorem binomial_22_12 (h1 : Nat.choose 20 10 = 184756)
                        (h2 : Nat.choose 20 11 = 167960)
                        (h3 : Nat.choose 20 12 = 125970) :
  Nat.choose 22 12 = 646646 := by
  sorry

end binomial_22_12_l1813_181310


namespace base_dimensions_of_divided_volume_l1813_181385

/-- Given a volume of 120 cubic cubits divided into 10 parts, each with a height of 1 cubit,
    and a rectangular base with sides in the ratio 1:3/4, prove that the dimensions of the base
    are 4 cubits and 3 cubits. -/
theorem base_dimensions_of_divided_volume (total_volume : ℝ) (num_parts : ℕ) 
    (part_height : ℝ) (base_ratio : ℝ) :
  total_volume = 120 →
  num_parts = 10 →
  part_height = 1 →
  base_ratio = 3/4 →
  ∃ (a b : ℝ), a = 4 ∧ b = 3 ∧
    a * b * part_height * num_parts = total_volume ∧
    b / a = base_ratio :=
by sorry

end base_dimensions_of_divided_volume_l1813_181385


namespace increasing_cubic_function_parameter_range_l1813_181371

theorem increasing_cubic_function_parameter_range 
  (f : ℝ → ℝ) (a : ℝ) (h : ∀ x ∈ Set.Ioo (-1) 1, StrictMono f) : a ≥ 3 := by
  sorry

end increasing_cubic_function_parameter_range_l1813_181371


namespace ratio_fraction_equality_l1813_181381

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end ratio_fraction_equality_l1813_181381


namespace vincent_book_expenditure_l1813_181321

/-- The total amount Vincent spent on books -/
def total_spent (animal_books train_books space_books book_price : ℕ) : ℕ :=
  (animal_books + train_books + space_books) * book_price

/-- Theorem stating that Vincent spent $224 on books -/
theorem vincent_book_expenditure :
  total_spent 10 3 1 16 = 224 := by
  sorry

end vincent_book_expenditure_l1813_181321


namespace smallest_solution_floor_equation_l1813_181396

theorem smallest_solution_floor_equation :
  (∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 12) ∧
  (∀ (y : ℝ), y > 0 ∧ ⌊y^2⌋ - y * ⌊y⌋ = 12 → y ≥ 169/13) :=
by sorry

end smallest_solution_floor_equation_l1813_181396


namespace cube_root_of_a_plus_b_plus_one_l1813_181398

theorem cube_root_of_a_plus_b_plus_one (a b : ℝ) 
  (h1 : (2 * a - 1) = 9)
  (h2 : (3 * a + b - 1) = 16) : 
  (a + b + 1)^(1/3 : ℝ) = 2 := by
  sorry

end cube_root_of_a_plus_b_plus_one_l1813_181398


namespace blueberry_zucchini_trade_l1813_181336

/-- The number of containers of blueberries yielded by one bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries that can be traded for zucchinis -/
def containers_traded : ℕ := 6

/-- The number of zucchinis received in trade for containers_traded -/
def zucchinis_received : ℕ := 3

/-- The total number of zucchinis Natalie wants to obtain -/
def target_zucchinis : ℕ := 60

/-- The number of bushes needed to obtain the target number of zucchinis -/
def bushes_needed : ℕ := 12

theorem blueberry_zucchini_trade :
  bushes_needed * containers_per_bush * zucchinis_received = 
  target_zucchinis * containers_traded := by
  sorry

end blueberry_zucchini_trade_l1813_181336


namespace right_triangle_max_ratio_l1813_181324

theorem right_triangle_max_ratio :
  ∀ (x y z : ℝ), 
    x > 0 → y > 0 → z > 0 →
    x^2 + y^2 = z^2 →
    (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a^2 + b^2 = c^2 → (a + 2*b) / c ≤ (x + 2*y) / z) →
    (x + 2*y) / z = 3 * Real.sqrt 2 / 2 :=
by sorry

end right_triangle_max_ratio_l1813_181324


namespace sum_independence_and_value_l1813_181307

theorem sum_independence_and_value (a : ℝ) (h : a ≥ -3/4) :
  let s := (((a + 1) / 2 + (a + 3) / 6 * Real.sqrt ((4 * a + 3) / 3)) ^ (1/3 : ℝ) : ℝ)
  let t := (((a + 1) / 2 - (a + 3) / 6 * Real.sqrt ((4 * a + 3) / 3)) ^ (1/3 : ℝ) : ℝ)
  s + t = 1 := by sorry

end sum_independence_and_value_l1813_181307


namespace consecutive_heads_probability_l1813_181319

/-- The number of coin flips -/
def n : ℕ := 12

/-- The number of desired heads -/
def k : ℕ := 9

/-- The probability of getting heads on a single flip of a fair coin -/
def p : ℚ := 1/2

/-- The number of ways to arrange k consecutive heads in n flips -/
def consecutive_arrangements : ℕ := n - k + 1

theorem consecutive_heads_probability :
  (consecutive_arrangements : ℚ) * p^k * (1-p)^(n-k) = 1/1024 := by
  sorry

end consecutive_heads_probability_l1813_181319


namespace steven_extra_seeds_l1813_181349

/-- Represents the number of seeds in different fruits -/
structure FruitSeeds where
  apple : Nat
  pear : Nat
  grape : Nat
  orange : Nat
  watermelon : Nat

/-- Represents the number of each fruit Steven has -/
structure StevenFruits where
  apples : Nat
  pears : Nat
  grapes : Nat
  oranges : Nat
  watermelons : Nat

def required_seeds : Nat := 420

def average_seeds : FruitSeeds := {
  apple := 6,
  pear := 2,
  grape := 3,
  orange := 10,
  watermelon := 300
}

def steven_fruits : StevenFruits := {
  apples := 2,
  pears := 3,
  grapes := 5,
  oranges := 1,
  watermelons := 2
}

/-- Calculates the total number of seeds Steven has -/
def total_seeds (avg : FruitSeeds) (fruits : StevenFruits) : Nat :=
  avg.apple * fruits.apples +
  avg.pear * fruits.pears +
  avg.grape * fruits.grapes +
  avg.orange * fruits.oranges +
  avg.watermelon * fruits.watermelons

/-- Theorem stating that Steven has 223 more seeds than required -/
theorem steven_extra_seeds :
  total_seeds average_seeds steven_fruits - required_seeds = 223 := by
  sorry

end steven_extra_seeds_l1813_181349


namespace jared_popcorn_order_l1813_181301

/-- Calculate the number of servings of popcorn needed for a group -/
def popcorn_servings (pieces_per_serving : ℕ) (jared_pieces : ℕ) (friend_count : ℕ) (friend_pieces : ℕ) : ℕ :=
  (jared_pieces + friend_count * friend_pieces) / pieces_per_serving

theorem jared_popcorn_order :
  popcorn_servings 30 90 3 60 = 9 := by
  sorry

end jared_popcorn_order_l1813_181301


namespace smoothie_combinations_l1813_181323

theorem smoothie_combinations (n_smoothies : ℕ) (n_supplements : ℕ) : 
  n_smoothies = 7 → n_supplements = 8 → n_smoothies * (n_supplements.choose 3) = 392 := by
  sorry

end smoothie_combinations_l1813_181323


namespace max_pairs_after_loss_l1813_181335

theorem max_pairs_after_loss (initial_pairs : ℕ) (lost_shoes : ℕ) (max_pairs : ℕ) : 
  initial_pairs = 24 →
  lost_shoes = 9 →
  max_pairs = initial_pairs - (lost_shoes / 2) →
  max_pairs = 20 :=
by sorry

end max_pairs_after_loss_l1813_181335


namespace circle_center_radius_sum_l1813_181302

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*y - 16 = -y^2 + 24*x + 16

-- Define the center and radius of the circle
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ a b r : ℝ, is_center_radius a b r ∧ a + b + r = 10 + 2 * Real.sqrt 41 :=
sorry

end circle_center_radius_sum_l1813_181302


namespace max_total_points_l1813_181317

/-- Represents the types of buckets in the ring toss game -/
inductive Bucket
| Red
| Green
| Blue

/-- Represents the game state -/
structure GameState where
  money : ℕ
  points : ℕ
  rings_per_play : ℕ
  red_points : ℕ
  green_points : ℕ
  blue_points : ℕ
  blue_success_rate : ℚ
  red_buckets_hit : ℕ
  green_buckets_hit : ℕ
  blue_buckets_hit : ℕ

/-- Calculates the maximum points achievable in one play -/
def max_points_per_play (gs : GameState) : ℕ :=
  gs.rings_per_play * max gs.red_points (max gs.green_points gs.blue_points)

/-- Calculates the total points from already hit buckets -/
def current_points (gs : GameState) : ℕ :=
  gs.red_buckets_hit * gs.red_points +
  gs.green_buckets_hit * gs.green_points +
  gs.blue_buckets_hit * gs.blue_points

/-- Theorem: The maximum total points Tiffany can achieve in three games is 43 -/
theorem max_total_points (gs : GameState)
  (h1 : gs.money = 3)
  (h2 : gs.rings_per_play = 5)
  (h3 : gs.red_points = 2)
  (h4 : gs.green_points = 3)
  (h5 : gs.blue_points = 5)
  (h6 : gs.blue_success_rate = 1/10)
  (h7 : gs.red_buckets_hit = 4)
  (h8 : gs.green_buckets_hit = 5)
  (h9 : gs.blue_buckets_hit = 1) :
  current_points gs + max_points_per_play gs = 43 :=
by sorry

end max_total_points_l1813_181317


namespace total_salary_formula_l1813_181350

/-- Represents the total annual salary (in ten thousand yuan) paid by the enterprise in the nth year -/
def total_salary (n : ℕ) : ℝ :=
  (3 * n + 5) * (1.2 : ℝ)^n + 2.4

/-- The initial number of workers -/
def initial_workers : ℕ := 8

/-- The initial annual salary per worker (in yuan) -/
def initial_salary : ℝ := 10000

/-- The annual salary increase rate -/
def salary_increase_rate : ℝ := 0.2

/-- The number of new workers added each year -/
def new_workers_per_year : ℕ := 3

/-- The first-year salary of new workers (in yuan) -/
def new_worker_salary : ℝ := 8000

theorem total_salary_formula (n : ℕ) :
  total_salary n = (3 * n + initial_workers - 3) * (1 + salary_increase_rate)^n +
    (new_workers_per_year * new_worker_salary / 10000) := by
  sorry

end total_salary_formula_l1813_181350


namespace empty_bucket_weight_l1813_181394

theorem empty_bucket_weight (full_weight : ℝ) (partial_weight : ℝ) : 
  full_weight = 3.4 →
  partial_weight = 2.98 →
  ∃ (empty_weight : ℝ),
    empty_weight = 1.3 ∧
    full_weight = empty_weight + (3.4 - empty_weight) ∧
    partial_weight = empty_weight + 4/5 * (3.4 - empty_weight) := by
  sorry

end empty_bucket_weight_l1813_181394


namespace tom_uncommon_cards_l1813_181373

/-- Represents the deck composition and cost in Tom's trading card game. -/
structure DeckInfo where
  rare_count : ℕ
  common_count : ℕ
  rare_cost : ℚ
  uncommon_cost : ℚ
  common_cost : ℚ
  total_cost : ℚ

/-- Calculates the number of uncommon cards in the deck. -/
def uncommon_count (deck : DeckInfo) : ℕ :=
  let rare_total := deck.rare_count * deck.rare_cost
  let common_total := deck.common_count * deck.common_cost
  let uncommon_total := deck.total_cost - rare_total - common_total
  (uncommon_total / deck.uncommon_cost).num.toNat

/-- Theorem stating that Tom's deck contains 11 uncommon cards. -/
theorem tom_uncommon_cards : 
  let deck : DeckInfo := {
    rare_count := 19,
    common_count := 30,
    rare_cost := 1,
    uncommon_cost := 1/2,
    common_cost := 1/4,
    total_cost := 32
  }
  uncommon_count deck = 11 := by sorry

end tom_uncommon_cards_l1813_181373


namespace eg_fh_ratio_l1813_181334

/-- Given points E, F, G, and H on a line in that order, prove that EG:FH = 10:17 -/
theorem eg_fh_ratio (E F G H : ℝ) (h_order : E ≤ F ∧ F ≤ G ∧ G ≤ H) 
  (h_ef : F - E = 3) (h_fg : G - F = 7) (h_eh : H - E = 20) :
  (G - E) / (H - F) = 10 / 17 := by
  sorry

end eg_fh_ratio_l1813_181334


namespace cuboid_height_l1813_181392

/-- The surface area of a rectangular cuboid given its length, width, and height. -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: For a rectangular cuboid with surface area 442 cm², width 7 cm, and length 8 cm, the height is 11 cm. -/
theorem cuboid_height (A l w h : ℝ) 
  (h_area : A = 442)
  (h_width : w = 7)
  (h_length : l = 8)
  (h_surface : surface_area l w h = A) : h = 11 := by
  sorry

end cuboid_height_l1813_181392


namespace power_sum_theorem_l1813_181365

theorem power_sum_theorem (a b x y : ℝ) 
  (h1 : a*x + b*y = 5)
  (h2 : a*x^2 + b*y^2 = 11)
  (h3 : a*x^3 + b*y^3 = 25)
  (h4 : a*x^4 + b*y^4 = 58) :
  a*x^5 + b*y^5 = 136.25 := by
  sorry

end power_sum_theorem_l1813_181365


namespace missing_number_proof_l1813_181391

def set1_sum (x y : ℝ) : ℝ := x + 50 + 78 + 104 + y
def set2_sum (x : ℝ) : ℝ := 48 + 62 + 98 + 124 + x

theorem missing_number_proof (x y : ℝ) :
  set1_sum x y / 5 = 62 ∧ set2_sum x / 5 = 76.4 → y = 28 := by
  sorry

end missing_number_proof_l1813_181391


namespace coefficient_x_cubed_in_binomial_expansion_l1813_181384

theorem coefficient_x_cubed_in_binomial_expansion : 
  let n : ℕ := 5
  let a : ℝ := 1
  let b : ℝ := 2
  let r : ℕ := 3
  let coeff : ℝ := (n.choose r) * a^(n-r) * b^r
  coeff = 80 := by sorry

end coefficient_x_cubed_in_binomial_expansion_l1813_181384


namespace exists_number_with_specific_digit_sum_l1813_181362

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with specific digit sum properties -/
theorem exists_number_with_specific_digit_sum : 
  ∃ m : ℕ, digit_sum m = 1990 ∧ digit_sum (m^2) = 1990^2 := by sorry

end exists_number_with_specific_digit_sum_l1813_181362


namespace base12_addition_theorem_l1813_181330

-- Define a custom type for base-12 digits
inductive Base12Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

-- Define a type for base-12 numbers
def Base12Number := List Base12Digit

-- Define the two numbers we're adding
def num1 : Base12Number := [Base12Digit.D5, Base12Digit.D2, Base12Digit.D8]
def num2 : Base12Number := [Base12Digit.D2, Base12Digit.D7, Base12Digit.D3]

-- Define the expected result
def expected_result : Base12Number := [Base12Digit.D7, Base12Digit.D9, Base12Digit.B]

-- Function to add two base-12 numbers
def add_base12 (a b : Base12Number) : Base12Number :=
  sorry

theorem base12_addition_theorem :
  add_base12 num1 num2 = expected_result :=
sorry

end base12_addition_theorem_l1813_181330


namespace series_sum_l1813_181347

/-- The general term of the series -/
def a (n : ℕ) : ℚ := (3 * n^2 + 2 * n + 1) / (n * (n + 1) * (n + 2) * (n + 3))

/-- The series sum -/
noncomputable def S : ℚ := ∑' n, a n

/-- Theorem: The sum of the series is 7/6 -/
theorem series_sum : S = 7/6 := by
  sorry

end series_sum_l1813_181347


namespace hour_division_theorem_l1813_181372

/-- The number of seconds in an hour -/
def seconds_in_hour : ℕ := 3600

/-- The number of ways to divide an hour into periods -/
def num_divisions : ℕ := 44

/-- Theorem: The number of ordered pairs of positive integers (n, m) 
    such that n * m = 3600 is equal to 44 -/
theorem hour_division_theorem : 
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = seconds_in_hour) 
    (Finset.product (Finset.range (seconds_in_hour + 1)) (Finset.range (seconds_in_hour + 1)))).card 
  = num_divisions := by
  sorry


end hour_division_theorem_l1813_181372


namespace right_triangle_hypotenuse_l1813_181341

theorem right_triangle_hypotenuse (base height hypotenuse : ℝ) : 
  base = 8 →
  (1/2) * base * height = 24 →
  base^2 + height^2 = hypotenuse^2 →
  hypotenuse = 10 := by
sorry

end right_triangle_hypotenuse_l1813_181341


namespace todds_contribution_ratio_l1813_181375

theorem todds_contribution_ratio (total_cost : ℕ) (boss_contribution : ℕ) 
  (num_employees : ℕ) (employee_contribution : ℕ) : 
  total_cost = 100 →
  boss_contribution = 15 →
  num_employees = 5 →
  employee_contribution = 11 →
  (total_cost - (boss_contribution + num_employees * employee_contribution)) / boss_contribution = 2 := by
  sorry

end todds_contribution_ratio_l1813_181375


namespace domino_coloring_properties_l1813_181309

/-- Definition of the number of possible colorings for a domino of length n -/
def A (n : ℕ) : ℕ := 2^n

/-- Definition of the number of valid colorings (no adjacent painted squares) for a domino of length n -/
def F : ℕ → ℕ
  | 0 => 1  -- Base case for convenience
  | 1 => 2
  | 2 => 3
  | (n+3) => F (n+2) + F (n+1)

theorem domino_coloring_properties :
  (∀ n : ℕ, A n = 2^n) ∧
  F 1 = 2 ∧ F 2 = 3 ∧ F 3 = 5 ∧ F 4 = 8 ∧
  (∀ n : ℕ, n ≥ 3 → F n = F (n-1) + F (n-2)) ∧
  (∀ n p : ℕ+, F (n + p + 1) = F n * F p + F (n-1) * F (p-1)) := by
  sorry

end domino_coloring_properties_l1813_181309


namespace calculation_proof_l1813_181318

theorem calculation_proof :
  ((-1 : ℝ)^2023 + |(-3 : ℝ)| - (π - 7)^0 + 2^4 * (1/2 : ℝ)^4 = 2) ∧
  (∀ (a b : ℝ), 6*a^3*b^2 / (3*a^2*b^2) + (2*a*b^3)^2 / (a*b)^2 = 2*a + 4*b^4) :=
by sorry

end calculation_proof_l1813_181318


namespace frankie_candy_count_l1813_181382

theorem frankie_candy_count (max_candy : ℕ) (extra_candy : ℕ) (frankie_candy : ℕ) : 
  max_candy = 92 → 
  extra_candy = 18 → 
  max_candy = frankie_candy + extra_candy → 
  frankie_candy = 74 := by
sorry

end frankie_candy_count_l1813_181382


namespace equation_roots_l1813_181364

-- Define the equation
def equation (x : ℝ) : Prop :=
  (3*x^2 + 1)/(x-2) - (3*x+8)/4 + (5-9*x)/(x-2) + 2 = 0

-- Define the roots
def root1 : ℝ := 3.29
def root2 : ℝ := -0.40

-- Theorem statement
theorem equation_roots :
  ∃ (ε : ℝ), ε > 0 ∧ 
  (∀ (x : ℝ), equation x → (|x - root1| < ε ∨ |x - root2| < ε)) :=
sorry

end equation_roots_l1813_181364


namespace derivative_at_three_l1813_181329

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_at_three : 
  (deriv f) 3 = 6 := by sorry

end derivative_at_three_l1813_181329


namespace calculate_expressions_l1813_181315

theorem calculate_expressions : 
  (1 - 1^2 + (64 : ℝ)^(1/3) - (-2) * (9 : ℝ)^(1/2) = 9) ∧ 
  ((-1/2 : ℝ) * (-2)^2 - (-1/8 : ℝ)^(1/3) + ((-1/2 : ℝ)^2)^(1/2) = -1) := by
  sorry

end calculate_expressions_l1813_181315


namespace mary_took_three_crayons_l1813_181342

/-- Given an initial number of crayons and the number left after some are taken,
    calculate the number of crayons taken. -/
def crayons_taken (initial : ℕ) (left : ℕ) : ℕ := initial - left

theorem mary_took_three_crayons :
  let initial_crayons : ℕ := 7
  let crayons_left : ℕ := 4
  crayons_taken initial_crayons crayons_left = 3 := by
sorry

end mary_took_three_crayons_l1813_181342


namespace solution_set_and_range_l1813_181399

def f (x : ℝ) : ℝ := |x + 1|

theorem solution_set_and_range :
  (∀ x : ℝ, f x ≤ 5 - f (x - 3) ↔ -2 ≤ x ∧ x ≤ 3) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * f x + |x + a| ≤ x + 4 → -2 ≤ a ∧ a ≤ 4) :=
sorry

end solution_set_and_range_l1813_181399


namespace journey_speed_calculation_l1813_181388

/-- 
Given a journey with total distance D and total time T, 
where a person travels 2/3 of D in 1/3 of T at 40 km/h,
prove that they must travel at 10 km/h for the remaining distance
to reach the destination on time.
-/
theorem journey_speed_calculation 
  (D T : ℝ) 
  (h1 : D > 0) 
  (h2 : T > 0) 
  (h3 : (2/3 * D) / (1/3 * T) = 40) : 
  (1/3 * D) / (2/3 * T) = 10 := by
sorry

end journey_speed_calculation_l1813_181388


namespace determinant_sin_matrix_l1813_181368

theorem determinant_sin_matrix (a b : Real) : 
  Matrix.det !![1, Real.sin (a - b), Real.sin a; 
                 Real.sin (a - b), 1, Real.sin b; 
                 Real.sin a, Real.sin b, 1] = 0 := by
  sorry

end determinant_sin_matrix_l1813_181368


namespace cube_root_two_identity_l1813_181361

theorem cube_root_two_identity (x : ℝ) (h : 32 = x^6 + 1/x^6) :
  x^2 + 1/x^2 = 2 * Real.rpow 2 (1/3) := by
  sorry

end cube_root_two_identity_l1813_181361


namespace roberto_chicken_investment_break_even_l1813_181357

/-- Represents Roberto's chicken investment scenario -/
structure ChickenInvestment where
  num_chickens : ℕ
  cost_per_chicken : ℕ
  weekly_feed_cost : ℕ
  eggs_per_chicken_per_week : ℕ
  previous_dozen_cost : ℕ

/-- Calculates the break-even point in weeks for the chicken investment -/
def break_even_point (ci : ChickenInvestment) : ℕ :=
  let initial_cost := ci.num_chickens * ci.cost_per_chicken
  let weekly_egg_production := ci.num_chickens * ci.eggs_per_chicken_per_week
  let weekly_savings := ci.previous_dozen_cost - ci.weekly_feed_cost
  initial_cost / weekly_savings + 1

/-- Theorem stating that Roberto's chicken investment breaks even after 81 weeks -/
theorem roberto_chicken_investment_break_even :
  let ci : ChickenInvestment := {
    num_chickens := 4,
    cost_per_chicken := 20,
    weekly_feed_cost := 1,
    eggs_per_chicken_per_week := 3,
    previous_dozen_cost := 2
  }
  break_even_point ci = 81 := by sorry

end roberto_chicken_investment_break_even_l1813_181357


namespace seahawks_touchdowns_l1813_181338

theorem seahawks_touchdowns (final_score : ℕ) (field_goals : ℕ) 
  (h1 : final_score = 37)
  (h2 : field_goals = 3) :
  (final_score - field_goals * 3) / 7 = 4 := by
  sorry

#check seahawks_touchdowns

end seahawks_touchdowns_l1813_181338


namespace carrots_equal_fifteen_l1813_181322

/-- The price relationship between apples, bananas, and carrots -/
structure FruitPrices where
  apple_banana_ratio : ℚ
  banana_carrot_ratio : ℚ
  apple_banana_eq : apple_banana_ratio = 10 / 5
  banana_carrot_eq : banana_carrot_ratio = 2 / 5

/-- The number of carrots that can be bought for the price of 12 apples -/
def carrots_for_apples (prices : FruitPrices) : ℚ :=
  12 * (prices.banana_carrot_ratio / prices.apple_banana_ratio)

theorem carrots_equal_fifteen (prices : FruitPrices) :
  carrots_for_apples prices = 15 := by
  sorry

end carrots_equal_fifteen_l1813_181322


namespace multinomial_expansion_terms_l1813_181379

/-- The number of terms in the simplified multinomial expansion of (x+y+z)^10 -/
def multinomial_terms : ℕ := 66

/-- Theorem stating that the number of terms in the simplified multinomial expansion of (x+y+z)^10 is 66 -/
theorem multinomial_expansion_terms :
  multinomial_terms = 66 := by sorry

end multinomial_expansion_terms_l1813_181379


namespace xy_values_l1813_181395

theorem xy_values (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 62) :
  x * y = -126/25 ∨ x * y = -6 := by
  sorry

end xy_values_l1813_181395


namespace odd_function_properties_l1813_181340

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an increasing function on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Define the minimum value of a function on an interval
def HasMinValueOn (f : ℝ → ℝ) (v : ℝ) (a b : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → v ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = v)

-- Define the maximum value of a function on an interval
def HasMaxValueOn (f : ℝ → ℝ) (v : ℝ) (a b : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ v) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = v)

-- Theorem statement
theorem odd_function_properties (f : ℝ → ℝ) :
  OddFunction f →
  IncreasingOn f 1 3 →
  HasMinValueOn f 0 1 3 →
  IncreasingOn f (-3) (-1) ∧ HasMaxValueOn f 0 (-3) (-1) :=
by sorry

end odd_function_properties_l1813_181340


namespace point_m_property_l1813_181393

theorem point_m_property (a : ℝ) : 
  let m : ℝ × ℝ := (3*a - 9, 10 - 2*a)
  (m.1 < 0 ∧ m.2 > 0) →  -- M is in the second quadrant
  (|m.1| = |m.2|) →      -- Distance to y-axis equals distance to x-axis
  (a + 2)^2023 - 1 = 0   -- The expression equals 0
  := by sorry

end point_m_property_l1813_181393


namespace disjunction_true_l1813_181366

def p : Prop := ∃ k : ℕ, 2 = 2 * k

def q : Prop := ∃ k : ℕ, 3 = 2 * k

theorem disjunction_true : p ∨ q := by
  sorry

end disjunction_true_l1813_181366


namespace min_x_prime_factorization_l1813_181369

theorem min_x_prime_factorization (x y : ℕ+) (h : 5 * x^7 = 13 * y^17) :
  ∃ (a b c d : ℕ),
    x.val = a^c * b^d ∧
    a.Prime ∧ b.Prime ∧
    x.val ≥ 13^5 * 5^10 ∧
    (x.val = 13^5 * 5^10 → a + b + c + d = 33) :=
sorry

end min_x_prime_factorization_l1813_181369


namespace rectangle_composition_l1813_181343

theorem rectangle_composition (total_width total_height : ℕ) 
  (h_width : total_width = 3322) (h_height : total_height = 2020) : ∃ (r s : ℕ),
  2 * r + s = total_height ∧ 2 * r + 3 * s = total_width ∧ s = 651 := by
  sorry

end rectangle_composition_l1813_181343


namespace cubic_polynomial_q_value_l1813_181316

/-- A cubic polynomial Q(x) = x^3 + px^2 + qx + d -/
def cubicPolynomial (p q d : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + d

theorem cubic_polynomial_q_value 
  (p q d : ℝ) 
  (h1 : -(p/3) = q) -- mean of zeros equals product of zeros taken two at a time
  (h2 : q = 1 + p + q + d) -- product of zeros taken two at a time equals sum of coefficients
  (h3 : d = 7) -- y-intercept is 7
  : q = 8/3 := by
sorry

end cubic_polynomial_q_value_l1813_181316


namespace numbers_with_five_in_range_l1813_181314

def count_numbers_with_five (n : ℕ) : ℕ :=
  n - (6 * 9 * 9)

theorem numbers_with_five_in_range :
  count_numbers_with_five 700 = 214 := by
  sorry

end numbers_with_five_in_range_l1813_181314


namespace half_jar_days_l1813_181376

/-- Represents the area of kombucha in the jar as a function of time -/
def kombucha_area (t : ℕ) : ℝ := 2^t

/-- The number of days it takes to fill the entire jar -/
def full_jar_days : ℕ := 17

theorem half_jar_days : 
  (kombucha_area full_jar_days = 2 * kombucha_area (full_jar_days - 1)) → 
  (kombucha_area (full_jar_days - 1) = (1/2) * kombucha_area full_jar_days) := by
  sorry

end half_jar_days_l1813_181376


namespace factorization_equality_l1813_181306

theorem factorization_equality (a b : ℝ) : a * b^2 - 2 * a * b + a = a * (b - 1)^2 := by
  sorry

end factorization_equality_l1813_181306


namespace distribute_5_balls_4_boxes_l1813_181359

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 68 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_5_balls_4_boxes : distribute_balls 5 4 = 68 := by sorry

end distribute_5_balls_4_boxes_l1813_181359


namespace functional_equation_problem_l1813_181312

/-- The functional equation problem -/
theorem functional_equation_problem (α : ℝ) (hα : α ≠ 0) :
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x^2 + y + f y) = (f x)^2 + α * y) ↔ α = 2 :=
by sorry

end functional_equation_problem_l1813_181312


namespace simplify_square_roots_l1813_181360

theorem simplify_square_roots : Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^3) = 225 := by
  sorry

end simplify_square_roots_l1813_181360


namespace exists_x_squared_minus_one_nonnegative_l1813_181380

theorem exists_x_squared_minus_one_nonnegative :
  ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ x^2 - 1 ≥ 0 := by
  sorry

end exists_x_squared_minus_one_nonnegative_l1813_181380


namespace intersection_difference_is_zero_l1813_181325

noncomputable def f (x : ℝ) : ℝ := 2 - x^3 + x^4
noncomputable def g (x : ℝ) : ℝ := 1 + 2*x^3 + x^4

theorem intersection_difference_is_zero :
  ∀ x y : ℝ, f x = g x → f y = g y → |f x - g y| = 0 :=
by sorry

end intersection_difference_is_zero_l1813_181325


namespace inequality_relation_l1813_181377

theorem inequality_relation (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^3 := by
  sorry

end inequality_relation_l1813_181377


namespace quadratic_roots_and_inequality_l1813_181326

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop := x^2 + a*x + a - 1/2 = 0

-- Define the set of possible values for a
def valid_a_set : Set ℝ := {a | a ≤ 2 - Real.sqrt 2 ∨ a ≥ 2 + Real.sqrt 2}

-- Define the inequality
def inequality (m t a x₁ x₂ : ℝ) : Prop :=
  m^2 + t*m + 4*Real.sqrt 2 + 6 ≥ (x₁ - 3*x₂)*(x₂ - 3*x₁)

theorem quadratic_roots_and_inequality :
  ∀ a ∈ valid_a_set,
  ∀ x₁ x₂ : ℝ,
  quadratic_equation a x₁ ∧ quadratic_equation a x₂ →
  (∀ t ∈ Set.Icc (-1 : ℝ) 1,
    ∃ m : ℝ, inequality m t a x₁ x₂) ↔
  ∃ m : ℝ, m ≤ -1 ∨ m = 0 ∨ m ≥ 1 :=
sorry

end quadratic_roots_and_inequality_l1813_181326


namespace first_sales_amount_l1813_181332

/-- The amount of the first sales in millions of dollars -/
def first_sales : ℝ := sorry

/-- The profit on the first sales in millions of dollars -/
def first_profit : ℝ := 5

/-- The profit on the next $30 million in sales in millions of dollars -/
def second_profit : ℝ := 12

/-- The amount of the second sales in millions of dollars -/
def second_sales : ℝ := 30

/-- The increase in profit ratio from the first to the second sales -/
def profit_ratio_increase : ℝ := 0.2000000000000001

theorem first_sales_amount :
  (first_profit / first_sales) * (1 + profit_ratio_increase) = second_profit / second_sales ∧
  first_sales = 15 := by
  sorry

end first_sales_amount_l1813_181332
